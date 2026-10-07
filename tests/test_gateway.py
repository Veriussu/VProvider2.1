# ─────────────────────────────────────────────────────────────
#  Bölüm:    API Gateway Testleri
#  Dosya:    tests/test_gateway.py
#  Amaç:     Ort/API anahtarının yaşam döngüsü, limitler, kota,
#            bakiye ve kullanım kaydını gerçek DB ile doğrular.
#  Not:      Veritabanı geçicidir; gerçek verilere dokunulmaz.
# ─────────────────────────────────────────────────────────────

import datetime
import hashlib

import pytest
from fastapi import HTTPException

import app.auth as auth_mod
import app.main as main_mod
import app.gateway as gateway_mod
from app.gateway import ApiContext, authenticate, generate_key, hash_key
from app.model_manager import ModelManager
from app.user_store import UserStore


class _FakeEngine:
    def __init__(self, path):
        self.path = path

    def load(self):
        pass

    def unload(self):
        pass

    @property
    def is_loaded(self):
        return True


def _make_env(tmp_path, monkeypatch, **keyopts):
    """Geçici depo + anahtar + sahte motor; (store, raw_key, ctx) döner."""
    store = UserStore(tmp_path / "gw.db")
    store.init()
    raw, prefix, h = generate_key()
    store.create_api_key_with_options(name=keyopts.pop("name", "Gateway"), key_hash=h, prefix=prefix, **keyopts)
    monkeypatch.setattr(auth_mod, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "gpt-test.gguf").write_bytes(b"data")
    manager = ModelManager(
        models_dir=models_dir,
        engine_factory=lambda info: _FakeEngine(str(info.path)),
        memory_mode="keep",
    )
    return store, raw


# ------------------------------------------------------------------
# Anahtar üretimi / hash
# ------------------------------------------------------------------

def test_generate_key_shape():
    raw, prefix, h = generate_key()
    assert len(raw) == 64
    assert prefix == raw[:12]
    assert h == hashlib.sha256(raw.encode()).hexdigest()


def test_hash_key_deterministic():
    assert hash_key("abc") == hashlib.sha256(b"abc").hexdigest()


# ------------------------------------------------------------------
# Kimlik doğrulama: temel durumlar
# ------------------------------------------------------------------

def test_authenticate_valid(tmp_path, monkeypatch):
    store, raw = _make_env(tmp_path, monkeypatch)
    ctx = authenticate(raw)
    assert isinstance(ctx, ApiContext)
    assert ctx.key_id > 0


def test_authenticate_legacy_plaintext_key(tmp_path, monkeypatch):
    """create_api_key (miras) ile eklenen anahtar da gateway'den geçer."""
    store = UserStore(tmp_path / "gw.db"); store.init()
    store.create_api_key("Mir", "miras-anahtar")
    monkeypatch.setattr(auth_mod, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)
    assert authenticate("miras-anahtar").key_id > 0


def test_authenticate_invalid(tmp_path, monkeypatch):
    _make_env(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as ei:
        authenticate("olmayan-anahtar")
    assert ei.value.status_code == 401
    assert ei.value.detail["error"]["code"] == "invalid_api_key"


def test_authenticate_missing_key(tmp_path, monkeypatch):
    _make_env(tmp_path, monkeypatch)
    with pytest.raises(HTTPException) as ei:
        authenticate("")
    assert ei.value.status_code == 401
    assert ei.value.detail["error"]["code"] == "missing_api_key"


def test_authenticate_revoked(tmp_path, monkeypatch):
    store, raw = _make_env(tmp_path, monkeypatch)
    store.get_api_key_by_hash(hash_key(raw))
    store.set_api_key_revoked(store.get_api_key_by_hash(hash_key(raw))["id"], True)
    with pytest.raises(HTTPException) as ei:
        authenticate(raw)
    assert ei.value.status_code == 403
    assert ei.value.detail["error"]["code"] == "api_key_revoked"


def test_authenticate_expired(tmp_path, monkeypatch):
    past = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=1)).isoformat()
    _store, raw = _make_env(tmp_path, monkeypatch, expires_at=past)
    with pytest.raises(HTTPException) as ei:
        authenticate(raw)
    assert ei.value.status_code == 401
    assert ei.value.detail["error"]["code"] == "api_key_expired"


# ------------------------------------------------------------------
# Model allowlist + RPM + kota + bakiye
# ------------------------------------------------------------------

def test_enforce_model_allowlist_blocks(tmp_path, monkeypatch):
    _, raw = _make_env(tmp_path, monkeypatch, model_allowlist=["gpt-test"])
    ctx = authenticate(raw)
    with pytest.raises(HTTPException) as ei:
        ctx.charge("chat", model_id="bash-model", prompt_tokens=10)
    assert ei.value.status_code == 403
    assert ei.value.detail["error"]["code"] == "model_not_allowed"


def test_enforce_model_allowlist_allows(tmp_path, monkeypatch):
    _, raw = _make_env(tmp_path, monkeypatch, model_allowlist=["gpt-test"])
    ctx = authenticate(raw)
    ctx.charge("chat", model_id="gpt-test", prompt_tokens=10)


def test_rpm_limit_429(tmp_path, monkeypatch):
    _, raw = _make_env(tmp_path, monkeypatch, rate_limit_rpm=2)
    assert authenticate(raw).key_id > 0
    assert authenticate(raw).key_id > 0
    with pytest.raises(HTTPException) as ei:
        authenticate(raw)
    assert ei.value.status_code == 429
    assert ei.value.detail["error"]["code"] == "rate_limit_exceeded"


def test_tpm_limit_429(tmp_path, monkeypatch):
    _, raw = _make_env(tmp_path, monkeypatch, rate_limit_tpm=100)
    ctx = authenticate(raw)
    ctx.charge("chat", prompt_tokens=100)  # pencere dolar
    with pytest.raises(HTTPException) as ei:
        ctx.charge("chat", prompt_tokens=10)
    assert ei.value.status_code == 429


def test_monthly_quota_blocks(tmp_path, monkeypatch):
    store, raw = _make_env(tmp_path, monkeypatch, monthly_token_quota=100)
    key_id = store.get_api_key_by_hash(hash_key(raw))["id"]
    today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
    store.record_usage(key_id, today, prompt_tokens=100)
    with pytest.raises(HTTPException) as ei:
        authenticate(raw)
    assert ei.value.status_code == 402
    assert ei.value.detail["error"]["code"] == "monthly_quota_exceeded"


def test_balance_debt_blocks(tmp_path, monkeypatch):
    _store, raw = _make_env(tmp_path, monkeypatch, balance_credits=-1)
    with pytest.raises(HTTPException) as ei:
        authenticate(raw)
    assert ei.value.status_code == 402
    assert ei.value.detail["error"]["code"] == "insufficient_credits"


# ------------------------------------------------------------------
# Ücretlendirme: bakiye düşümü + kullanım kaydı
# ------------------------------------------------------------------

def test_charge_records_usage_and_deducts(tmp_path, monkeypatch):
    store, raw = _make_env(tmp_path, monkeypatch, balance_credits=10.0)
    key_id = store.get_api_key_by_hash(hash_key(raw))["id"]
    ctx = authenticate(raw)
    cost = ctx.charge("chat", prompt_tokens=1000, completion_tokens=500)
    assert cost == pytest.approx(0.003, abs=1e-6)  # 1500 tok * 0.002 / 1k
    rec = store.get_api_key_record(key_id)
    assert rec["balance_credits"] == pytest.approx(10.0 - cost, abs=1e-6)
    u = store.usage_totals(key_id)
    assert u["prompt_tokens"] == 1000
    assert u["completion_tokens"] == 500
    assert u["requests"] == 1  # doğrulama anında sayıldı
    assert u["cost"] == pytest.approx(cost, abs=1e-6)


def test_charge_unlimited_balance_stays_zero(tmp_path, monkeypatch):
    """Bakiye 0 olan (sınırsız) anahtardan asla düşüm yapılmaz."""
    store, raw = _make_env(tmp_path, monkeypatch, balance_credits=0.0)
    key_id = store.get_api_key_by_hash(hash_key(raw))["id"]
    ctx = authenticate(raw)
    ctx.charge("chat", prompt_tokens=99999)
    assert store.get_api_key_record(key_id)["balance_credits"] == 0.0


def test_charge_exhausted_goes_debt(tmp_path, monkeypatch):
    """Prepaid bakiye bittiğinde anahtar borç haliyle kilitlenir (sonraki istek 402)."""
    store, raw = _make_env(tmp_path, monkeypatch, balance_credits=0.01)
    key_id = store.get_api_key_by_hash(hash_key(raw))["id"]
    ctx = authenticate(raw)
    ctx.charge("chat", prompt_tokens=10000)  # maliyet > bakiye
    assert store.get_api_key_record(key_id)["balance_credits"] < 0
    with pytest.raises(HTTPException) as ei:
        authenticate(raw)
    assert ei.value.status_code == 402


def test_media_charge_records_separate_counters(tmp_path, monkeypatch):
    store, raw = _make_env(tmp_path, monkeypatch)
    key_id = store.get_api_key_by_hash(hash_key(raw))["id"]
    ctx = authenticate(raw)
    ctx.charge("image", images=2)
    ctx.charge("video", videos=1)
    ctx.charge("tts", audio_sec=3.5)
    u = store.usage_totals(key_id)
    assert u["images"] == 2
    assert u["videos"] == 1
    assert u["audio_sec"] == pytest.approx(3.5, abs=1e-6)


# ------------------------------------------------------------------
# HTTP katmanı (Bearer başlığı)
# ------------------------------------------------------------------

def test_api_auth_header_parsing(tmp_path, monkeypatch):
    _store, raw = _make_env(tmp_path, monkeypatch)
    ctx = gateway_mod.api_auth(authorization=f"Bearer {raw}")
    assert ctx.key_id > 0
    with pytest.raises(HTTPException) as ei:
        gateway_mod.api_auth(authorization=None)
    assert ei.value.status_code == 401