# ─────────────────────────────────────────────────────────────
#  Bölüm:    Reasoning (Düşünme) Modeli Testleri
#  Dosya:    tests/test_reasoning.py
#  Amaç:     Reasoning modellerinin /v1/models'te "pricing.task" olarak
#            işaretlenmesini ve "reasoning" tarifesiyle ücretlenmesini doğrular.
#  Mekanik:  - models/reasoning/ altına GGUF konur (klasör sözleşmesi) ve
#              köke konan "R1" adlı model (ad deseni) karşılaştırılır.
#            - Sahte motor (FakeEngine) gerçek llama.cpp gerektirmez.
#            - Kredi farkı, tarife farkından (0.004 vs 0.002 / 1K token)
#              doğrulanır.
# ─────────────────────────────────────────────────────────────

import pytest
from fastapi.testclient import TestClient

from app import auth, main as main_mod, openai_api as openai_api_mod
from app.gateway import hash_key
from app.model_manager import ModelManager
from app.user_store import UserStore

API_KEY = "test-reasoning-anahtar"
CHAT_MODEL = "sade-model"
REASONING_FOLDER_MODEL = "derin-dusunucu"
REASONING_NAME_MODEL = "ornek-R1-Model"


class FakeEngine:
    """Sabit kullanım bilgisi üreten sahte motor (token: 5 + 8)."""

    def __init__(self, path):
        self.path = path
        self._loaded = False

    def load(self):
        self._loaded = True

    def unload(self):
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    def run_chat(self, messages, **params):
        return "yanit"

    def usage_info(self):
        return {"prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """Reasoning + chat modelleri bulunan bir API istemcisi kurar."""
    store = UserStore(tmp_path / "test.db")
    store.init()
    store.create_api_key("Test", API_KEY)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / f"{CHAT_MODEL}.gguf").write_bytes(b"data")
    (models_dir / f"{REASONING_NAME_MODEL}.gguf").write_bytes(b"data")
    reasoning_dir = models_dir / "reasoning"
    reasoning_dir.mkdir()
    (reasoning_dir / f"{REASONING_FOLDER_MODEL}.gguf").write_bytes(b"data")

    manager = ModelManager(
        models_dir=models_dir,
        engine_factory=lambda info: FakeEngine(str(info.path)),
        memory_mode="keep",
    )
    monkeypatch.setattr(openai_api_mod, "get_manager", lambda: manager)
    return TestClient(main_mod.app), store


def _auth():
    return {"Authorization": f"Bearer {API_KEY}"}


# ------------------------------------------------------------------
# /v1/models işaretlemesi
# ------------------------------------------------------------------

def test_models_marks_reasoning_category(client):
    """Klasör ve ad deseniyle reasoning modelleri işaretlenir."""
    api, _store = client
    body = api.get("/v1/models", headers=_auth()).json()
    tasks = {m["id"]: m["pricing"]["task"] for m in body["data"]}

    assert tasks[CHAT_MODEL] == "chat"
    assert tasks[REASONING_FOLDER_MODEL] == "reasoning"
    assert tasks[REASONING_NAME_MODEL] == "reasoning"


# ------------------------------------------------------------------
# Ücretlendirme farkı
# ------------------------------------------------------------------

def _charge(client, store, model_id):
    """Belirtilen modelle chat çağrısı yapar ve harcanan krediyi döner."""
    api, _ = client
    key_id = store.get_api_key_by_hash(hash_key(API_KEY))["id"]
    before = store.usage_totals(key_id)["cost"]
    r = api.post(
        "/v1/chat/completions",
        headers=_auth(),
        json={"model": model_id, "messages": [{"role": "user", "content": "merhaba"}]},
    )
    assert r.status_code == 200
    return store.usage_totals(key_id)["cost"] - before


def test_reasoning_model_charged_at_reasoning_price(client):
    """Reasoning model 13 token'da 0.004/1K, chat model 0.002/1K kredi düşer."""
    _api, store = client

    reasoning_cost = _charge(client, store, REASONING_FOLDER_MODEL)
    chat_cost = _charge(client, store, CHAT_MODEL)

    assert reasoning_cost == pytest.approx(13 / 1000 * 0.004, abs=1e-9)
    assert chat_cost == pytest.approx(13 / 1000 * 0.002, abs=1e-9)
    assert reasoning_cost == pytest.approx(chat_cost * 2, abs=1e-9)


def test_reasoning_detection_via_name_pattern(client):
    """models/reasoning/ altında olmayan ancak adında R1 geçen model de reasoning."""
    _api, store = client
    cost = _charge(client, store, REASONING_NAME_MODEL)
    assert cost == pytest.approx(13 / 1000 * 0.004, abs=1e-9)
