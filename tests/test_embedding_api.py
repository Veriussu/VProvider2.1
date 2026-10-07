# ─────────────────────────────────────────────────────────────
#  Bölüm:    /v1/embeddings Uç Noktası Testleri
#  Dosya:    tests/test_embedding_api.py
#  Amaç:     OpenAI uyumlu embedding uç noktasını (yetkilendirme, model
#            çözümü, biçim, ücretlendirme) sahte arka uçla doğrular.
#  Mekanik:  - Gerçek sentence-transformers kurulmaz: embedding_backend
#            tekili testte sahte bir kodlayıcı ile değiştirilir.
#            - models/embeddings/ altına geçici bir model dizini kurulur.
#            - Kullanım kaydı (usage_log) üzerinden kredi düşümü doğrulanır.
# ─────────────────────────────────────────────────────────────

import json

import pytest
from fastapi.testclient import TestClient

from app import auth, embedding_backend, main as main_mod, openai_api as openai_api_mod
from app.gateway import hash_key
from app.model_manager import ModelManager
from app.user_store import UserStore

API_KEY = "test-embed-anahtar"
EMBED_MODEL = "all-MiniLM-L6-v2"


class _Vectors(list):
    """numpy benzeri dizi: .tolist() ile iç içe listeye dönüşür."""

    def tolist(self):
        return [list(row) for row in self]


class FakeEncoder:
    """Gerçek model yüklemeden vektör üreten sahte SentenceTransformer."""

    def __init__(self, path):
        self.path = path
        self.calls = 0

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, normalize_embeddings=True):
        self.calls += 1
        return _Vectors([[float(len(t)), 1.0, 0.0, 0.0] for t in texts])


def _install_model(models_dir):
    """models/embeddings/<model>/ altına geçici bir embedding modeli kurar."""
    d = models_dir / "embeddings" / EMBED_MODEL
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": "bert"}), "utf-8")
    (d / "model.safetensors").write_bytes(b"x")
    return d


def _make_client(tmp_path, monkeypatch, *, available=True):
    """API anahtarı, sahte kodlayıcı ve sahte model yöneticisi kurar."""
    store = UserStore(tmp_path / "test.db")
    store.init()
    store.create_api_key("Test", API_KEY)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)

    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "gguf-model.gguf").write_bytes(b"data")
    _install_model(models_dir)

    # Registry'nin bu testin geçici dizinini görmesi için ayarlar yönlendirilir
    monkeypatch.setattr(
        embedding_backend.settings, "models_dir", str(models_dir), raising=False
    )
    monkeypatch.setattr(embedding_backend.settings, "embedding_model", "", raising=False)

    manager = ModelManager(models_dir=models_dir, memory_mode="keep")
    monkeypatch.setattr(openai_api_mod, "get_manager", lambda: manager)

    encoder = FakeEncoder(str(models_dir / "embeddings" / EMBED_MODEL))
    embedding_backend._reset()
    monkeypatch.setattr(
        embedding_backend.EmbeddingBackend, "is_available",
        property(lambda self: available),
    )
    monkeypatch.setattr(
        embedding_backend.EmbeddingBackend, "_load", lambda self: encoder
    )
    return TestClient(main_mod.app), store, encoder


@pytest.fixture()
def client(tmp_path, monkeypatch):
    client, _store, encoder = _make_client(tmp_path, monkeypatch)
    yield client
    embedding_backend._reset()


def _auth():
    return {"Authorization": f"Bearer {API_KEY}"}


# ------------------------------------------------------------------
# Yetkilendirme
# ------------------------------------------------------------------

def test_embeddings_requires_api_key(client):
    """Anahtarsız istek 401 döner."""
    assert client.post("/v1/embeddings", json={"model": EMBED_MODEL, "input": "x"}).status_code == 401


def test_embeddings_invalid_key(client):
    """Geçersiz anahtar 401 döner."""
    r = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer yanlis"},
        json={"model": EMBED_MODEL, "input": "x"},
    )
    assert r.status_code == 401


# ------------------------------------------------------------------
# Model çözümü ve biçim
# ------------------------------------------------------------------

def test_embeddings_unknown_model_404(client):
    """Kurulu olmayan model OpenAI biçiminde 404 döner."""
    r = client.post("/v1/embeddings", headers=_auth(), json={"model": "yok", "input": "x"})
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_embeddings_returns_openai_shape(client):
    """Tek metin girdisi OpenAI biçiminde vektör döndürür."""
    r = client.post("/v1/embeddings", headers=_auth(), json={"model": EMBED_MODEL, "input": "merhaba"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == EMBED_MODEL
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert body["data"][0]["embedding"] == [7.0, 1.0, 0.0, 0.0]
    assert body["usage"]["prompt_tokens"] >= 1
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"]


def test_embeddings_accepts_list_input(client):
    """Metin listesi sırayla vektörlenir."""
    r = client.post(
        "/v1/embeddings",
        headers=_auth(),
        json={"model": EMBED_MODEL, "input": ["ab", "abcd"]},
    )
    assert r.status_code == 200
    data = r.json()["data"]
    assert [d["index"] for d in data] == [0, 1]
    assert data[0]["embedding"][0] == 2.0
    assert data[1]["embedding"][0] == 4.0


def test_embeddings_empty_input_400(client):
    """Boş girdi 400 döner."""
    r = client.post("/v1/embeddings", headers=_auth(), json={"model": EMBED_MODEL, "input": []})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_input"


def test_embeddings_unavailable_backend_400(tmp_path, monkeypatch):
    """Paket/hub kapalıyken 400 (embeddings_not_available) döner."""
    client, _store, _enc = _make_client(tmp_path, monkeypatch, available=False)
    r = client.post("/v1/embeddings", headers=_auth(), json={"model": EMBED_MODEL, "input": "x"})
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "embeddings_not_available"
    embedding_backend._reset()


# ------------------------------------------------------------------
# Ücretlendirme
# ------------------------------------------------------------------

def test_embeddings_records_usage_and_cost(tmp_path, monkeypatch):
    """Başarılı istek kullanım kaydeder ve bakiyeden kredi düşer."""
    client, store, _enc = _make_client(tmp_path, monkeypatch)
    key_id = store.get_api_key_by_hash(hash_key(API_KEY))["id"]
    before = store.usage_totals(key_id)["cost"]

    r = client.post("/v1/embeddings", headers=_auth(), json={"model": EMBED_MODEL, "input": "merhaba"})

    assert r.status_code == 200
    tokens = r.json()["usage"]["prompt_tokens"]
    totals = store.usage_totals(key_id)
    assert totals["prompt_tokens"] == tokens
    # embedding tarifesi: 0.0001 kredi / 1K token
    assert totals["cost"] - before == pytest.approx(tokens / 1000 * 0.0001, abs=1e-9)
    embedding_backend._reset()


def test_embeddings_model_allowlist_enforced(tmp_path, monkeypatch):
    """Anahtarın izin listesi embedding için de uygulanır (403)."""
    client, store, _enc = _make_client(tmp_path, monkeypatch)
    store.update_api_key(store.get_api_key_by_hash(hash_key(API_KEY))["id"], model_allowlist=["başka"])
    r = client.post("/v1/embeddings", headers=_auth(), json={"model": EMBED_MODEL, "input": "x"})
    assert r.status_code == 403
    assert r.json()["error"]["code"] == "model_not_allowed"
    embedding_backend._reset()
