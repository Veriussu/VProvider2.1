# ─────────────────────────────────────────────────────────────
#  Bölüm:    Görsel/Video API Testleri (in-process + ComfyUI)
#  Dosya:    tests/test_media_api.py
#  Amaç:     /v1/images/* ve /v1/videos/* uç noktalarının iki motorlu
#            (yerel difüzör / ComfyUI köprüsü) yönlendirmesini doğrular.
#  Mekanik:  - Sahte MedyaManager takılır (diffusers yüklenmez).
#            - ComfyUI köprüsü sahte client ile doğrulanır (geriye uyum).
#            - Ücretlendirme: image -> per_image, video -> per_video.
# ─────────────────────────────────────────────────────────────

import base64
import json

import pytest
from fastapi.testclient import TestClient

from app import comfy_api, comfy_client
from app import main as main_mod
from app import auth
from app.diffusers_backend import MediaError, MediaInfo, MediaResult
from app.user_store import UserStore

API_KEY = "medya-test-anahtar"


class FakeMediaManager:
    """Gerçek diffusers yerine kullanılan sahte medya yöneticisi."""

    def __init__(self, models: list[MediaInfo]):
        self.models = models
        self.calls = []

    def list_models(self):
        return list(self.models)

    def get_model(self, model_id):
        return next((m for m in self.models if m.model_id == model_id), None)

    async def generate(self, model_id, prompt, **kw):
        self.calls.append({"model_id": model_id, "prompt": prompt, **kw})
        info = self.get_model(model_id)
        if info is None:
            raise MediaError(f"'{model_id}' modeli bulunamadı.")
        if info.category == "video":
            payload = {
                "bytes": b"GIF89a-fake",
                "mime": "image/gif",
                "width": kw.get("size", "512x512").split("x")[1] + "",
                "height": kw.get("size", "512x512").split("x")[0],
                "frames": kw.get("frames", 2),
            }
            return MediaResult(model_id=model_id, video=payload)
        return MediaResult(model_id=model_id, images=[b"\x89PNG-fake"] * max(1, kw.get("n", 1)))


def _info(model_id, category):
    from pathlib import Path

    return MediaInfo(model_id=model_id, path=Path("/tmp") / model_id, category=category)


def _make_client(tmp_path, monkeypatch, models=None):
    """API anahtarı + sahte medya yöneticisi ile istemci kurar."""
    store = UserStore(tmp_path / "test.db")
    store.init()
    store.create_api_key("Test", API_KEY)
    monkeypatch.setattr(auth, "get_store", lambda: store)
    monkeypatch.setattr(main_mod, "get_store", lambda: store)

    if models is None:
        models = [_info("yerel-gorsel", "image"), _info("yerel-video", "video")]
    manager = FakeMediaManager(models)
    monkeypatch.setattr(comfy_api.diffusers_backend, "get_media_manager", lambda: manager)
    return TestClient(main_mod.app), store, manager


def _auth():
    return {"Authorization": f"Bearer {API_KEY}"}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    client, store, manager = _make_client(tmp_path, monkeypatch)
    yield client, store, manager


# ------------------------------------------------------------------
# Yetkilendirme ve model listeleri
# ------------------------------------------------------------------

def test_endpoints_require_api_key(client):
    """Tüm medya uç noktaları API anahtarı ister."""
    api, _s, _m = client
    assert api.get("/v1/images/models").status_code == 401
    assert api.post("/v1/images/generations", json={"prompt": "x"}).status_code == 401
    assert api.post("/v1/videos/generations", json={"prompt": "x"}).status_code == 401


def test_images_models_lists_local(client):
    """Görsel model listesi yalnızca image kategorisini döner."""
    api, _s, _m = client
    body = api.get("/v1/images/models", headers=_auth()).json()
    ids = [m["id"] for m in body["data"]]
    assert "yerel-gorsel" in ids
    assert "yerel-video" not in ids


def test_videos_models_lists_local(client):
    """Video model listesi yalnızca video kategorisini döner."""
    api, _s, _m = client
    body = api.get("/v1/videos/models", headers=_auth()).json()
    ids = [m["id"] for m in body["data"]]
    assert "yerel-video" in ids
    assert "yerel-gorsel" not in ids


# ------------------------------------------------------------------
# Görsel üretimi (in-process)
# ------------------------------------------------------------------

def test_generate_image_local_b64(client):
    """Yerel modelle b64_json üretimi OpenAI biçiminde döner."""
    api, store, manager = client
    r = api.post("/v1/images/generations", headers=_auth(), json={
        "model": "yerel-gorsel", "prompt": "bir kedi",
        "response_format": "b64_json", "n": 2, "size": "512x512", "steps": 10,
    })
    assert r.status_code == 200
    data = r.json()["data"]
    assert len(data) == 2
    assert base64.b64decode(data[0]["b64_json"]).startswith(b"\x89PNG")
    assert manager.calls[0]["model_id"] == "yerel-gorsel"
    assert manager.calls[0]["steps"] == 10


def test_generate_image_local_url_is_servable(client):
    """url modu /v1/images/file/... üzerinden gerçekten sunulabilir."""
    api, _s, _m = client
    r = api.post("/v1/images/generations", headers=_auth(), json={
        "model": "yerel-gorsel", "prompt": "kedi", "response_format": "url",
    })
    url = r.json()["data"][0]["url"]
    assert url.startswith("/v1/images/file/")

    served = api.get(url, headers=_auth())
    assert served.status_code == 200
    assert served.content.startswith(b"\x89PNG")
    assert served.headers["content-type"].startswith("image/png")


def test_generate_image_local_charges_per_image(client):
    """Her görsel için 'image' tarifesi düşülür."""
    from app.gateway import hash_key

    api, store, _m = client
    key_id = store.get_api_key_by_hash(hash_key(API_KEY))["id"]
    before = store.usage_totals(key_id)["cost"]

    api.post("/v1/images/generations", headers=_auth(), json={
        "model": "yerel-gorsel", "prompt": "x", "n": 3, "response_format": "b64_json",
    })

    delta = store.usage_totals(key_id)["cost"] - before
    assert delta == pytest.approx(3 * 0.04, abs=1e-9)


def test_generate_image_video_model_rejected(client):
    """Video modeli /v1/images/generations'da kabul edilmez (ComfyUI'ye düşer → 400)."""
    api, _s, _m = client
    r = api.post("/v1/images/generations", headers=_auth(), json={
        "model": "yerel-video", "prompt": "x",
    })
    # ComfyUI kapalı olduğu için 400 (checkpoint/in-process eşleşmedi)
    assert r.status_code == 400
    assert r.json()["error"]["code"] in ("comfy_disabled", "checkpoint_required")


def test_generate_image_invalid_n(client):
    """n 1-8 aralığında olmalıdır."""
    api, _s, _m = client
    r = api.post("/v1/images/generations", headers=_auth(), json={
        "model": "yerel-gorsel", "prompt": "x", "n": 20,
    })
    assert r.status_code == 400


def test_generate_image_invalid_response_format(client):
    """response_format yalnızca url | b64_json olabilir."""
    api, _s, _m = client
    r = api.post("/v1/images/generations", headers=_auth(), json={
        "model": "yerel-gorsel", "prompt": "x", "response_format": "raw",
    })
    assert r.status_code == 400


def test_generate_image_unknown_local_model_falls_back_to_comfy(client):
    """Bilinmeyen model ComfyUI yoluna düşer; ComfyUI kapalıysa anlaşılır 400."""
    api, _s, _m = client
    r = api.post("/v1/images/generations", headers=_auth(), json={
        "model": "bilinmeyen", "prompt": "x",
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "comfy_disabled"


# ------------------------------------------------------------------
# Görsel düzenleme (edits)
# ------------------------------------------------------------------

def test_image_edits_multipart(client):
    """Multipart yüklenen görselle img2img üretimi yapılır."""
    from PIL import Image
    import io

    api, store, manager = client
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), (10, 20, 30)).save(buf, format="PNG")
    buf.seek(0)

    r = api.post(
        "/v1/images/edits",
        headers=_auth(),
        files={"image": ("girdi.png", buf, "image/png")},
        data={"prompt": "piksel sanatı", "model": "yerel-gorsel",
              "response_format": "b64_json", "n": "1"},
    )
    assert r.status_code == 200
    assert base64.b64decode(r.json()["data"][0]["b64_json"]).startswith(b"\x89PNG")
    assert manager.calls[0]["prompt"] == "piksel sanatı"


def test_image_edits_requires_local_model(client):
    """edits yalnızca yerel modellerle çalışır; bilinmeyen modelde 404 verir."""
    api, _s, _m = client
    r = api.post(
        "/v1/images/edits",
        headers=_auth(),
        files={"image": ("a.png", b"not-a-png", "image/png")},
        data={"prompt": "x", "model": "yok"},
    )
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "model_not_found"


def test_image_edits_rejects_broken_image(client):
    """Okunamayan görsel 400 döner."""
    api, _s, _m = client
    r = api.post(
        "/v1/images/edits",
        headers=_auth(),
        files={"image": ("a.png", b"bozuk-veri", "image/png")},
        data={"prompt": "x", "model": "yerel-gorsel"},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_image"


# ------------------------------------------------------------------
# Video üretimi
# ------------------------------------------------------------------

def test_generate_video_local_url(client):
    """Yerel video modeli GIF üretir ve /v1/videos/file/... sunar."""
    api, store, manager = client
    r = api.post("/v1/videos/generations", headers=_auth(), json={
        "model": "yerel-video", "prompt": "uçan kuş", "frames": 6,
    })
    assert r.status_code == 200
    item = r.json()["data"][0]
    assert item["url"].startswith("/v1/videos/file/")
    assert item["frames"] == 6
    assert manager.calls[0]["frames"] == 6

    served = api.get(item["url"], headers=_auth())
    assert served.status_code == 200
    assert served.content.startswith(b"GIF89a")
    assert served.headers["content-type"].startswith("image/gif")


def test_generate_video_local_b64(client):
    """b64_json modu GIF'i base64 döner."""
    api, _s, _m = client
    r = api.post("/v1/videos/generations", headers=_auth(), json={
        "model": "yerel-video", "prompt": "x", "response_format": "b64_json",
    })
    assert base64.b64decode(r.json()["data"][0]["b64_json"]).startswith(b"GIF89a")


def test_generate_video_charges_per_video(client):
    """Video başına 'video' tarifesi (0.5) düşülür."""
    from app.gateway import hash_key

    api, store, _m = client
    key_id = store.get_api_key_by_hash(hash_key(API_KEY))["id"]
    before = store.usage_totals(key_id)["cost"]

    api.post("/v1/videos/generations", headers=_auth(), json={
        "model": "yerel-video", "prompt": "x", "response_format": "b64_json",
    })

    assert store.usage_totals(key_id)["cost"] - before == pytest.approx(0.5, abs=1e-9)


# ------------------------------------------------------------------
# ComfyUI yolu (geriye dönük uyum)
# ------------------------------------------------------------------

def test_generate_image_uses_comfy_when_model_unknown(client, monkeypatch):
    """Registry'de olmayan model için ComfyUI köprüsü çalışır."""
    api, _s, manager = client
    seen = {}

    async def fake_generate_image(**kw):
        seen.update(kw)
        return {"prompt_id": "pid", "images": [{"bytes": b"\x89PNG-comfy", "mime": "image/png"}]}

    monkeypatch.setattr(comfy_api.comfy_client, "generate_image", fake_generate_image)
    monkeypatch.setattr(comfy_api.settings, "comfyui_enabled", True)

    r = api.post("/v1/images/generations", headers=_auth(), json={
        "model": "checkpoint-ckpt", "prompt": "x", "response_format": "b64_json",
    })

    assert r.status_code == 200
    assert base64.b64decode(r.json()["data"][0]["b64_json"]).startswith(b"\x89PNG-comfy")
    assert seen["checkpoint"] == "checkpoint-ckpt"
    assert manager.calls == []      # yerel yöneticiye hiç gidilmedi


def test_generate_image_comfy_disabled_400(client, monkeypatch):
    """ComfyUI kapalıyken yerel olmayan model 400 (comfy_disabled) verir."""
    api, _s, _m = client
    monkeypatch.setattr(comfy_api.settings, "comfyui_enabled", False)
    r = api.post("/v1/images/generations", headers=_auth(), json={
        "model": "checkpoint-ckpt", "prompt": "x",
    })
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "comfy_disabled"


def test_image_models_includes_comfy_checkpoints(client, monkeypatch):
    """ComfyUI açıkken model listesine checkpoint'ler de eklenir."""
    api, _s, _m = client
    monkeypatch.setattr(comfy_api.settings, "comfyui_enabled", True)
    monkeypatch.setattr(comfy_api, "get_client", lambda: type(
        "C", (), {"checkpoints": staticmethod(lambda: ["ckpt-1"])}
    )())
    body = api.get("/v1/images/models", headers=_auth()).json()
    sources = {m["id"]: m.get("source") for m in body["data"]}
    assert sources["yerel-gorsel"] == "local"
    assert sources["ckpt-1"] == "comfyui"
