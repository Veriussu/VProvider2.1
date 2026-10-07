# ─────────────────────────────────────────────────────────────
#  Bölüm:    OpenAI Uyumlu Görsel + Video Üretim API'si
#  Dosya:    app/comfy_api.py
#  Amaç:     Görsel ve video üretimini OpenAI /v1/images/* ve /v1/videos/*
#            biçiminde sunar. İKİ MOTOR desteklenir ve tek API altında birleşir:
#              1) in-process: models/safetensors/ altındaki difüzör modelleri
#                 (app/diffusers_backend.py) — ComfyUI gerekmez.
#              2) ComfyUI köprüsü: harici motor (app/comfy_client.py).
#            Model alanı önce registry'de aranır; bulunamazsa ComfyUI
#            checkpoint'i olarak yorumlanır (geriye dönük uyum).
#  Mekanik:  - /v1/images/generations        -> görsel; url veya b64_json
#            - /v1/images/edits             -> görsel→görsel (giriş görseli ile)
#            - /v1/images/file/{id}/{i}      -> url modunda görsel
#            - /v1/images/models            -> kullanılabilir görsel modelleri
#            - /v1/images/comfy/checkpoints  -> ComfyUI checkpoint listesi
#            - /v1/videos/generations        -> video (GIF); url modu
#            - /v1/videos/file/{id}          -> url modunda video
#            - /v1/videos/models            -> kullanılabilir video modelleri
#            - Aynı kimlik doğrulama (Bearer) ve hata yapısı (detail.error).
#  Kullanım: app/main.py içinde app.include_router(comfy_router)
# ─────────────────────────────────────────────────────────────

import base64
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from fastapi.routing import APIRouter as _Router
from pydantic import BaseModel, Field

from app import comfy_client, diffusers_backend, registry
from app.comfy_client import ComfyGenerationError, ComfyUnavailableError, get_client
from app.config import settings
from app.diffusers_backend import MediaError, MediaUnavailableError
from app.gateway import ApiContext, api_auth
from app.openai_api import _openai_error

router: _Router = APIRouter(prefix="/v1/images", dependencies=[Depends(api_auth)])
video_router: _Router = APIRouter(prefix="/v1/videos", dependencies=[Depends(api_auth)])


# ------------------------------------------------------------------
# İstek modeli (OpenAI uyumlu alanlar)
# ------------------------------------------------------------------

class ImagesGenerationRequest(BaseModel):
    """POST /v1/images/generations istek gövdesi."""

    prompt: str = Field(..., min_length=1)
    model: str = ""                 # model adı (registry) veya checkpoint
    n: int = 1                      # üretilecek görsel sayısı
    size: str = "512x512"
    response_format: str = "url"    # url | b64_json
    negative_prompt: str = ""       # OpenAI'de yok; VProvider eklentisi
    steps: int = Field(20, ge=1, le=100)
    cfg: float = Field(7.0, ge=0.0, le=30.0)
    seed: int = -1


class VideoGenerationRequest(BaseModel):
    """POST /v1/videos/generations istek gövdesi."""

    prompt: str = Field(..., min_length=1)
    model: str = ""                 # model adı (registry) veya checkpoint
    size: str = "512x512"
    frames: int = Field(16, ge=4, le=120)   # kare sayısı
    fps: int = Field(diffusers_backend.DEFAULT_FPS, ge=1, le=30)
    response_format: str = "url"           # url | b64_json
    steps: int = Field(25, ge=1, le=100)
    cfg: float = Field(7.0, ge=0.0, le=30.0)
    seed: int = -1


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------

def _require_comfy() -> None:
    """ComfyUI kapalıysa OpenAI uyumlu 400 hatası verir."""
    if not settings.comfyui_enabled:
        raise _openai_error(
            400,
            "Görsel üretim modülü kapalı: .env içinde COMFYUI_ENABLED=true yapın "
            "ve ComfyUI'yi çalıştırın (bkz. deploy/comfyui-rehber.md).",
            code="comfy_disabled",
        )


def _pick_checkpoint(model: str) -> str:
    """İstenen model checkpoint olarak kullanılır; yoksa varsayılana düşer."""
    if model:
        return model
    if settings.comfyui_default_checkpoint:
        return settings.comfyui_default_checkpoint
    raise _openai_error(
        400,
        "Checkpoint belirtilmedi. request.model ile veya COMFYUI_DEFAULT_CHECKPOINT "
        "ile seçin; mevcutları GET /v1/images/comfy/checkpoints döner.",
        param="model",
        code="checkpoint_required",
    )


def _local_model(model_id: str, category: str):
    """Verilen kategorideki yerel (in-process) modeli bulur; yoksa None."""
    if not model_id:
        return None
    manager = diffusers_backend.get_media_manager()
    info = manager.get_model(model_id)
    if info is None or info.category != category:
        return None
    return info


def _require_local(model_id: str, category: str):
    """Yerel modeli döner; yoksa OpenAI uyumlu 404 üretir."""
    info = _local_model(model_id, category)
    if info is None:
        raise _openai_error(
            404,
            f"'{model_id}' {category} modeli bulunamadı. Panelden "
            f"models/{registry.KIND_SAFETENSORS}/ altına kurun "
            "(GET /v1/images/models listesine bakın).",
            param="model",
            code="model_not_found",
        )
    return info


def _media_error(exc: Exception) -> HTTPException:
    """Medya hatasını OpenAI biçimine çevirir."""
    if isinstance(exc, MediaUnavailableError):
        return _openai_error(400, str(exc), code="media_unavailable")
    return _openai_error(500, f"Üretim sırasında hata: {exc}", code="generation_failed")


def _decode_image(data: bytes, mime: str = "image/png"):
    """Baytı PIL görseline çevirir (img2img girişi için)."""
    try:
        from PIL import Image
    except Exception as exc:  # Pillow yoksa
        raise _openai_error(
            400, f"Görsel işlemek için Pillow gerekli: {exc}", code="pillow_missing"
        )
    import io

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:
        raise _openai_error(400, f"Görsel okunamadı: {exc}", code="invalid_image")


def _validate_response_format(value: str) -> str:
    if value not in ("url", "b64_json"):
        raise _openai_error(
            400, "response_format yalnızca url veya b64_json olabilir.", param="response_format"
        )
    return value


# ------------------------------------------------------------------
# Uç noktalar
# ------------------------------------------------------------------

@router.get("/models")
def list_image_models():
    """Bu uç noktadan üretilebilen görsel modellerini listeler.

    Kaynak: models/safetensors/ altındaki image kategorili modeller (in-process)
    ve ComfyUI açıksa checkpoint'ler. OpenAI'nin /v1/models listesiyle aynı
    adları kullanır; böylece istemci tek listeden seçim yapabilir.
    """
    manager = diffusers_backend.get_media_manager()
    data = [
        {"id": m.model_id, "object": "model", "source": "local",
         "loaded": m.loaded, "category": m.category}
        for m in manager.list_models() if m.category == registry.CAT_IMAGE
    ]
    if settings.comfyui_enabled:
        try:
            for c in get_client().checkpoints():
                data.append({"id": c, "object": "model", "source": "comfyui"})
        except Exception:
            pass  # ComfyUI erişilemiyorsa yalnızca yerel modeller listelenir
    return {"object": "list", "data": data}


@router.get("/comfy/checkpoints")
def list_checkpoints():
    """ComfyUI'de kurulu checkpoint'leri listeler (ürün arayüzleri için)."""
    if not comfy_client.settings.comfyui_enabled:
        raise _openai_error(400, "Görsel üretim modülü kapalı.", code="comfy_disabled")
    try:
        checkpoints = get_client().checkpoints()
    except Exception as exc:
        raise _openai_error(502, f"ComfyUI'ye ulaşılamadı: {exc}", code="comfy_unreachable")
    return {"object": "list", "data": [{"id": c, "object": "model"} for c in checkpoints]}


@router.post("/generations")
async def create_image(req: ImagesGenerationRequest, ctx: ApiContext = Depends(api_auth)):
    """Görsel üretir ve OpenAI biçiminde url veya b64_json döner.

    model alanı registry'de bir görsel modeline işaret ediyorsa sunucu içinde
    (in-process) üretilir; değilse ComfyUI köprüsü kullanılır.
    """
    if not 1 <= req.n <= 8:
        raise _openai_error(400, "n değeri 1-8 arasında olmalıdır.", param="n")
    fmt = _validate_response_format(req.response_format)

    local = _local_model(req.model, registry.CAT_IMAGE)
    if local is not None:
        return await _generate_local_image(req, local, fmt, ctx)

    _require_comfy()
    checkpoint = _pick_checkpoint(req.model)
    try:
        result = await comfy_client.generate_image(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            checkpoint=checkpoint,
            size=req.size,
            steps=req.steps,
            cfg=req.cfg,
            seed=req.seed,
            batch_size=req.n,
        )
    except ComfyUnavailableError as exc:
        raise _openai_error(400, str(exc), code="comfy_disabled")
    except ComfyGenerationError as exc:
        raise _openai_error(500, f"Üretim sırasında hata: {exc}", code="generation_failed")
    except Exception as exc:
        raise _openai_error(502, f"ComfyUI bağlantı hatası: {exc}", code="comfy_unreachable")

    prompt_id = result["prompt_id"]
    ctx.charge("image", model_id=checkpoint, images=len(result["images"]))
    data = []
    for i, img in enumerate(result["images"]):
        if fmt == "b64_json":
            data.append({"b64_json": base64.b64encode(img["bytes"]).decode("ascii")})
        else:
            data.append({"url": f"/v1/images/file/{prompt_id}/{i}"})
    return {"created": int(time.time()), "data": data}


async def _generate_local_image(req: ImagesGenerationRequest, info, fmt: str,
                                ctx: ApiContext) -> dict:
    """In-process görsel üretimi ve OpenAI biçiminde yanıt."""
    manager = diffusers_backend.get_media_manager()
    try:
        result = await manager.generate(
            info.model_id,
            req.prompt,
            negative_prompt=req.negative_prompt,
            size=req.size,
            steps=req.steps,
            cfg=req.cfg,
            seed=req.seed,
            n=req.n,
        )
    except MediaError as exc:
        raise _media_error(exc)
    except Exception as exc:
        raise _openai_error(500, f"Üretim sırasında hata: {exc}", code="generation_failed")

    ctx.charge("image", model_id=info.model_id, images=len(result.images))
    prompt_id = uuid.uuid4().hex
    stored = [
        {"bytes": b, "mime": "image/png", "width": 0, "height": 0}
        for b in result.images
    ]
    comfy_client.store_generated_images(prompt_id, stored)

    data = []
    for i, raw in enumerate(result.images):
        if fmt == "b64_json":
            data.append({"b64_json": base64.b64encode(raw).decode("ascii")})
        else:
            data.append({"url": f"/v1/images/file/{prompt_id}/{i}"})
    return {"created": int(time.time()), "data": data}


@router.post("/edits")
async def edit_image(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    model: str = Form(""),
    n: int = Form(1),
    size: str = Form("512x512"),
    response_format: str = Form("url"),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    seed: int = Form(-1),
    ctx: ApiContext = Depends(api_auth),
):
    """Görsel→görsel düzenleme (OpenAI /v1/images/edits ile uyumlu).

    Giriş görseli multipart olarak yüklenir; model registry'deki bir görsel
    modeli olmalıdır. Güçlük (strength) istekte verilmezse 0.75 kullanılır.
    """
    if not 1 <= n <= 8:
        raise _openai_error(400, "n değeri 1-8 arasında olmalıdır.", param="n")
    fmt = _validate_response_format(response_format)
    info = _require_local(model, registry.CAT_IMAGE)

    raw = await image.read()
    source = _decode_image(raw, image.content_type or "image/png")

    manager = diffusers_backend.get_media_manager()
    try:
        result = await manager.generate(
            info.model_id,
            prompt,
            size=size,
            steps=steps,
            cfg=cfg,
            seed=seed,
            n=n,
            image=source,
        )
    except MediaError as exc:
        raise _media_error(exc)
    except Exception as exc:
        raise _openai_error(500, f"Üretim sırasında hata: {exc}", code="generation_failed")

    ctx.charge("image", model_id=info.model_id, images=len(result.images))
    prompt_id = uuid.uuid4().hex
    comfy_client.store_generated_images(
        prompt_id, [{"bytes": b, "mime": "image/png"} for b in result.images]
    )

    data = []
    for i, b in enumerate(result.images):
        if fmt == "b64_json":
            data.append({"b64_json": base64.b64encode(b).decode("ascii")})
        else:
            data.append({"url": f"/v1/images/file/{prompt_id}/{i}"})
    return {"created": int(time.time()), "data": data}


@router.get("/file/{prompt_id}/{index}")
def serve_image(prompt_id: str, index: int):
    """url modunda dönen /v1/images/file/... adresinin arkasındaki görsel."""
    img = comfy_client.get_stored_image(prompt_id, index)
    if img is None:
        raise HTTPException(status_code=404, detail="Görsel bulunamadı (depo temizlenmiş olabilir).")
    return Response(content=img["bytes"], media_type=img.get("mime", "image/png"))


# ------------------------------------------------------------------
# Uç noktalar: /v1/videos/*
# ------------------------------------------------------------------

@video_router.get("/models")
def list_video_models():
    """Bu uç noktadan üretilebilen video modellerini listeler."""
    manager = diffusers_backend.get_media_manager()
    data = [
        {"id": m.model_id, "object": "model", "source": "local",
         "loaded": m.loaded, "category": m.category}
        for m in manager.list_models() if m.category == registry.CAT_VIDEO
    ]
    if settings.comfyui_enabled:
        try:
            for c in get_client().checkpoints():
                data.append({"id": c, "object": "model", "source": "comfyui"})
        except Exception:
            pass
    return {"object": "list", "data": data}


@video_router.get("/comfy/checkpoints")
def list_video_checkpoints():
    """Video üretimi için de aynı checkpoint seti kullanılır."""
    if not comfy_client.settings.comfyui_enabled:
        raise _openai_error(400, "Görsel/video üretim modülü kapalı.", code="comfy_disabled")
    try:
        checkpoints = get_client().checkpoints()
    except Exception as exc:
        raise _openai_error(502, f"ComfyUI'ye ulaşılamadı: {exc}", code="comfy_unreachable")
    return {"object": "list", "data": [{"id": c, "object": "model"} for c in checkpoints]}


@video_router.post("/generations")
async def create_video(req: VideoGenerationRequest, ctx: ApiContext = Depends(api_auth)):
    """Video üretir; GIF olarak url ya da b64_json döner.

    model alanı registry'deki bir video modeline işaret ediyorsa sunucu içinde
    üretilir; değilse ComfyUI köprüsü (AnimateDiff) kullanılır.
    """
    fmt = _validate_response_format(req.response_format)

    local = _local_model(req.model, registry.CAT_VIDEO)
    if local is not None:
        return await _generate_local_video(req, local, fmt, ctx)

    _require_comfy()
    checkpoint = _pick_checkpoint(req.model)
    try:
        result = await comfy_client.generate_video(
            prompt=req.prompt,
            negative_prompt="",
            checkpoint=checkpoint,
            size=req.size,
            frames=req.frames,
            steps=req.steps,
            cfg=req.cfg,
            seed=req.seed,
        )
    except ComfyUnavailableError as exc:
        raise _openai_error(400, str(exc), code="comfy_disabled")
    except ComfyGenerationError as exc:
        raise _openai_error(500, f"Üretim sırasında hata: {exc}", code="generation_failed")
    except Exception as exc:
        raise _openai_error(502, f"ComfyUI bağlantı hatası: {exc}", code="comfy_unreachable")

    video = result["video"]
    ctx.charge("video", model_id=checkpoint, videos=1)
    if fmt == "b64_json":
        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(video["bytes"]).decode("ascii")}],
        }
    return {
        "created": int(time.time()),
        "data": [{
            "url": f"/v1/videos/file/{result['prompt_id']}",
            "mime_type": video["mime"],
            "width": video["width"],
            "height": video["height"],
            "frames": video["frames"],
        }],
    }


async def _generate_local_video(req: VideoGenerationRequest, info, fmt: str,
                                 ctx: ApiContext) -> dict:
    """In-process video üretimi ve OpenAI biçiminde yanıt."""
    manager = diffusers_backend.get_media_manager()
    try:
        result = await manager.generate(
            info.model_id,
            req.prompt,
            size=req.size,
            steps=req.steps,
            cfg=req.cfg,
            seed=req.seed,
            frames=req.frames,
            fps=req.fps,
        )
    except MediaError as exc:
        raise _media_error(exc)
    except Exception as exc:
        raise _openai_error(500, f"Üretim sırasında hata: {exc}", code="generation_failed")

    video = result.video or {}
    ctx.charge("video", model_id=info.model_id, videos=1)
    if fmt == "b64_json":
        return {
            "created": int(time.time()),
            "data": [{"b64_json": base64.b64encode(video["bytes"]).decode("ascii")}],
        }
    prompt_id = uuid.uuid4().hex
    comfy_client.store_generated_video(prompt_id, video)
    return {
        "created": int(time.time()),
        "data": [{
            "url": f"/v1/videos/file/{prompt_id}",
            "mime_type": video.get("mime", "image/gif"),
            "width": video.get("width", 0),
            "height": video.get("height", 0),
            "frames": video.get("frames", 0),
        }],
    }


@video_router.get("/file/{prompt_id}")
def serve_video(prompt_id: str):
    """url modunda dönen videoyu (GIF) sunar."""
    video = comfy_client.get_stored_video(prompt_id)
    if video is None:
        raise HTTPException(status_code=404, detail="Video bulunamadı (depo temizlenmiş olabilir).")
    return Response(content=video["bytes"], media_type=video.get("mime", "image/gif"))