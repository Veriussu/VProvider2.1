# ─────────────────────────────────────────────────────────────
#  Bölüm:    OpenAI Uyumlu Müzik (Music) API'si
#  Amaç:     Kullanıcıdan prompt alıp ses dosyası üretir.
#            Şu an basit bir ton döngüsü; ileride MusicGen kullanılır.
# ─────────────────────────────────────────────────────────────

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from app import music_backend
from app.config import settings
from app.gateway import ApiContext, api_auth
from app.openai_api import _openai_error
from app.music_backend import MusicError

router = APIRouter(prefix="/v1/audio", dependencies=[Depends(api_auth)])


# ------------------------------------------------------------------
# İstek modeli (OpenAI uyumlu alanlar)
# ------------------------------------------------------------------

class MusicGenerateRequest(BaseModel):
    """POST /v1/audio/music/generate isteği."""
    model: str = Field(default="musicgen-small", description="Model adı")
    prompt: str = Field(..., min_length=1, description="Müzik açıklaması")
    duration: float = Field(default=10.0, ge=1.0, le=30.0, description="Süre (saniye)")
    temperature: float = Field(default=0.8, ge=0.1, le=1.5, description="Örnekleme sıcaklığı")
    top_k: int = Field(default=250, ge=1, le=1000, description="Top-k örnekleme")
    top_p: float = Field(default=0.0, ge=0.0, le=1.0, description="Top-p (nucleus) örnekleme")
    response_format: str = Field(default="wav", description="Çıktı formatı: wav, mp3")


# ------------------------------------------------------------------
# Uç noktalar
# ------------------------------------------------------------------

@router.post("/music/generate")
async def generate_music(req: MusicGenerateRequest, ctx: ApiContext = Depends(api_auth)):
    """
    Prompt'tan müzik üretir.
    
    POST /v1/audio/music/generate
    JSON body: {model, prompt, duration, temperature, ...}
    """
    if not settings.music_enabled:
        raise HTTPException(
            status_code=400,
            detail="Müzik üretim modülü kapalı (MUSIC_ENABLED=false)."
        )

    try:
        audio_bytes, mime = await music_backend.generate_music(
            prompt=req.prompt,
            model=req.model,
            duration=req.duration,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
            response_format=req.response_format,
        )
    except MusicError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Müzik üretim hatası: {exc}")

    # Kredi düşümü: saniye başına maliyet (varsayılan: 0.01/saniye)
    ctx.charge("music", model_id=req.model, audio_sec=req.duration)

    # Dosya sakla ve URL döndür (panel için)
    from uuid import uuid4
    key = uuid4().hex[:12]
    # Ses dosyasını geçici olarak sakla
    import tempfile
    import os
    tmp_dir = tempfile.gettempdir()
    file_path = os.path.join(tmp_dir, f"vp_music_{key}.wav")
    with open(file_path, "wb") as f:
        f.write(audio_bytes)

    # Media benzeri bir URL yapısı döndür
    # Gerçekte storage backend'i kullanılır
    # return {"url": f"/v1/audio/file/{key}"}
    
    # Şimdilik doğrudan ses döndür (OpenAI tarzı)
    if req.response_format == "mp3":
        with open(file_path, "rb") as f:
            return Response(content=f.read(), media_type="audio/mpeg")
    else:
        return Response(content=audio_bytes, media_type="audio/wav")


@router.get("/music/models")
async def list_music_models():
    """Müzik modelleri listesi (panel için)."""
    # MusicGen modelleri
    models = [
        {"id": "musicgen-small", "object": "model", "category": "music", "description": "MusicGen Small"},
        {"id": "musicgen-medium", "object": "model", "category": "music", "description": "MusicGen Medium"},
        {"id": "musicgen-large", "object": "model", "category": "music", "description": "MusicGen Large"},
        {"id": "meta-musicgen-3.5s", "object": "model", "category": "music", "description": "MusicGen 3.5s (kısa video)"},
    ]
    return {"object": "list", "data": models}


@router.get("/music/status")
async def music_status_endpoint():
    """Müzik üretim durumu (panel ve teşhis için)."""
    return await music_backend.music_status()