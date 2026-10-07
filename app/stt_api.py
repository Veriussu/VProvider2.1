# ─────────────────────────────────────────────────────────────
#  Bölüm:    OpenAI Uyumlu STT (Speech-to-Text) API'si
#  Amaç:     OpenAI /v1/audio/transcriptions ile uyumlu ses-yazı uçları.
#            faster-whisper (ctranslate2) modellerini kullanır.
# ─────────────────────────────────────────────────────────────

from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from app import stt_backend
from app.config import settings
from app.gateway import ApiContext, api_auth
from app.openai_api import _openai_error
from app.stt_backend import STTError

router = APIRouter(prefix="/v1/audio", dependencies=[Depends(api_auth)])


# ------------------------------------------------------------------
# İstek modeli (OpenAI uyumlu alanlar)
# ------------------------------------------------------------------

class TranscriptionRequest(BaseModel):
    """POST /v1/audio/transcriptions istek gövdesi (multipart form-data)."""
    model: str = Field(default="base", description="ct2 model adı (base, small, medium, large-v2, vb.)")
    language: Optional[str] = Field(default=None, description="Dil kodu (örn: 'tr', 'en') veya None (otomatik)")
    prompt: Optional[str] = Field(default=None, description="İlk segment için ipucu metni")
    response_format: str = Field(default="json", description="Yanıt formatı: json, text, srt, vtt, tsv")
    temperature: float = Field(default=0.0, ge=0.0, le=1.0, description="Örnekleme sıcaklığı")
    timestamp_granularities: list[str] = Field(default=["segment"], description="Zaman damgası hassasiyeti: ['word', 'segment']")


# ------------------------------------------------------------------
# Uç noktalar
# ------------------------------------------------------------------

@router.post("/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form("base"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: str = Form("segment"),  # JSON string olarak gelir
    ctx: ApiContext = Depends(api_auth),
):
    """
    Ses dosyasını metne çevirir (OpenAI uyumlu).
    
    Multipart form-data:
    - file: WAV/MP3/M4A formatında ses dosyası
    - model: ct2 model adı (base, small, medium, large-v2, vb.)
    - language: Dil kodu (örn: 'tr', 'en')
    - prompt: İpucu metni
    - response_format: json, text, srt, vtt, tsv
    - temperature: 0.0-1.0 arası
    - timestamp_granularities: "segment" veya "word" (JSON string)
    """
    # Dosya formatı kontrolü
    allowed_types = ["audio/wav", "audio/mp3", "audio/m4a", "audio/ogg", "audio/flac", "audio/webm"]
    if file.content_type not in allowed_types:
        raise _openai_error(
            400,
            f"Desteklenmeyen ses formatı: {file.content_type}. "
            f"İzin verilenler: {', '.join(allowed_types)}",
            param="file",
        )

    # Dosyayı oku
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise _openai_error(400, f"Dosya okunamadı: {e}", code="file_read_error")

    if len(audio_bytes) == 0:
        raise _openai_error(400, "Boş ses dosyası.", param="file")

    # timestamp_granularities parse et
    try:
        import json
        ts_gran = json.loads(timestamp_granularities) if isinstance(timestamp_granularities, str) else timestamp_granularities
    except Exception:
        ts_gran = ["segment"]

    # Transkripsiyonu çalıştır
    try:
        result = await stt_backend.transcribe_audio(
            audio_bytes=audio_bytes,
            model=model,
            language=language,
            prompt=prompt,
            temperature=temperature,
            response_format=response_format,
            timestamp_granularities=ts_gran,
        )
    except STTError as exc:
        raise _openai_error(400, str(exc), code="transcription_failed")
    except Exception as exc:
        raise _openai_error(502, f"STT motoru hatası: {exc}", code="stt_unreachable")

    # Kredi düşümü: saniye başına maliyet (varsayılan)
    duration = 0.0
    if isinstance(result, dict) and "duration" in result:
        duration = result["duration"]
    elif isinstance(result, str):
        # text formatında süre kestirimi (tahmini)
        duration = len(result.split()) / 2.5  # ~2.5 kelime/saniye

    ctx.charge("stt", model_id=model, audio_sec=duration)

    # Yanıt formatı
    if response_format == "text":
        return Response(content=result, media_type="text/plain")
    elif response_format in ["srt", "vtt", "tsv"]:
        return Response(content=result, media_type=f"text/{response_format}")
    else:
        return JSONResponse(content=result)


@router.get("/transcriptions/status")
async def stt_status_endpoint():
    """STT motor durumu (panel ve teşhis için)."""
    return await stt_backend.stt_status()


@router.get("/transcriptions/models")
async def list_stt_models():
    """Mevcut ct2 modellerini listeler (panel için)."""
    from app import registry
    import os
    
    models = []
    ct2_dir = settings.models_path / "ct2"
    if ct2_dir.exists():
        for name in os.listdir(ct2_dir):
            model_path = ct2_dir / name
            if model_path.is_dir() and (model_path / "model.bin").exists():
                info = registry.get_model_info(name)
                models.append({
                    "id": name,
                    "object": "model",
                    "category": "stt",
                    "kind": "ct2",
                    "loaded": False,
                    "size_mb": info.get("size_mb", 0) if info else 0,
                })
    
    return {"object": "list", "data": models}