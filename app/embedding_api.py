# ─────────────────────────────────────────────────────────────
#  Bölüm:    OpenAI Uyumlu Embedding API
#  Dosya:    app/embedding_api.py
#  Amaç:     POST /v1/embeddings uç noktasını OpenAI biçiminde sunar.
#  Mekanik:  - API anahtarı korumalıdır (gateway.api_auth).
#            - Model çözümü: önce models/embeddings/ altına kurulu model
#              adı aranır; bulunamazsa config.embedding_model (HF kimliği)
#              kullanılabilir; ikisi de yoksa OpenAI uyumlu 404 döner.
#            - Token tahmini (karakter/4) ile "embedding" tarifesinden
#              kredi düşülür (pricing.DEFAULT_PRICES["embedding"]).
#  Kullanım: POST /v1/embeddings  {"model":"all-MiniLM-L6-v2","input":"merhaba"}
# ─────────────────────────────────────────────────────────────

import math
from typing import Optional, Union

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app import embedding_backend, registry
from app.config import settings
from app.gateway import ApiContext, api_auth
router = APIRouter(
    prefix="/v1",
    tags=["embeddings"],
    responses={404: {"description": "Bulunamadı"}},
)


class EmbeddingRequest(BaseModel):
    """OpenAI uyumlu embedding isteği."""

    model: str
    input: Union[str, list[str]]


def _openai_error(status_code: int, message: str, code: str, param: Optional[str] = None) -> JSONResponse:
    """OpenAI uyumlu hata yanıtı üretir."""
    detail: dict = {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "param": param,
            "code": code,
        }
    }
    return JSONResponse(status_code=status_code, content=detail)


def _estimate_tokens(texts: list[str]) -> int:
    """Karakter sayısından kabaca token tahmini (İngilizce ~4 karakter)."""
    return sum(max(1, math.ceil(len(t) / 4)) for t in texts)


def _model_hint(exc: Exception) -> str:
    """Sık görülen model klasörü hatalarına kısa bir yönlendirme ekler.

    En sık görülen durum, model klasörünün eksik kopyalanmasıdır: yalnızca
    config.json + ağırlık dosyası gelir, "1_Pooling/" klasörü gelmez.
    """
    text = str(exc)
    if "Pooling" in text:
        return (" — model klasörü eksik olabilir. HuggingFace'teki tüm içeriği "
                "(1_Pooling, 2_Normalize vb. klasörler dahil) models/embeddings/ "
                "altına kopyalayın.")
    if "No such file" in text or "not found" in text.lower():
        return (" — model klasörü eksik olabilir. config.json ve ağırlık dosyası "
                "models/embeddings/<model>/ altında bulunmalı.")
    return ""


def resolve_embedding_model(model_id: str) -> Optional[str]:
    """İstenen modelin yerel yolunu (ya da varsayılan HF kimliğini) bulur.

    Öncelik: models/embeddings/ altına kurulu model adı. İkincil: config'te
    tanımlı varsayılan embedding model kimliği. İkisi de yoksa None.
    """
    for entry in registry.scan_typed():
        if entry.kind == registry.KIND_EMBEDDINGS and entry.model_id == model_id:
            return str(entry.path)
    if settings.embedding_model and model_id == settings.embedding_model:
        return settings.embedding_model
    return None


@router.post("/embeddings")
async def create_embeddings(
    req: EmbeddingRequest,
    ctx: ApiContext = Depends(api_auth),
):
    """Metinleri vektöre çevirir; OpenAI formatında döner."""
    model_path = resolve_embedding_model(req.model)
    if model_path is None:
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "message": (
                        f"Aradığınız '{req.model}' embedding modeli bulunamadı. "
                        "models/embeddings/ altına bir model kurun veya "
                        ".env'de EMBEDDING_MODEL tanımlayın."
                    ),
                    "type": "invalid_request_error",
                    "param": "model",
                    "code": "model_not_found",
                }
            },
        )

    backend = embedding_backend.get_backend(path=model_path)
    if not backend.is_available:
        return _openai_error(
            400,
            "Embedding desteği kapalı: sentence-transformers kurulu değil veya AI hub devre dışı.",
            code="embeddings_not_available",
        )

    texts = [req.input] if isinstance(req.input, str) else list(req.input)
    if not texts:
        return _openai_error(400, "input boş olamaz.", code="invalid_input")

    try:
        vectors = backend.embed(texts)
    except Exception as exc:
        return _openai_error(
            503,
            f"Embedding üretilemedi: {exc}{_model_hint(exc)}",
            code="embedding_failure",
        )

    prompt_tokens = _estimate_tokens(texts)
    ctx.charge("embedding", model_id=req.model, prompt_tokens=prompt_tokens)

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ],
        "model": req.model,
        "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens},
    }