# ─────────────────────────────────────────────────────────────
#  Bölüm:    AI Hub — Ağır Modül Kabı (import-guard)
#  Dosya:    app/ai_hub.py
#  Amaç:     torch/transformers/diffusers gibi ağır paketleri KESİNLİKLE
#            modül yüklenirken içe aktarmaz; yalnızca talep edildiğinde
#            (lazy) yükler. Paket kurulu değilse çekirdek aynen çalışır.
#  Mekanik:  - available(name)  : kurulu mu / toggle açık mı.
#            - get(name)        : paketi döner (kuruluysa) veya None.
#            - device()         : uygun hesaplama cihazı (cuda/mps/cpu).
#            - status()         : panel "sistem" ekranı için özet.
#  Kullanım: if ai_hub.get("diffusers") is None: → modül yok, sessiz kapan.
# ─────────────────────────────────────────────────────────────

import importlib
import importlib.util
from typing import Optional

from app.config import settings

# capability -> (paket adı, "ne işe yaradığı" açıklaması)
_CAPABILITIES: dict[str, tuple[str, str]] = {
    "image":   ("diffusers", "Görsel üretimi (SD/SDXL)"),
    "video":   ("diffusers", "Video üretimi (AnimateDiff benzeri in-process)"),
    "stt":     ("faster_whisper", "Konuşma tanıma (Whisper)"),
    "music":   ("transformers", "Müzik/ses üretimi (Musicgen benzeri)"),
    "embedding": ("sentence_transformers", "Metin gömme (embeddings)"),
    "transformers": ("transformers", "Genel dönüştürücü modelleri"),
}

# Paket yüklenebilirliği: bir kez kontrol edilip saklanır
_probed: dict[str, bool] = {}
_cached: dict[str, Optional[object]] = {}


def _probe(name: str) -> bool:
    """Paketin import edilebilir olup olmadığını döner (bir kez ölçer)."""
    if name not in _probed:
        _probed[name] = importlib.util.find_spec(name) is not None
    return _probed[name]


def enabled() -> bool:
    """Hub açık mı? (settings.ai_hub_enabled)"""
    return bool(settings.ai_hub_enabled)


def available(capability: str) -> bool:
    """Verilen yetenek kurulu + açık mı; bilinmeyen yetenek False."""
    if not enabled():
        return False
    entry = _CAPABILITIES.get(capability)
    if entry is None:
        return False
    return _probe(entry[0])


def get(name: str) -> Optional[object]:
    """Paket modülünü döner; yoksa None (asla hata fırlatmaz)."""
    if not enabled():
        return None
    if name in _cached:
        return _cached[name]
    if not _probe(name):
        _cached[name] = None
        return None
    try:
        mod = importlib.import_module(name)
    except Exception:
        _cached[name] = None
        return None
    _cached[name] = mod
    return mod


def device() -> str:
    """Uygun hesaplama cihazı adı: cuda | mps | cpu."""
    torch = get("torch")
    if torch is None:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def status() -> dict:
    """Panel "Sistem" ekranı için hub özeti (kurulu fakat kapalı satırı da verir)."""
    out = {"enabled": enabled(), "device": device(), "capabilities": {}}
    for cap, (_pkg, label) in _CAPABILITIES.items():
        if cap in ("image", "video"):
            # aynı paket; görsel/video tek satırda gösterilir
            continue
        out["capabilities"][cap] = {
            "label": label,
            "available": available(cap),
        }
    image_ok = available("image")
    out["capabilities"]["image"] = {"label": "Görsel üretimi (SD/SDXL)", "available": image_ok}
    out["capabilities"]["video"] = {
        "label": "Video üretimi (in-process)",
        "available": image_ok and get("torch") is not None,
    }
    return out