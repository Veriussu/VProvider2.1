# ─────────────────────────────────────────────────────────────
#  Bölüm:    STT (Speech-to-Text) Backend
#  Amaç:     faster-whisper (ctranslate2) ile ses metne dönüştürme
# ─────────────────────────────────────────────────────────────

import asyncio
import io
import logging
import os
import threading
import time
from typing import Optional, Tuple

import numpy as np
from app.config import settings

logger = logging.getLogger("vprovider")

MAX_AUDIO_SECONDS = 300  # Tek seferde işlenecek maksimum audio süresi (saniye)
STT_STORE_CAP = 20       # Panel deposunda tutulacak son transkripsiyon sayısı

# Panel seçiminde sunulan ses dosyaları için klasör
_AUDIO_STORE: dict[str, dict] = {}
_AUDIO_STORE_LOCK = threading.Lock()


class STTError(RuntimeError):
    """Ses-to-Text hataları (model yüklenemedi, ses bozuk, vb.)."""


def _store_transcription(key: str, text: str, language: str = "tr") -> None:
    """Transkripsiyonu çoktan-azan düzenle (LRU) saklar."""
    with _AUDIO_STORE_LOCK:
        _AUDIO_STORE[key] = {"text": text, "language": language, "timestamp": time.time()}
        while len(_AUDIO_STORE) > STT_STORE_CAP:
            _AUDIO_STORE.pop(next(iter(_AUDIO_STORE)))


def _get_stored_transcription(key: str) -> Optional[dict]:
    """Depodan transkripsiyonu döner (yoksa None)."""
    with _AUDIO_STORE_LOCK:
        return _AUDIO_STORE.get(key)


def _ensure_model_dir(model_name: str) -> str:
    """Modelin yerel yolu döner; yoksa HF'ten indirmeye çalışır."""
    model_path = settings.models_path / "ct2" / model_name
    if model_path.exists() and (model_path / "model.bin").exists():
        return str(model_path)

    # Model mevcut değilse, varsayılan modeli indirelim
    logger.info(f"Model {model_name} bulunamadı, varsayılan model kuruluyor...")
    # Not: Gerçek indirme admin API üzerinden yapılmalı, buradaki fallback sadece dev için
    default_model = settings.models_path / "ct2" / "default"
    if default_model.exists():
        return str(default_model)
    
    # Hiç model yoksa hata
    raise STTError(f"Model {model_name} mevcut değil ve indirilemedi.")


async def transcribe_audio(
    audio_bytes: bytes,
    model: str = "base",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    temperature: float = 0.0,
    response_format: str = "json",
    timestamp_granularities: list = ["segment"],
) -> dict:
    """
    Ses verisini metne çevirir (faster-whisper kullanarak).
    
    Args:
        audio_bytes: WAV formatında ses verisi
        model: ct2 model adı (base, small, medium, large-v2, vb.)
        language: Dil kodu (örn: "tr", "en") veya None (otomatik tespit)
        prompt: İlk parti için ipucu metni
        temperature: Örnekleme sıcaklığı (0.0 = deterministik)
        response_format: "json", "text", "srt", "vtt", "tsv"
        timestamp_granularities: ["word", "segment"] listesi
    
    Returns:
        OpenAI uyumlu transcription yanıtı
    """
    if not settings.stt_enabled:
        raise STTError("STT modülü kapalı (STT_ENABLED=false).")

    # Ses verisini numpy array'e çevir (16-bit PCM WAV varsayımı)
    try:
        # Basit WAV parsing - gerçekte daha iyi bir kütüphane kullanılmalı
        # Şimdilik audio_bytes zaten float32 veya int16 PCM olduğunu varsayalım
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        raise STTError(f"Ses verisi işlenemedi: {e}")

    # Audio süresini kontrol et
    duration_seconds = len(audio_np) / 16000  # 16kHz sample rate varsayımı
    if duration_seconds > MAX_AUDIO_SECONDS:
        raise STTError(f"Ses çok uzun: maksimum {MAX_AUDIO_SECONDS} saniye.")

    # faster-whisper modelini yükle
    try:
        from faster_whisper import WhisperModel
        
        model_path = _ensure_model_dir(model)
        
        # Model zaten yüklenmemişse yükle
        if not hasattr(transcribe_audio, "_model") or transcribe_audio._model is None or transcribe_audio._model_path != model_path:
            logger.info(f"STT modeli yükleniyor: {model_path}")
            device = "cuda" if settings.gpu_mode != "cpu" else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            transcribe_audio._model = WhisperModel(
                model_path, 
                device=device, 
                compute_type=compute_type
            )
            transcribe_audio._model_path = model_path
            
    except ImportError:
        raise STTError(
            "faster-whisper kurulu değil. Kurun: pip install faster-whisper "
            "(veya requirements-hub.txt'yi yükleyin)."
        )
    except Exception as e:
        logger.error(f"STT modeli yüklenemedi: {e}")
        raise STTError(f"STT modeli yüklenemedi: {e}")

    # Transkripsiyonu çalıştır
    try:
        segments, info = transcribe_audio._model.transcribe(
            audio_np,
            language=language,
            initial_prompt=prompt,
            temperature=temperature,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        
        # Sonuçları topla
        segments_list = []
        full_text = []
        
        for segment in segments:
            segment_dict = {
                "id": len(segments_list),
                "seek": segment.seek,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": [],
                "temperature": temperature,
                "avg_logprob": segment.avg_logprob,
                "compression_ratio": segment.compression_ratio,
                "no_speech_prob": segment.no_speech_prob,
            }
            segments_list.append(segment_dict)
            full_text.append(segment.text)
        
        text = "".join(full_text).strip()
        
        # Yanıtı OpenAI formatında hazırla
        result = {
            "task": "transcribe",
            "language": info.language,
            "duration": info.duration,
            "text": text,
            "segments": segments_list,
        }
        
        # response_format'a göre çıktıyı ayarla
        if response_format == "text":
            return text
        elif response_format in ["srt", "vtt", "tsv"]:
            # Basit implementasyon - gerçekte daha kapsamlı olmalı
            if response_format == "srt":
                srt_lines = []
                for i, seg in enumerate(segments_list, 1):
                    start = seg["start"]
                    end = seg["end"]
                    srt_lines.append(
                        f"{i}\n"
                        f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d},{int((start%1)*1000):03d} --> "
                        f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d},{int((end%1)*1000):03d}\n"
                        f"{seg['text']}\n"
                    )
                return "\n".join(srt_lines)
            elif response_format == "vtt":
                vtt_lines = ["WEBVTT\n"]
                for seg in segments_list:
                    start = seg["start"]
                    end = seg["end"]
                    vtt_lines.append(
                        f"{int(start//3600):02d}:{int((start%3600)//60):02d}:{int(start%60):02d}.{int((start%1)*1000):03d} --> "
                        f"{int(end//3600):02d}:{int((end%3600)//60):02d}:{int(end%60):02d}.{int((end%1)*1000):03d}\n"
                        f"{seg['text']}\n"
                    )
                return "\n".join(vtt_lines)
            elif response_format == "tsv":
                tsv_lines = ["start\tend\ttext"]
                for seg in segments_list:
                    tsv_lines.append(f"{seg['start']:.3f}\t{seg['end']:.3f}\t{seg['text']}")
                return "\n".join(tsv_lines)
        
        # JSON formatı için tüm segmentleri ekle
        if response_format == "json":
            result["segments"] = segments_list
            return result
        
        # Varsayılan JSON
        return result
        
    except Exception as e:
        logger.error(f"STT transkripsiyonu başarısız: {e}")
        raise STTError(f"Transkripsiyon başarısız: {e}")


async def store_transcription(key: str, text: str, language: str = "tr") -> None:
    """Transkripsiyonu depoda saklar (panel için)."""
    _store_transcription(key, text, language)


async def get_stored_transcription(key: str) -> Optional[dict]:
    """Depodan transkripsiyonu döner."""
    return _get_stored_transcription(key)


async def stt_status() -> dict:
    """Panel için STT durum bilgisi."""
    try:
        from faster_whisper import WhisperModel
        faster_whisper_available = True
    except ImportError:
        faster_whisper_available = False
    
    return {
        "enabled": settings.stt_enabled,
        "faster_whisper_installed": faster_whisper_available,
        "default_model": getattr(settings, "stt_model", "base"),
        "model_dir_exists": (settings.models_path / "ct2").exists(),
    }


# Backwards compatibility alias
speech_to_text = transcribe_audio
