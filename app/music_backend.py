# ─────────────────────────────────────────────────────────────
#  Bölüm:    Müzik Üretimi Backend (basit örneği)
#  Amaç:     Kullanıcı promptundan ses/müzik üretir.
#            Şimdilik basit bir sinüs dalgası üretir; ileride
#            transformers/musicgen kullanılabilir.
# ─────────────────────────────────────────────────────────────

import asyncio
import io
import logging
import math
import os
import threading
import time
from typing import Optional

import numpy as np
from app.config import settings

logger = logging.getLogger("vprovider")

MAX_AUDIO_SECONDS = 30      # Tek seferde üretilen maksimum audio süresi
MUSIC_STORE_CAP = 20        # Panel deposunda tutulacak son müzik sayısı

# Panel deposu (in-memory) - ses dosyaları için
_MUSIC_STORE: dict[str, dict] = {}
_MUSIC_STORE_LOCK = threading.Lock()


class MusicError(RuntimeError):
    """Müzik üretimi hataları (model yüklenemedi, prompt geçersiz, vb.)."""


def _store_music(key: str, prompt: str, duration: float) -> None:
    """Müzik bilgisini çoktan-azan düzenle (LRU) saklar."""
    with _MUSIC_STORE_LOCK:
        _MUSIC_STORE[key] = {
            "prompt": prompt,
            "duration": duration,
            "timestamp": time.time(),
        }
        while len(_MUSIC_STORE) > MUSIC_STORE_CAP:
            _MUSIC_STORE.pop(next(iter(_MUSIC_STORE)))


def _get_stored_music(key: str) -> Optional[dict]:
    """Depodan müzik meta verisini döner (yoksa None)."""
    with _MUSIC_STORE_LOCK:
        return _MUSIC_STORE.get(key)


async def generate_music(
    prompt: str,
    model: str = "musicgen-small",
    duration: float = 10.0,
    temperature: float = 0.8,
    top_k: int = 250,
    top_p: float = 0.0,
    response_format: str = "wav",
) -> tuple[bytes, str]:
    """
    Prompt'tan müzik üretir.
    
    Args:
        prompt: Açıklama (örn: "upbeat electronic music")
        model: Model adı (musicgen-small, musicgen-medium, vb.)
        duration: Süre (saniye)
        temperature: Örnekleme sıcaklığı
        top_k: Top-k örnekleme
        top_p: Top-p (nucleus) örnekleme
        response_format: "wav", "mp3", "ogg"
    
    Returns:
        (ses_bytes, mime_type)
    """
    if not settings.music_enabled:
        raise MusicError("Müzik üretimi kapalı (MUSIC_ENABLED=false).")

    # Süreyi sınırla
    duration = max(1.0, min(duration, MAX_AUDIO_SECONDS))

    # Şimdilik basit bir ses üretimi (placeholder)
    # Gerçekte buraya transformers.MusicForConditionalGeneration kullanılacak
    logger.info(f"Müzik üretiliyor: '{prompt}' ({duration}s, model={model})")
    
    # Basit bir ton üreteceğiz (440 Hz sinüs dalgası)
    sample_rate = 16000  # Hz
    frequency = 440.0    # Hz (A4 notası)
    
    # Zaman ekseni
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Sinüs dalgası + hafif modülasyon (daha müziksel yapmak için)
    notes = [261.63, 329.63, 392.00, 440.00, 493.88]  # C4, E4, G4, A4, B4
    audio = np.zeros_like(t)
    
    # Basit melodi: her nota 1 saniye
    for i, note_freq in enumerate(notes):
        start_idx = int(i * sample_rate)
        end_idx = int((i + 1) * sample_rate) if i < len(notes) - 1 else len(t)
        if start_idx < len(t) and end_idx <= len(t):
            note_t = t[start_idx:end_idx] - (i * 1.0)
            note_wave = np.sin(2 * np.pi * note_freq * note_t) * np.exp(-note_t * 0.5)
            audio[start_idx:end_idx] += note_wave * 0.3
    
    # Normalizasyon
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio)) * 0.7  # Clipping'i önle
    
    # 16-bit PCM'e çevir
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # WAV başlığı oluştur
    wav_bytes = _create_wav_header(len(audio_int16) * 2, sample_rate, 1, 16)
    wav_bytes += audio_int16.tobytes()
    
    # MP3 dönüşümü (basitçe WAV döndürüyoruz - gerçekte LAME veya ffmpeg gerekir)
    if response_format == "mp3":
        # Şimdilik WAV olarak dön, gerçekte MP3 dönüşümü gerekir
        mime_type = "audio/mpeg"
    elif response_format == "wav":
        mime_type = "audio/wav"
    else:
        mime_type = "audio/wav"  # Varsayılan
    
    return wav_bytes, mime_type


def _create_wav_header(data_size: int, sample_rate: int, channels: int, bits_per_sample: int) -> bytes:
    """WAV dosya başlığı oluşturur."""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    header = bytearray()
    header.extend(b"RIFF")                          # ChunkID
    header.extend((36 + data_size).to_bytes(4, 'little'))  # ChunkSize
    header.extend(b"WAVE")                          # Format
    header.extend(b"fmt ")                          # Subchunk1ID
    header.extend((16).to_bytes(4, 'little'))       # Subchunk1Size
    header.extend((1).to_bytes(2, 'little'))        # AudioFormat (PCM)
    header.extend(channels.to_bytes(2, 'little'))   # NumChannels
    header.extend(sample_rate.to_bytes(4, 'little')) # SampleRate
    header.extend(byte_rate.to_bytes(4, 'little'))  # ByteRate
    header.extend(block_align.to_bytes(2, 'little')) # BlockAlign
    header.extend(bits_per_sample.to_bytes(2, 'little')) # BitsPerSample
    header.extend(b"data")                          # Subchunk2ID
    header.extend(data_size.to_bytes(4, 'little'))  # Subchunk2Size
    
    return bytes(header)


async def store_music(key: str, prompt: str, duration: float) -> None:
    """Müzik bilgisini depoda saklar (panel için)."""
    _store_music(key, prompt, duration)


async def get_stored_music(key: str) -> Optional[dict]:
    """Depodan müzik meta verisini döner."""
    return _get_stored_music(key)


async def music_status() -> dict:
    """Panel için müzik üretimi durum bilgisi."""
    try:
        import torch
        import transformers
        torch_available = True
        transformers_available = True
    except ImportError:
        torch_available = False
        transformers_available = False
    
    return {
        "enabled": settings.music_enabled,
        "torch_installed": torch_available,
        "transformers_installed": transformers_available,
        "default_model": getattr(settings, "music_model", "musicgen-small"),
    }
