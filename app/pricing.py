# ─────────────────────────────────────────────────────────────
#  Bölüm:    Tarife (Pricing) Katmanı
#  Dosya:    app/pricing.py
#  Amaç:     OpenAI-uyumlu API tüketimini kredi cinsinden fiyatlar.
#            Bakiye (balance) sistemi bu tarifeyle düşülür; aylık kota
#            yalnızca token sayar. Tarife kod-ağacındaki sabitlerden
#            gelir; istenirse ortam değişkeniyle üzerine yazılabilir.
#  Mekanik:  - Metin işleri token/1K üzerinden fiyatlanır (chat, embeddings).
#            - Görsel/video/müzik üretimleri adet, TTS saniye üzerinden.
#            - compute_cost(task, ...) -> kredi; kullanıcı tarafından
#              bakiye düşülebilir veya yalnızca raporlama amaçlıdır.
# ─────────────────────────────────────────────────────────────

from typing import Optional

# Kredi (balance) birimi başına 1/1000 hassasiyet; tüm tarifeler "kredi" cinsindendir.
# Tokens are billed per 1K tokens; media units are billed per item.

# Varsayılan tarife (kredi cinsinden):
#   token_1k      : metin/embedding token başına (prompt ve completion ayrı)
#   per_image     : görsel üretim başına
#   per_video     : video üretim başına
#   per_audio_sec : TTS saniyesi başına
#   per_music     : müzik üretim başına
DEFAULT_PRICES: dict[str, dict] = {
    "chat":       {"token_1k": 0.002,   "per_image": 0.0, "per_video": 0.0, "per_audio_sec": 0.0, "per_music": 0.0},
    "completion": {"token_1k": 0.002,   "per_image": 0.0, "per_video": 0.0, "per_audio_sec": 0.0, "per_music": 0.0},
    "responses":  {"token_1k": 0.002,   "per_image": 0.0, "per_video": 0.0, "per_audio_sec": 0.0, "per_music": 0.0},
    "reasoning":  {"token_1k": 0.004,   "per_image": 0.0, "per_video": 0.0, "per_audio_sec": 0.0, "per_music": 0.0},
    "embedding":  {"token_1k": 0.0001,  "per_image": 0.0, "per_video": 0.0, "per_audio_sec": 0.0, "per_music": 0.0},
    "image":      {"token_1k": 0.0,     "per_image": 0.04, "per_video": 0.0, "per_audio_sec": 0.0, "per_music": 0.0},
    "video":      {"token_1k": 0.0,     "per_image": 0.0, "per_video": 0.5, "per_audio_sec": 0.0, "per_music": 0.0},
    "tts":        {"token_1k": 0.0,     "per_image": 0.0, "per_video": 0.0, "per_audio_sec": 0.0005, "per_music": 0.0},
    "stt":        {"token_1k": 0.002,   "per_image": 0.0, "per_video": 0.0, "per_audio_sec": 0.0, "per_music": 0.0},
    "music":      {"token_1k": 0.0,     "per_image": 0.0, "per_video": 0.0, "per_audio_sec": 0.0, "per_music": 0.05},
}

# Bilinmeyen görevler chat tarifesiyle fiyatlanır (zararsız varsayılan).
DEFAULT_TASK = "chat"


def get_prices(task: str) -> dict:
    """Görev adına göre tarife sözlüğünü döner (bilinmiyorsa varsayılan)."""
    return DEFAULT_PRICES.get(task, DEFAULT_PRICES[DEFAULT_TASK])


def compute_cost(
    task: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    images: int = 0,
    videos: int = 0,
    audio_sec: float = 0.0,
    music: int = 0,
) -> float:
    """Verilen kullanımın kredi maliyetini hesaplar."""
    p = get_prices(task)
    cost = 0.0
    cost += (prompt_tokens + completion_tokens) / 1000.0 * p["token_1k"]
    cost += images * p["per_image"]
    cost += videos * p["per_video"]
    cost += audio_sec * p["per_audio_sec"]
    cost += music * p["per_music"]
    # 9 ondalık: ucuz tarifelerde (embedding 0.0001/1K) tek tek küçük
    # istekler yuvarlamada sıfırlanıp ücretsiz görünmesin.
    return round(cost, 9)