# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yapılandırma Katmanı
#  Dosya:    app/config.py
#  Amaç:     .env dosyasındaki tüm ayarları okur ve tek noktadan sunar
#  Mekanik:  pydantic-settings, .env'i okur ve Settings sınıfına eşler.
#            Tüm modüller ortadaki "settings" nesnesini kullanır.
#  Kullanım: from app.config import settings
# ─────────────────────────────────────────────────────────────

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────
#  SABİT SİTE KİMLİĞİ — kod içine gömülüdür, değiştirilemez.
#  Projenin temel kimliği gibi düşünün: her kurulum aynı değerleri
#  taşır, panelden/DB'den değiştirilmez ve .env'ye taşınmaz.
# ─────────────────────────────────────────────────────────────
SITE_IDENTITY: dict = {
    "project_name": "VProvider",
    "github_url": "https://github.com/Veriussu/",
    "developer_domain": "https://veriussu.com",
    "docs_url": "https://vprovider.veriussu.com/Docs",
    "contact_email": "vprovider@veriussu.com",
}


class Settings(BaseSettings):
    """Uygulama geneli ayarlar. Değerler .env dosyasından gelir."""

    # Uygulama kimliği
    app_name: str = "VProvider"

    # Sunucu dinleme adresi ve portu
    host: str = "0.0.0.0"
    port: int = 9055

    # Bellek stratejisi: keep (her daim hazır) | dynamic (boşta boşalt)
    # idle_timeout_minutes = 0 -> kullanım bitince anında GPU'dan boşalt
    # (tek GPU'da birden çok model arasında geçiş için idealdir).
    memory_mode: str = "dynamic"
    idle_timeout_minutes: int = 0

    # Model kullanılırken sistem RAM'inin en fazla bu yüzdesi kullanılabilir.
    # 0 = sınırsız (eski davranış). idle'da zaten hiçbir model bellekte yoktur.
    memory_limit_pct: int = 90

    # Çalışma klasörleri (proje köküne göre otomatik tam yol üretilir)
    models_dir: str = "models"
    data_dir: str = "data"
    database_path: str = "data/vprovider.db"

    # LLM çalıştırma parametreleri
    context_size: int = 4096
    # GPU kullanım modu: auto (donanıma göre) | cuda | rocm | sycl | vulkan | metal | cpu
    gpu_mode: str = "auto"
    gpu_layers: int = -1        # -1 = otomatik (VRAM'e göre); pozitif = sabit
    threads: int = 0            # 0 = otomatik

    # ComfyUI (görsel üretim motoru) köprüsü — isteğe bağlı modül
    comfyui_enabled: bool = False
    comfyui_host: str = "127.0.0.1"
    comfyui_port: int = 8188
    comfyui_dir: str = ""       # ComfyUI kurulum klasörü (scriptler için)
    comfyui_default_checkpoint: str = ""
    comfyui_default_negative: str = "blur, ugly, low quality, watermark"

    # Ses üretimi (TTS) köprüsü — isteğe bağlı modül
    tts_enabled: bool = False
    tts_engine: str = "edge"            # edge-tts (Microsoft çevrimiçi motor)
    tts_voice: str = "tr-TR-EmelNeural" # varsayılan Türkçe kadın sesi
    tts_voice_rate: str = "+0%"         # edge-tts hız ayarı (+0%/-10%/+20%...)

    # İnternet erişimi (web araçları) — yalnızca çağrı anında ağ kullanır
    # web_tools_enabled varsayılanı .env'den gelir; paneldeki toggle DB'de
    # kalıcıdır ve bu varsayılanı ezer.
    web_tools_enabled: bool = True
    web_search_max_results: int = 5     # metin sonucu sayısı
    web_search_max_images: int = 4      # görsel sonucu sayısı
    web_search_region: str = "tr-tr"    # DuckDuckGo bölgesi (tr-tr | us-en ...)
    web_fetch_max_chars: int = 8000     # sayfa metninden alınacak karakter
    web_fetch_max_bytes: int = 2_097_152  # sayfa için ASCII üst sınır (2 MB)
    web_tool_timeout: int = 15          # ağ çağrısı zaman aşımı (saniye)
    web_tools_max_rounds: int = 6       # sunucu-side araç döngüsü üst sınırı

    # HuggingFace (isteğe bağlı: gated repolar için)
    hf_token: str = ""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def models_path(self) -> Path:
        """Modeller klasörünün mutlak yolu."""
        return Path(self.models_dir)

    @property
    def database_file(self) -> Path:
        """Veritabanı dosyasının mutlak yolu."""
        return Path(self.database_path)


# Uygulama genelinde kullanılan tek settings örneği
settings = Settings()