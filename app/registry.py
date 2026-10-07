# ─────────────────────────────────────────────────────────────
#  Bölüm:    Birleşik Model Sayaç (Registry)
#  Dosya:    app/registry.py
#  Amaç:     Disk üzerindeki tüm model biçimlerini (GGUF, safetensors,
#            onnx, ctranslate2, embedding) tek bir katalog içinde sunar.
#  Mekanik:  - GGUF dosyaları model yöneticisi (ModelManager) tarafından
#              yönetilir; tipik olarak models/ kökünde veya models/reasoning/
#              altında durur. Kategori "reasoning" klasörden ya da dosya
#              adı deseninden (r1/thinking/reasoning) çıkarılır.
#            - Tip bazlı (typed) modeller için alt klasör sözleşmesi:
#                models/safetensors/  -> difüzör (diffusers) pipeline -> image/video/music
#                models/onnx/         -> ONNX aktarımı                  -> generic
#                models/ct2/          -> faster-whisper (ctranslate2)   -> stt
#                models/embeddings/   -> sentence-transformers          -> embedding
#            - Registry taraflı disk taraması yalnızca bu alt klasörlere
#              bakar; GGUF listesi model yöneticisinden gelir (bellek
#              yükleme/memory_mode bilgisini yeniden üretmez).
#  Kullanım: registry.list_entries()           -> tüm tip, birden fazla
#            registry.scan_typed()             -> yalnızca tip bazlı
#            registry.category_for_gguf(id, path, models_dir)
#            registry.get_entry(model_id)
# ─────────────────────────────────────────────────────────────

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.config import settings

# Model biçimi sabitleri (HF indiriciyle aynı adlandırma)
KIND_GGUF = "gguf"
KIND_SAFETENSORS = "safetensors"
KIND_ONNX = "onnx"
KIND_CT2 = "ct2"
KIND_EMBEDDINGS = "embeddings"

KINDS = (KIND_GGUF, KIND_SAFETENSORS, KIND_ONNX, KIND_CT2, KIND_EMBEDDINGS)

# Her biçimin models/ altındaki varsayılan klasör adı
KIND_SUBDIRS: dict[str, str] = {
    KIND_SAFETENSORS: KIND_SAFETENSORS,
    KIND_ONNX: KIND_ONNX,
    KIND_CT2: KIND_CT2,
    KIND_EMBEDDINGS: KIND_EMBEDDINGS,
}

# Kategoriler (tarife görev adıyla aynı tutulur)
CAT_CHAT = "chat"
CAT_REASONING = "reasoning"
CAT_EMBEDDING = "embedding"
CAT_IMAGE = "image"
CAT_VIDEO = "video"
CAT_MUSIC = "music"
CAT_STT = "stt"
CAT_GENERIC = "generic"


@dataclass
class ModelEntry:
    """Diskteki tek bir model kaydı (biçimden bağımsız)."""

    model_id: str            # benzersiz tanımlayıcı (dosya/dizin adı)
    kind: str                # Biçim: gguf/safetensors/onnx/ct2/embeddings
    category: str            # İş kategorisi: chat/reasoning/embedding/...
    path: Path               # Dosya ya da model dizini
    size_bytes: int = 0
    loaded: bool = False     # (yalnızca GGUF için anlamlı)
    memory_mode: str = ""
    extra: dict = field(default_factory=dict)


# Reasoning (düşünme) modeli deseni: dosya adındaki işaretler
_REASONING_RE = re.compile(
    r"(^|[\W_])(r1|thinking|reasoning)(?=$|[\W_])", re.IGNORECASE
)


# ------------------------------------------------------------------
# Sınıflandırma yardımcıları
# ------------------------------------------------------------------

def _looks_like_reasoning(model_id: str, rel_parts: tuple) -> bool:
    """Models dizinine göreli olarak bu model bir reasoning modeli mi?

    Either the folder path contains "reasoning" veya dosya adı r1/thinking/
    reasoning işaretleri taşır. Klasör sözleşmesi (models/reasoning/) en
    kesin yoldur; ad deseni ikincil bir sezgisel olarak kullanılır.
    """
    if any(part.lower() == "reasoning" for part in rel_parts):
        return True
    return _REASONING_RE.search(model_id) is not None


def category_for_gguf(model_id: str, path: Path, models_dir: Path | str) -> str:
    """Bir GGUF modelin iş kategorisini döner (reasoning | chat)."""
    models_dir = Path(models_dir)
    try:
        rel_parts = Path(path).relative_to(models_dir).parts
    except ValueError:
        rel_parts = ()
    if _looks_like_reasoning(model_id, rel_parts):
        return CAT_REASONING
    return CAT_CHAT


def _class_names(model_index: dict) -> list[str]:
    """diffusers model_index.json içindeki tüm sınıf adlarını toplar.

    Hem eski düzlem (component -> "ClassName") hem yeni yapı
    (component -> {"class_name": ...}) desteklenir.
    """
    names: list[str] = []
    for value in model_index.values():
        if isinstance(value, str):
            names.append(value)
        elif isinstance(value, dict):
            cn = value.get("class_name") or value.get("_class_name")
            if isinstance(cn, str):
                names.append(cn)
    return names


def _safetensors_category(model_dir: Path) -> str:
    """Bir diffusers pipeline dizinini içeriğine göre kategorize eder."""
    index = {}
    try:
        index = json.loads((model_dir / "model_index.json").read_text("utf-8"))
    except Exception:
        pass
    class_names = " ".join(_class_names(index))
    upper = class_names.upper()
    if any(t in upper for t in ("TEXT2VIDEO", "TEXT-TO-VIDEO", "UNET3DCONDITION", "VIDEOGEN")):
        return CAT_VIDEO
    if any(t in upper for t in ("AUDIOLDM", "MUSICLDM", "MUSICGEN")):
        return CAT_MUSIC
    return CAT_IMAGE


def _dir_size(model_dir: Path) -> int:
    """Bir model dizinindeki düzenli dosyaların toplam boyutu (bayt)."""
    return sum(p.stat().st_size for p in model_dir.rglob("*") if p.is_file())


# ------------------------------------------------------------------
# Tip bazlı (typed) tarama
# ------------------------------------------------------------------

def _is_safetensors_pipeline(model_dir: Path) -> bool:
    return (model_dir / "model_index.json").is_file()


def _is_onnx_model(model_dir: Path) -> bool:
    return any(p.suffix == ".onnx" for p in model_dir.rglob("*.onnx"))


def _is_ct2_model(model_dir: Path) -> bool:
    return (model_dir / "config.json").is_file() and (model_dir / "model.bin").is_file()


def _is_embedding_model(model_dir: Path) -> bool:
    cfg = (model_dir / "config.json").is_file()
    if not cfg:
        return False
    return any(
        p.name in ("model.safetensors", "pytorch_model.bin", "model.ckpt")
        for p in model_dir.iterdir()
        if p.is_file()
    )


def scan_typed(models_dir: Optional[Path | str] = None) -> list[ModelEntry]:
    """Yalnızca tip bazlı (GGUF dışı) modelleri tarar.

    Sözleşmeye uygun alt klasörleri (safetensors/onnx/ct2/embeddings)
    inceler. GGUF listesi burada değil, model yöneticisindedir.
    """
    models_dir = Path(models_dir or settings.models_dir)
    if not models_dir.exists():
        return []
    entries: list[ModelEntry] = []

    def _scan_dir(kind: str, subdir: str, detect, category_for) -> None:
        base = models_dir / subdir
        if not base.is_dir():
            return
        for model_dir in sorted(base.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            try:
                if not detect(model_dir):
                    continue
            except Exception:
                continue
            category = category_for(model_dir)
            entries.append(
                ModelEntry(
                    model_id=model_dir.name,
                    kind=kind,
                    category=category,
                    path=model_dir,
                    size_bytes=_dir_size(model_dir),
                )
            )

    _scan_dir(KIND_SAFETENSORS, KIND_SAFETENSORS, _is_safetensors_pipeline, _safetensors_category)
    _scan_dir(KIND_ONNX, KIND_ONNX, _is_onnx_model, lambda _d: CAT_GENERIC)
    _scan_dir(KIND_CT2, KIND_CT2, _is_ct2_model, lambda _d: CAT_STT)
    _scan_dir(KIND_EMBEDDINGS, KIND_EMBEDDINGS, _is_embedding_model, lambda _d: CAT_EMBEDDING)
    return entries


def _gguf_entries(manager=None) -> list[ModelEntry]:
    """Model yöneticisindeki GGUF kayıtlarını kategoriyle birlikte döner."""
    from app.model_manager import get_manager

    manager = manager or get_manager()
    entries = []
    for info in manager.list_models():
        entries.append(
            ModelEntry(
                model_id=info.model_id,
                kind=KIND_GGUF,
                category=category_for_gguf(info.model_id, info.path, manager.models_dir),
                path=info.path,
                size_bytes=info.size_bytes,
                loaded=info.loaded,
                memory_mode=info.memory_mode,
            )
        )
    return entries


def list_entries(manager=None, include_typed: bool = True) -> list[ModelEntry]:
    """Tüm model kayıtlarını (GGUF + tip bazlı) tek liste olarak döner."""
    gguf = _gguf_entries(manager)
    if not include_typed:
        return gguf
    return gguf + scan_typed()


def get_typed_entry(model_id: str, models_dir: Optional[Path | str] = None) -> Optional[ModelEntry]:
    """Tip bazlı bir modeli adına göre bulur; yoksa None."""
    for entry in scan_typed(models_dir=models_dir):
        if entry.model_id == model_id:
            return entry
    return None


def get_entry(model_id: str, manager=None, models_dir: Optional[Path | str] = None) -> Optional[ModelEntry]:
    """Bir modelı biçimden bağımsız olarak bulur (GGUF önce)."""
    from app.model_manager import get_manager

    manager = manager or get_manager()
    for info in manager.list_models():
        if info.model_id == model_id:
            return ModelEntry(
                model_id=info.model_id,
                kind=KIND_GGUF,
                category=category_for_gguf(info.model_id, info.path, manager.models_dir),
                path=info.path,
                size_bytes=info.size_bytes,
                loaded=info.loaded,
                memory_mode=info.memory_mode,
            )
    return get_typed_entry(model_id, models_dir=models_dir)


# ------------------------------------------------------------------
# Silme (tip bazlı)
# ------------------------------------------------------------------

def delete_typed_model(model_id: str, models_dir: Optional[Path | str] = None) -> list[str]:
    """models/<kind>/<model_id> klasörünü siler (varsa).

    GGUF tipi model yöneticisi + hf_downloader tarafından yönetilir;
    bu işlev yalnızca tip bazlı (safetensors/onnx/ct2/embeddings) dizinleri
    temizler. Silinen kayıt yollarını döner.
    """
    models_dir = Path(models_dir or settings.models_dir)
    removed: list[str] = []
    for kind, subdir in KIND_SUBDIRS.items():
        target = models_dir / subdir / model_id
        if target.is_dir():
            try:
                shutil.rmtree(target)
                removed.append(str(target))
            except OSError:
                pass
    return removed