# ─────────────────────────────────────────────────────────────
#  Bölüm:    Medya Motoru (in-process diffusers)
#  Dosya:    app/diffusers_backend.py
#  Amaç:     models/safetensors/ altındaki difüzör modellerini sunucu içinde
#            çalıştırır: görsel (metin→görsel, görsel→görsel) ve video üretir.
#            ComfyUI köprüsüne gerek kalmadan /v1/images/* ve /v1/videos/*
#            uç noktalarına hizmet verir.
#  Mekanik:  - Registry (app/registry.py) model dizinlerini ve kategorilerini
#              verir; burada yalnızca image/video kategorileri yönetilir.
#            - Pipeline'lar tembel (lazy) yüklenir; ağır paketler yalnızca
#              ilk çağrıda içe aktarılır (ai_hub import-guard).
#            - Her model için asyncio.Lock: aynı pipeline eşzamanlı iki işlem
#              almaz (tek GPU paylaşımı).
#            - Bellek modu GGUF yöneticisiyle aynıdır: keep (kalıcı) /
#              dynamic (işlem bitince boşalt; idle_timeout_minutes=0 ise anında).
#            - Çıktı: görseller PNG baytı, video GIF baytı olarak döner.
#  Kullanım: manager = get_media_manager()
#            res = await manager.generate("sd-turbo", "bir kedi", size="512x512")
# ─────────────────────────────────────────────────────────────

import asyncio
import gc
import io
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from app import ai_hub, registry
from app.config import settings

# Bu yönetici yalnızca şu kategorileri üretir
SUPPORTED_CATEGORIES = (registry.CAT_IMAGE, registry.CAT_VIDEO)

# GIF kare hızı (fps) — video çıktısı GIF olarak döner
DEFAULT_FPS = 8
# PNG/JPEG kalite ayarı yerine PNG (kayıpsız) kullanılır


class MediaError(Exception):
    """Medya üretiminde oluşan, kullanıcıya anlamlı mesaj taşıyan hata."""


class MediaUnavailableError(MediaError):
    """Gerekli paket kurulu değil ya da yapay zeka hub'ı kapalı."""


@dataclass
class MediaInfo:
    """Diskteki tek bir medya (görsel/video) modelinin bilgisi."""

    model_id: str
    path: Path
    category: str            # image | video
    size_bytes: int = 0
    loaded: bool = False
    memory_mode: str = ""
    task: str = "txt2img"    # yüklü pipeline görevi (txt2img/img2img/video)


@dataclass
class MediaResult:
    """Üretim sonucu: görseller ve/veya video."""

    model_id: str
    images: list[bytes] = field(default_factory=list)          # PNG baytları
    video: Optional[dict] = None                                # {bytes,mime,width,height,frames}


# Pipeline üretim fonksiyonu: (MediaInfo, task) -> pipeline
# Yalnızca "diskten yükle" sorumluluğundadır; cihaza taşıma ve bellek
# ayarları yöneticinin (_prepare) işidir; böylece testlerde sahte pipeline
# kullanılabilir.
PipelineFactory = Callable[[MediaInfo, str], object]


# ------------------------------------------------------------------
# Yardımcılar
# ------------------------------------------------------------------

def parse_size(size: str) -> tuple[int, int]:
    """'512x512' biçimini (genişlik, yükseklik) çiftine çevirir.

    Diffusions modelleri 8'in katı boyut ister; değerler 64'e yuvarlanır ve
    en az 64'e çekilir. Geçersiz biçimde varsayılan 512x512 kullanılır.
    """
    try:
        w, h = (int(x) for x in str(size).lower().split("x", 1))
    except Exception:
        return 512, 512
    w = max(64, int(round(w / 64)) * 64)
    h = max(64, int(round(h / 64)) * 64)
    return w, h


def _torch():
    """torch modülünü döner (kurulu değilse None)."""
    return ai_hub.get("torch")


def _dtype_name(device: str) -> str:
    """Cihaza göre ağırlık veri tipi adı (CPU'da float32 güvenli)."""
    return "float32" if device == "cpu" else "float16"


def _seed_generator(device: str, seed: int):
    """Tekrarlanabilir üretim için torch üreticisi (seed<0 ise rastgele)."""
    torch = _torch()
    if torch is None:
        return None
    actual = seed if seed is not None and seed >= 0 else random.randint(0, 2**31 - 1)
    try:
        gen_device = device if device != "cpu" else "cpu"
        return torch.Generator(device=gen_device).manual_seed(actual)
    except Exception:
        return torch.Generator().manual_seed(actual)


def encode_png(image) -> bytes:
    """PIL görselini PNG baytına çevirir."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def encode_gif(frames: list, fps: int = DEFAULT_FPS, loop: int = 0) -> bytes:
    """Kare listesini GIF baytına çevirir (süre kare hızından hesaplanır)."""
    if not frames:
        raise MediaError("Video üretimi hiç kare döndürmedi.")
    buf = io.BytesIO()
    duration = max(20, int(1000 / max(1, fps)))
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=list(frames[1:]),
        duration=duration,
        loop=loop,
    )
    return buf.getvalue()


def _supports(pipeline, name: str) -> bool:
    """Pipeline'ın çağrı imzası verilen parametreyi kabul ediyor mu?"""
    import inspect

    try:
        return name in inspect.signature(pipeline.__call__).parameters
    except (TypeError, ValueError):
        return False


# ------------------------------------------------------------------
# Yönetici
# ------------------------------------------------------------------

def _has_fp16_variant(model_dir: Path) -> bool:
    """Model dizininde .fp16.safetensors varyantı var mı?

    Panelden indirilen modern repolar yalnızca fp16 varyantını indirebilir
    (bkz. hf_downloader._safetensors_allow_patterns); bu durumda yüklerken
    variant="fp16" verilmelidir, aksi halde transformers "model.safetensors"
    arar ve bulamaz.
    """
    try:
        return any(
            p.name.endswith(".fp16.safetensors")
            for p in model_dir.rglob("*.safetensors")
        )
    except OSError:
        return False


class MediaManager:
    """Görsel/video modellerinin yükleme, üretim ve bellek yönetimini yürütür."""

    MEMORY_KEEP = "keep"
    MEMORY_DYNAMIC = "dynamic"

    def __init__(
        self,
        models_dir: Optional[Path | str] = None,
        pipeline_factory: Optional[PipelineFactory] = None,
        memory_mode: str = "",
        idle_timeout_minutes: int = 0,
    ) -> None:
        self.models_dir = Path(models_dir or settings.models_dir)
        self._factory = pipeline_factory or self._default_factory
        self.memory_mode = memory_mode or settings.memory_mode
        self.idle_timeout_minutes = idle_timeout_minutes or settings.idle_timeout_minutes

        self._pipelines: dict[tuple, object] = {}   # (model_id, task) -> pipeline
        self._locks: dict[str, asyncio.Lock] = {}
        self._modes: dict[str, str] = {}
        self._idle_tasks: dict[str, asyncio.Task] = {}

    # ------------------------------------------------------------------
    # Katalog
    # ------------------------------------------------------------------

    def list_models(self) -> list[MediaInfo]:
        """Görsel/video kategorisindeki modelleri listeler."""
        return [
            MediaInfo(
                model_id=e.model_id,
                path=e.path,
                category=e.category,
                size_bytes=e.size_bytes,
                loaded=self._is_loaded(e.model_id),
                memory_mode=self._modes.get(e.model_id, self.memory_mode),
                task=self._active_task(e.model_id),
            )
            for e in registry.scan_typed(models_dir=self.models_dir)
            if e.kind == registry.KIND_SAFETENSORS and e.category in SUPPORTED_CATEGORIES
        ]

    def get_model(self, model_id: str) -> Optional[MediaInfo]:
        """Tek bir medya modelini bulur; yoksa None."""
        for info in self.list_models():
            if info.model_id == model_id:
                return info
        return None

    def ids_for(self, category: str) -> list[str]:
        """Verilen kategorideki model adlarını döner."""
        return [m.model_id for m in self.list_models() if m.category == category]

    # ------------------------------------------------------------------
    # Kilitler ve durum
    # ------------------------------------------------------------------

    def _lock_for(self, model_id: str) -> asyncio.Lock:
        if model_id not in self._locks:
            self._locks[model_id] = asyncio.Lock()
        return self._locks[model_id]

    def _is_loaded(self, model_id: str) -> bool:
        return any(k[0] == model_id for k in self._pipelines)

    def _active_task(self, model_id: str) -> str:
        for key in self._pipelines:
            if key[0] == model_id:
                return key[1]
        return "txt2img"

    def _mode_for(self, model_id: str) -> str:
        return self._modes.get(model_id, self.memory_mode)

    # ------------------------------------------------------------------
    # Pipeline yükleme
    # ------------------------------------------------------------------

    def _load_kwargs(self, path: str) -> dict:
        """from_pretrained anahtarlarını döner (dtype + gerekiyorsa varyant).

        Notlar:
          - use_safetensors YOK: zorlamak yalnızca .bin ağırlıklı eski
            repoları (ör. bazı test/deneme modelleri) kırıyor; diffusers
            varsa safetensors'i varsayılan olarak tercih eder.
          - diffusers 0.41'de torch_dtype deprecated oldu; "dtype" kullanılır,
            sürüm buna izin vermezse eski ada düşülür (bkz. _from_pretrained).
          - Sürüm tespiti için model YÜKLENMEZ; yalnızca gerçek çağrının
            hatasına bakılır (prob, modeli iki kez yüklerdi).
        """
        torch = _torch()
        kwargs = {"dtype": getattr(torch, _dtype_name(ai_hub.device()))}
        if _has_fp16_variant(Path(path)):
            kwargs["variant"] = "fp16"
        return kwargs

    def _from_pretrained(self, cls, path: str, **extra):
        """Pipeline sınıfını diskten yükler (dtype/variant uyumlu)."""
        kwargs = self._load_kwargs(path)
        kwargs.update(extra)
        try:
            return cls.from_pretrained(path, **kwargs)
        except TypeError as exc:
            message = str(exc)
            legacy = {("torch_dtype" if k == "dtype" else k): v for k, v in kwargs.items()}
            # Yalnızca dtype adına itiraz varsa eski biçimle tekrar dene
            if "dtype" in message and "torch_dtype" in legacy:
                return cls.from_pretrained(path, **legacy)
            raise

    def _default_factory(self, info: MediaInfo, task: str):
        """diffusers pipeline'ını diskten yükler (cihaz işi _prepare'da)."""
        diffusers = ai_hub.get("diffusers")
        torch = _torch()
        if diffusers is None or torch is None:
            raise MediaUnavailableError(
                "Görsel/video üretimi için diffusers + torch kurulu olmalı "
                "(requirements-hub.txt) ve AI_HUB_ENABLED=true olmalı."
            )

        path = str(info.path)
        if info.category == registry.CAT_IMAGE:
            cls = (
                diffusers.AutoPipelineForImage2Image
                if task == "img2img"
                else diffusers.AutoPipelineForText2Image
            )
            return self._from_pretrained(cls, path)
        return self._load_video_pipeline(diffusers, info, path)

    def _prepare(self, pipeline, device: str):
        """Pipeline'ı hesap cihazına taşır ve bellek ayarlarını açar."""
        move = getattr(pipeline, "to", None)
        if callable(move):
            pipeline = move(device) or pipeline
        # Bellek için güvenli varsayılanlar (VAE dilimleme/döşeme)
        for method in ("enable_vae_slicing", "enable_vae_tiling"):
            fn = getattr(pipeline, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
        return pipeline

    def _load_video_pipeline(self, diffusers, info: MediaInfo, path: str):
        """Video pipeline'ını yükler (AnimateDiff hareket modülü desteklenir).

        Sıra: 1) dizindeki hareket modülü (*_motion_adapter*) + AnimateDiff
        2) doğrudan AnimateDiff olarak yüklenebilen repo
        3) genel DiffusionPipeline (kare desteği olanlar)
        """
        motion = sorted(info.path.glob("*motion_adapter*.safetensors"))
        animatediff_cls = getattr(diffusers, "AnimateDiffPipeline", None)
        if motion and animatediff_cls is not None:
            return self._from_pretrained(animatediff_cls, path, motion_adapter=str(motion[0]))
        try:
            if animatediff_cls is not None:
                return self._from_pretrained(animatediff_cls, path)
        except Exception:
            pass  # normal difüzör reposu olabilir -> genel yola düş
        return self._from_pretrained(diffusers.DiffusionPipeline, path)

    def _from_pretrained(self, cls, path: str, **extra):
        """Pipeline sınıfını diskten yükler (dtype anahtarı sürüme uyarlanır)."""
        kwargs = self._load_kwargs(path)
        kwargs.update(extra)
        try:
            return cls.from_pretrained(path, **kwargs)
        except TypeError:
            legacy = {("torch_dtype" if k == "dtype" else k): v for k, v in kwargs.items()}
            return cls.from_pretrained(path, **legacy)

    async def _ensure_pipeline(self, model_id: str, task: str):
        """Pipeline'ı yükler veya hazır olanı döner (kilit tutulmalı)."""
        key = (model_id, task)
        if key in self._pipelines:
            return self._pipelines[key]
        info = self.get_model(model_id)
        if info is None:
            raise MediaError(f"'{model_id}' adında bir görsel/video modeli bulunamadı.")
        expected_task = "video" if info.category == registry.CAT_VIDEO else task
        device = ai_hub.device()
        # Ağır yükleme olay döngüsünü bloklamasın
        pipeline = await asyncio.to_thread(self._factory, info, expected_task)
        pipeline = self._prepare(pipeline, device)
        self._pipelines[key] = pipeline
        if (
            self._mode_for(model_id) == self.MEMORY_DYNAMIC
            and self.idle_timeout_minutes > 0
        ):
            self._reschedule_idle(model_id)
        return pipeline

    # ------------------------------------------------------------------
    # Yükleme / boşaltma
    # ------------------------------------------------------------------

    async def load(self, model_id: str, memory_mode: str = "") -> bool:
        """Modeli belleğe yükler (panelden çağrılabilir)."""
        info = self.get_model(model_id)
        if info is None:
            raise MediaError(f"'{model_id}' modeli bulunamadı.")
        task = "video" if info.category == registry.CAT_VIDEO else "txt2img"
        if memory_mode:
            self._modes[model_id] = memory_mode
        async with self._lock_for(model_id):
            await self._ensure_pipeline(model_id, task)
        return True

    async def unload(self, model_id: str) -> bool:
        """Modeli bellekten boşaltır (VRAM/RAM bırakır)."""
        self._cancel_idle(model_id)
        async with self._lock_for(model_id):
            keys = [k for k in self._pipelines if k[0] == model_id]
            if not keys:
                return False
            for key in keys:
                self._pipelines.pop(key, None)
            # torch önbelleğini ve Python referanslarını bırak
            gc.collect()
            try:
                torch = _torch()
                if torch is not None and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            return True

    def set_memory_mode(self, model_id: str, memory_mode: str) -> None:
        """Tek modelin bellek modunu değiştirir (keep/dynamic)."""
        if memory_mode not in (self.MEMORY_KEEP, self.MEMORY_DYNAMIC):
            raise ValueError("memory_mode yalnızca 'keep' veya 'dynamic' olabilir")
        self._modes[model_id] = memory_mode
        if memory_mode == self.MEMORY_KEEP:
            self._cancel_idle(model_id)
        elif self._is_loaded(model_id):
            self._release_after_use(model_id)

    def set_global_memory_mode(self, memory_mode: str) -> None:
        if memory_mode not in (self.MEMORY_KEEP, self.MEMORY_DYNAMIC):
            raise ValueError("memory_mode yalnızca 'keep' veya 'dynamic' olabilir")
        self.memory_mode = memory_mode

    # ------------------------------------------------------------------
    # Üretim
    # ------------------------------------------------------------------

    async def generate(
        self,
        model_id: str,
        prompt: str,
        negative_prompt: str = "",
        size: str = "512x512",
        steps: int = 20,
        cfg: float = 7.0,
        seed: int = -1,
        n: int = 1,
        image=None,
        frames: int = 16,
        fps: int = DEFAULT_FPS,
    ) -> MediaResult:
        """Görsel veya video üretir; PNG/GIF baytları döner.

        image verilirse görsel→görsel (img2img), yoksa metin→görsel çalışır.
        Video modellerinde kare sayısı frames ile belirlenir.
        """
        info = self.get_model(model_id)
        if info is None:
            raise MediaError(f"'{model_id}' modeli bulunamadı.")
        task = "video" if info.category == registry.CAT_VIDEO else ("img2img" if image is not None else "txt2img")

        async with self._lock_for(model_id):
            pipeline = await self._ensure_pipeline(model_id, task)
            try:
                if task == "video":
                    result = await asyncio.to_thread(
                        self._run_video, pipeline, info, prompt, negative_prompt,
                        size, steps, cfg, seed, frames, fps,
                    )
                else:
                    result = await asyncio.to_thread(
                        self._run_image, pipeline, prompt, negative_prompt,
                        size, steps, cfg, seed, n, image,
                    )
            finally:
                self._release_after_use(model_id)
        result.model_id = model_id
        return result

    def _run_image(
        self, pipeline, prompt, negative_prompt, size, steps, cfg, seed, n, image
    ) -> MediaResult:
        """Görsel üretimini thread içinde yürütür (olay döngüsünü bloklamaz)."""
        width, height = parse_size(size)
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg),
            "generator": _seed_generator(ai_hub.device(), seed),
        }
        if image is not None:
            kwargs["image"] = image
            if _supports(pipeline, "strength"):
                kwargs["strength"] = 0.75
        else:
            kwargs["num_images_per_prompt"] = max(1, int(n))
        if negative_prompt and _supports(pipeline, "negative_prompt"):
            kwargs["negative_prompt"] = negative_prompt
        if _supports(pipeline, "width") and _supports(pipeline, "height"):
            kwargs["width"] = width
            kwargs["height"] = height

        output = pipeline(**kwargs)
        images = list(getattr(output, "images", None) or [])
        if not images:
            raise MediaError("Pipeline görsel döndürmedi.")
        return MediaResult(model_id="", images=[encode_png(im) for im in images])

    def _run_video(
        self, pipeline, info, prompt, negative_prompt, size, steps, cfg, seed, frames, fps
    ) -> MediaResult:
        """Video üretimini thread içinde yürütür; GIF baytı üretir."""
        width, height = parse_size(size)
        kwargs = {
            "prompt": prompt,
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg),
            "generator": _seed_generator(ai_hub.device(), seed),
        }
        if negative_prompt and _supports(pipeline, "negative_prompt"):
            kwargs["negative_prompt"] = negative_prompt
        if _supports(pipeline, "num_frames"):
            kwargs["num_frames"] = int(frames)
        elif _supports(pipeline, "num_videos"):
            kwargs["num_videos"] = 1
        if _supports(pipeline, "width") and _supports(pipeline, "height"):
            kwargs["width"] = width
            kwargs["height"] = height

        output = pipeline(**kwargs)
        # AnimateDiff -> .frames (PIL listesi); diğerleri -> .frames/.images
        frames_out = getattr(output, "frames", None)
        if frames_out is None:
            frames_out = getattr(output, "images", None)
        if frames_out and isinstance(frames_out[0], (list, tuple)):
            frames_out = list(frames_out[0])   # [batch][kare] -> [kare]
        gif = encode_gif(list(frames_out or []), fps=fps)
        return MediaResult(
            model_id="",
            video={
                "bytes": gif,
                "mime": "image/gif",
                "width": width,
                "height": height,
                "frames": len(frames_out or []),
            },
        )

    # ------------------------------------------------------------------
    # Dinamik boşaltma
    # ------------------------------------------------------------------

    def _release_after_use(self, model_id: str) -> None:
        if self._mode_for(model_id) != self.MEMORY_DYNAMIC:
            return
        self._cancel_idle(model_id)
        if self.idle_timeout_minutes > 0:
            self._reschedule_idle(model_id)
        else:
            asyncio.get_running_loop().create_task(self._deferred_unload(model_id))

    async def _deferred_unload(self, model_id: str) -> None:
        await asyncio.sleep(0)  # çağıranın kilidini bırakmasına izin ver
        if self._is_loaded(model_id) and self._mode_for(model_id) == self.MEMORY_DYNAMIC:
            await self.unload(model_id)

    def _reschedule_idle(self, model_id: str) -> None:
        self._cancel_idle(model_id)
        self._idle_tasks[model_id] = asyncio.create_task(self._idle_unload(model_id))

    def _cancel_idle(self, model_id: str) -> None:
        task = self._idle_tasks.pop(model_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def _idle_unload(self, model_id: str) -> None:
        try:
            await asyncio.sleep(self.idle_timeout_minutes * 60)
        except asyncio.CancelledError:
            return
        self._idle_tasks.pop(model_id, None)
        if self._is_loaded(model_id) and self._mode_for(model_id) == self.MEMORY_DYNAMIC:
            await self.unload(model_id)

    async def shutdown(self) -> None:
        """Bekleyen boşaltma görevlerini iptal eder ve tüm modelleri boşaltır."""
        for task in list(self._idle_tasks.values()):
            if not task.done():
                task.cancel()
        self._idle_tasks.clear()
        for model_id in {k[0] for k in self._pipelines}:
            await self.unload(model_id)


# ------------------------------------------------------------------
# Singleton erişimi
# ------------------------------------------------------------------

_manager: Optional[MediaManager] = None


def get_media_manager() -> MediaManager:
    """Uygulama genelinde tek MediaManager örneğini döner."""
    global _manager
    if _manager is None:
        _manager = MediaManager(
            models_dir=settings.models_dir,
            memory_mode=settings.memory_mode,
            idle_timeout_minutes=settings.idle_timeout_minutes,
        )
    return _manager


def _reset_manager() -> None:
    """Tekil örneği sıfırlar (yalnızca testler için)."""
    global _manager
    _manager = None
