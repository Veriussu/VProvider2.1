# ─────────────────────────────────────────────────────────────
#  Bölüm:    Medya Motoru (in-process diffusers) Testleri
#  Dosya:    tests/test_diffusers_backend.py
#  Amaç:     app/diffusers_backend.py içindeki katalog, tembel yükleme,
#            üretim ve bellek yönetimini sahte pipeline ile doğrular.
#  Mekanik:  - Gerçek diffusers/torch YOK: pipeline_factory testte takılır.
#            - Registry'nin görmesi için geçici models/safetensors/ dizinleri
#              kurulur (model_index.json ile image/video kategorileri).
#            - Sahte pipeline PIL görselleri döndürür; PNG/GIF kodlanması
#              gerçekten çalışır, yalnızca üretim taklit edilir.
# ─────────────────────────────────────────────────────────────

import asyncio
import json

import pytest
from PIL import Image

from app import diffusers_backend as mb


def _pipeline_dir(base, name, class_name="StableDiffusionPipeline"):
    """Registry'nin safetensors pipeline olarak tanıdığı dizin oluşturur."""
    d = base / "safetensors" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "model_index.json").write_text(
        json.dumps({"_class_name": class_name}), "utf-8"
    )
    (d / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    return d


class FakePipeline:
    """__call__ imzası diffusers'a benzeyen sahte pipeline."""

    def __init__(self, task="txt2img", frames=3):
        self.task = task
        self.frames = frames
        self.calls = []
        self.moved_to = None
        self.slicing = False

    def to(self, device):
        self.moved_to = device
        return self

    def enable_vae_slicing(self):
        self.slicing = True

    def __call__(self, prompt=None, negative_prompt=None, num_inference_steps=None,
                 guidance_scale=None, generator=None, num_images_per_prompt=1,
                 width=512, height=512, image=None, num_frames=None, strength=None):
        self.calls.append({
            "prompt": prompt, "negative_prompt": negative_prompt,
            "steps": num_inference_steps, "cfg": guidance_scale,
            "n": num_images_per_prompt, "w": width, "h": height,
            "image": image, "num_frames": num_frames, "strength": strength,
        })

        class Out:
            pass

        out = Out()
        if num_frames:
            out.frames = [
                [Image.new("RGB", (width, height), (i * 40 % 255, 80, 160))
                 for i in range(num_frames)]
            ]
        else:
            out.images = [
                Image.new("RGB", (width, height), (10 * (k + 1), 200, 90))
                for k in range(num_images_per_prompt)
            ]
        return out


def _make_manager(tmp_path, task="txt2img", frames=3, **kwargs):
    """Sahte pipeline fabrikası taşıyan yönetici kurar."""
    created = []

    def factory(info, want_task):
        pipe = FakePipeline(task=want_task, frames=frames)
        pipe.info = info
        created.append(pipe)
        return pipe

    return mb.MediaManager(
        models_dir=tmp_path / "models",
        pipeline_factory=factory,
        **kwargs,
    ), created


# ------------------------------------------------------------------
# Katalog
# ------------------------------------------------------------------

def test_lists_image_and_video_models(tmp_path):
    """Yalnızca image/video kategorili pipeline dizinleri listelenir."""
    models = tmp_path / "models"
    _pipeline_dir(models, "sd-turbo")
    _pipeline_dir(models, "animate-clip", "UNet3DConditionModel")
    (models / "embeddings" / "emb").mkdir(parents=True)
    (models / "embeddings" / "emb" / "config.json").write_text("{}", "utf-8")

    mgr, _ = _make_manager(tmp_path)
    listed = {m.model_id: m.category for m in mgr.list_models()}

    assert listed == {"sd-turbo": "image", "animate-clip": "video"}
    assert mgr.get_model("yok") is None
    assert mgr.ids_for("video") == ["animate-clip"]


def test_parse_size_rounds_to_multiple_of_64():
    """Boyutlar 8/64 katına yuvarlanır; geçersiz girişte varsayılan."""
    assert mb.parse_size("512x512") == (512, 512)
    assert mb.parse_size("512x768") == (512, 768)
    assert mb.parse_size("500x300") == (512, 320)   # yuvarlanır
    assert mb.parse_size("bozuk") == (512, 512)
    assert mb.parse_size("1x1") == (64, 64)          # en az 64


# ------------------------------------------------------------------
# Görsel üretimi
# ------------------------------------------------------------------

def test_generate_images_returns_png(tmp_path):
    """Metin→görsel üretimi PNG baytı döndürür ve pipeline'a doğru argüman geçer."""
    _pipeline_dir(tmp_path / "models", "sd-turbo")
    mgr, created = _make_manager(tmp_path, memory_mode="keep")

    result = asyncio.run(mgr.generate("sd-turbo", "bir kedi", size="512x768",
                                     steps=12, cfg=6.5, n=2, seed=7))

    assert len(result.images) == 2
    assert all(b.startswith(b"\x89PNG") for b in result.images)
    assert result.model_id == "sd-turbo"
    call = created[0].calls[0]
    assert call["prompt"] == "bir kedi"
    assert call["steps"] == 12
    assert call["cfg"] == 6.5
    assert call["n"] == 2
    assert (call["w"], call["h"]) == (512, 768)


def test_generate_image_moves_pipeline_to_device(tmp_path):
    """Pipeline cihaza taşınır ve VAE dilimleme açılır."""
    _pipeline_dir(tmp_path / "models", "sd-turbo")
    mgr, created = _make_manager(tmp_path, memory_mode="keep")
    asyncio.run(mgr.generate("sd-turbo", "x"))
    assert created[0].moved_to is not None
    assert created[0].slicing is True


def test_negative_prompt_passed_when_supported(tmp_path):
    """negative_prompt pipeline imzasında varsa iletilir."""
    _pipeline_dir(tmp_path / "models", "sd-turbo")
    mgr, created = _make_manager(tmp_path, memory_mode="keep")
    asyncio.run(mgr.generate("sd-turbo", "x", negative_prompt="bulanık"))
    assert created[0].calls[0]["negative_prompt"] == "bulanık"


def test_img2img_uses_image_and_strength(tmp_path):
    """Giriş görseli verilince img2img görevi ve strength kullanılır."""
    _pipeline_dir(tmp_path / "models", "sd-turbo")
    mgr, created = _make_manager(tmp_path, memory_mode="keep")
    src = Image.new("RGB", (512, 512))

    asyncio.run(mgr.generate("sd-turbo", "piksel sanatı", image=src))

    assert created[0].calls[0]["image"] is src
    assert created[0].calls[0]["strength"] == 0.75
    assert created[0].task == "img2img"


def test_generate_unknown_model_raises(tmp_path):
    """Kurulu olmayan model MediaError fırlatır."""
    _pipeline_dir(tmp_path / "models", "sd-turbo")
    mgr, _ = _make_manager(tmp_path)
    with pytest.raises(mb.MediaError):
        asyncio.run(mgr.generate("yok", "x"))


def test_generate_error_reports_pipeline_failure(tmp_path):
    """Pipeline patlarsa hata yutulmaz (yoklamak üretilmiş görsel döner)."""

    class BoomPipeline(FakePipeline):
        def __call__(self, **kwargs):
            raise RuntimeError("CUDA out of memory")

    models = tmp_path / "models"
    _pipeline_dir(models, "sd-turbo")
    mgr = mb.MediaManager(
        models_dir=models,
        pipeline_factory=lambda i, t: BoomPipeline(),
        memory_mode="keep",
    )
    with pytest.raises(RuntimeError, match="out of memory"):
        asyncio.run(mgr.generate("sd-turbo", "x"))


# ------------------------------------------------------------------
# Video üretimi
# ------------------------------------------------------------------

def test_generate_video_returns_gif(tmp_path):
    """Video üretimi kare sayısına uygun GIF baytı döndürür."""
    _pipeline_dir(tmp_path / "models", "anim", "UNet3DConditionModel")
    mgr, created = _make_manager(tmp_path, memory_mode="keep")

    result = asyncio.run(mgr.generate("anim", "uçan kuş", frames=4, fps=8))

    assert result.images == []
    assert result.video is not None
    assert result.video["bytes"][:6] in (b"GIF89a", b"GIF87a")
    assert result.video["frames"] == 4
    assert created[0].calls[0]["num_frames"] == 4


def test_video_model_never_uses_image_task(tmp_path):
    """Video modeli txt2img/img2img göreviyle yüklenmez."""
    _pipeline_dir(tmp_path / "models", "anim", "UNet3DConditionModel")
    mgr, created = _make_manager(tmp_path, memory_mode="keep")
    asyncio.run(mgr.generate("anim", "x", frames=2, image=Image.new("RGB", (64, 64))))
    assert created[0].task == "video"


# ------------------------------------------------------------------
# Bellek yönetimi
# ------------------------------------------------------------------

def test_keep_mode_stays_loaded(tmp_path):
    """keep modunda üretim sonrası pipeline bellekte kalır."""
    _pipeline_dir(tmp_path / "models", "sd-turbo")
    mgr, _ = _make_manager(tmp_path, memory_mode="keep")
    asyncio.run(mgr.generate("sd-turbo", "x"))
    assert mgr.get_model("sd-turbo").loaded is True


def test_dynamic_zero_unloads_after_use(tmp_path):
    """dynamic + idle=0 modunda üretim biter bitmez pipeline boşaltılır."""
    _pipeline_dir(tmp_path / "models", "sd-turbo")
    mgr, _ = _make_manager(tmp_path, memory_mode="dynamic", idle_timeout_minutes=0)

    async def run():
        await mgr.generate("sd-turbo", "x")
        await asyncio.sleep(0.05)   # boşaltma görevinin tamamlanması için

    asyncio.run(run())
    assert mgr.get_model("sd-turbo").loaded is False


def test_explicit_load_and_unload(tmp_path):
    """Manuel yükleme/boşaltma uçları doğru çalışır."""
    _pipeline_dir(tmp_path / "models", "sd-turbo")
    mgr, _ = _make_manager(tmp_path, memory_mode="dynamic", idle_timeout_minutes=0)

    assert asyncio.run(mgr.load("sd-turbo")) is True
    assert mgr.get_model("sd-turbo").loaded is True
    assert asyncio.run(mgr.unload("sd-turbo")) is True
    assert mgr.get_model("sd-turbo").loaded is False
    assert asyncio.run(mgr.unload("sd-turbo")) is False   # ikinci çağrı False


def test_load_unknown_model_raises(tmp_path):
    """Olmayan modeli yüklemek MediaError verir."""
    mgr, _ = _make_manager(tmp_path)
    with pytest.raises(mb.MediaError):
        asyncio.run(mgr.load("yok"))


def test_set_memory_mode_validates(tmp_path):
    """Geçersiz bellek modu reddedilir."""
    _pipeline_dir(tmp_path / "models", "sd-turbo")
    mgr, _ = _make_manager(tmp_path)
    with pytest.raises(ValueError):
        mgr.set_memory_mode("sd-turbo", "yanlis")
    with pytest.raises(ValueError):
        mgr.set_global_memory_mode("yanlis")


def test_shutdown_unloads_everything(tmp_path):
    """Kapanışta tüm modeller boşaltılır."""
    _pipeline_dir(tmp_path / "models", "a")
    _pipeline_dir(tmp_path / "models", "b", "UNet3DConditionModel")
    mgr, _ = _make_manager(tmp_path, memory_mode="keep")

    async def run():
        await mgr.generate("a", "x")
        await mgr.generate("b", "x", frames=2)
        await mgr.shutdown()

    asyncio.run(run())
    assert mgr._pipelines == {}


def test_encode_helpers_produce_valid_files(tmp_path):
    """PNG/GIF kodlama yardımcıları geçerli dosya başlıkları üretir."""
    img = Image.new("RGB", (64, 64), (1, 2, 3))
    assert mb.encode_png(img).startswith(b"\x89PNG")
    gif = mb.encode_gif([img, img, img], fps=8)
    assert gif[:6] in (b"GIF89a", b"GIF87a")
    with pytest.raises(mb.MediaError):
        mb.encode_gif([])


# ------------------------------------------------------------------
# Yükleme anahtarları (sürüm/varyant uyumluluğu)
# ------------------------------------------------------------------

def test_load_kwargs_never_forces_safetensors(tmp_path):
    """use_safetensors zorlanmaz; .bin ağırlıklı repolar kırılmaz."""
    d = _pipeline_dir(tmp_path / "models", "bin-model")
    (d / "unet").mkdir()
    (d / "unet" / "diffusion_pytorch_model.bin").write_bytes(b"x")
    mgr, _ = _make_manager(tmp_path)
    kwargs = mgr._load_kwargs(str(d))
    assert "use_safetensors" not in kwargs
    assert "dtype" in kwargs or "torch_dtype" in kwargs


def test_load_kwargs_adds_fp16_variant_when_present(tmp_path):
    """Dizinde .fp16.safetensors varsa variant='fp16' eklenir."""
    d = _pipeline_dir(tmp_path / "models", "fp16-model")
    (d / "unet").mkdir()
    (d / "unet" / "diffusion_pytorch_model.fp16.safetensors").write_bytes(b"x")
    mgr, _ = _make_manager(tmp_path)
    assert mgr._load_kwargs(str(d))["variant"] == "fp16"


def test_load_kwargs_without_variant_when_primary_only(tmp_path):
    """Yalnızca birincil ağırlık varsa variant verilmez."""
    d = _pipeline_dir(tmp_path / "models", "plain-model")
    (d / "unet").mkdir()
    (d / "unet" / "diffusion_pytorch_model.safetensors").write_bytes(b"x")
    mgr, _ = _make_manager(tmp_path)
    assert "variant" not in mgr._load_kwargs(str(d))


def test_has_fp16_variant_helper(tmp_path):
    """Varyant yoksa False, varsa True döner."""
    d = tmp_path / "models" / "a"
    d.mkdir(parents=True)
    assert mb._has_fp16_variant(d) is False
    (d / "unet").mkdir()
    (d / "unet" / "x.fp16.safetensors").write_bytes(b"x")
    assert mb._has_fp16_variant(d) is True


def test_from_pretrained_falls_back_to_torch_dtype(tmp_path):
    """Eski diffusers (dtype'ı bilmeyen) torch_dtype ile yüklenir."""
    _pipeline_dir(tmp_path / "models", "sd")
    mgr, _ = _make_manager(tmp_path)

    seen = []

    class LegacyCls:
        @staticmethod
        def from_pretrained(path, **kwargs):
            seen.append(kwargs)
            if "dtype" in kwargs:
                raise TypeError("unexpected keyword argument 'dtype'")
            return FakePipeline()

    mgr._from_pretrained(LegacyCls, str(tmp_path / "models" / "safetensors" / "sd"))
    assert seen[0] and "torch_dtype" in seen[-1]


def test_prepare_moves_pipeline_and_enables_slicing(tmp_path):
    """_prepare pipeline'ı cihaza taşır ve bellek ayarlarını açar."""
    _pipeline_dir(tmp_path / "models", "sd")
    mgr, _ = _make_manager(tmp_path)
    pipe = mgr._prepare(FakePipeline(), "cuda")
    assert pipe.moved_to == "cuda"
    assert pipe.slicing is True
