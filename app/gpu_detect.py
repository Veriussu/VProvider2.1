# ─────────────────────────────────────────────────────────────
#  Bölüm:    GPU Algılama ve Motor Seçimi
#  Dosya:    app/gpu_detect.py
#  Amaç:     Donanımı (NVIDIA/AMD/Intel/Apple) gerçek zamanlı algılar, kurulu
#            llama-cpp-python derlemesinin hangi GPU backend'ini içerdiğini
#            belirler ve çalışma zamanında GPU/CPU kararını üretir.
#  Platform: Linux (CUDA/ROCm/SYCL/Vulkan), macOS (Metal + Apple Silicon),
#            Windows (CUDA/Vulkan/SYCL). Backend dosya tespiti platforma göre
#            .so (Linux), .dylib (macOS) veya .dll (Windows) uzantısını kullanır.
#  Mekanik:  - detect_hardware:  nvidia-smi -> rocm-smi -> Apple (macOS) ->
#                                sysfs(amdgpu, Linux) -> lspci (Intel, Linux)
#            - compiled_backends: llama_cpp paketindeki ggml-* kütüphane dosyaları
#            - resolve_runtime: GPU_MODE + GPU_LAYERS + model boyutu ile
#              nihai (backend, gpu_layers, açıklama) kararını üretir.
#            - gpu_mode=auto: donanıma uygun derlenmiş backend varsa GPU,
#              yoksa anlaşılır açıklamayla CPU fallback.
#            - gpu_mode=cuda|rocm|sycl|vulkan|cpu: zorlamalı seçim; kurulu
#              derleme uymazsa Net RuntimeError üretilir (install.sh önerisi).
#  Kullanım: from app.gpu_detect import resolve_runtime
# ─────────────────────────────────────────────────────────────

import logging
import os
import platform
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

logger = logging.getLogger("vprovider")

# llama_cpp paketindeki backend kütüphane dosya adları. İşletim sistemi düzenine
# göre farklılık gösterir: Linux'ta "libggml-cuda.so", macOS'ta "*.dylib",
# Windows'ta "ggml-cuda.dll" (lib öneki yok). Aynı paket farklı sürümlerde
# birden çok dosya taşıyabileceğinden backend başına ad listesi tutulur.
BACKEND_LIB_PATTERNS = {
    "cuda": ("libggml-cuda.so", "libggml-cuda.dylib", "ggml-cuda.dll"),
    "rocm": ("libggml-hip.so", "libggml-rocm.so", "ggml-hip.dll", "ggml-rocm.dll"),
    "sycl": ("libggml-sycl.so", "ggml-sycl.dll"),
    "vulkan": ("libggml-vulkan.so", "libggml-vulkan.dylib", "ggml-vulkan.dll"),
    "metal": ("libggml-metal.dylib", "ggml-metal.dll"),
}

# Donanım satıcısı -> tercih sırasındaki backend adları (fallback: vulkan)
VENDOR_BACKEND_PREFERENCE = {
    "nvidia": ["cuda", "vulkan"],
    "amd": ["rocm", "vulkan"],
    "intel": ["sycl", "vulkan"],
    "apple": ["metal"],
}

# GGUF meta verisinde katman sayısı bilinmiyorsa kullanılan varsayılan
DEFAULT_GGUF_LAYERS = 32

# RAM bütçesi hesabındaki sabitler
RAM_BASELINE_MB = 150      # uygulamanın kendi (modelsiz) bellek ayak izi tahmini
RAM_VRAM_MARGIN_MB = 256   # VRAM tam dolumda bırakılan emniyet payı (RAM'i tutmak için)


def system_ram_mb() -> int:
    """Sistemin toplam fiziksel RAM miktarını döndürür (MB).

    Windows: GlobalMemoryStatusEx; Linux/macOS: sysconf. Bilinemezse 0 döner
    (o durumda RAM bütçesi hesabı devre dışı kalır, eski davranış korunur).
    """
    try:
        if sys.platform.startswith("win"):
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys // (1024 * 1024))
            return 0
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * size // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return 0


def process_rss_mb() -> int:
    """Mevcut sürecin çekirdek (RSS) bellek kullanımı (MB).

    Linux'ta /proc/self/statm üzerinden; diğer platformlarda 0 (bilinmiyor).
    Yalnızca tahmin amaçlıdır; başarısızlıkta 0 döner.
    """
    try:
        if sys.platform.startswith("win"):
            return 0
        with open("/proc/self/statm") as fh:
            parts = fh.read().split()
        page = os.sysconf("SC_PAGE_SIZE")
        return int(int(parts[1]) * page // (1024 * 1024))
    except Exception:  # noqa: BLE001
        return 0


@dataclass
class GPUInfo:
    """Algılanan donanımın özeti."""

    vendor: str          # "nvidia" | "amd" | "intel" | "none"
    name: str            # ekran kartı model adı (bilinmiyorsa boş)
    vram_total_mb: int   # toplam VRAM (bilinmiyorsa/intel ise 0)
    vram_free_mb: int    # boş VRAM (0 = bilinmiyor / paylaşımlı bellek)
    driver: str          # sürücü (bilinmiyorsa boş)


@dataclass
class RuntimeInfo:
    """Çalışma zamanı için üretilen (backend, gpu_layers) kararı."""

    hardware: GPUInfo
    backend: str                 # "cuda" | "rocm" | "sycl" | "vulkan" | "metal" | "cpu"
    compiled_backends: list      # kurulu llama_cpp derlemesindeki backend'ler
    gpu_layers: int              # LlamaCppEngine'e gidecek nihai değer
    note: str                    # loglara ve panele giden açıklama


# ─────────────────────────────────────────────
# Donanım algılama
# ─────────────────────────────────────────────

def _run(cmd: list, timeout: float = 5.0) -> Optional[str]:
    """Kısa bir alt süreç komutu çalıştırır; başarısız olursa None döner."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


def _nvidia_info() -> Optional[GPUInfo]:
    """nvidia-smi üzerinden NVIDIA kart bilgisini toplar."""
    out = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    if not out:
        return None
    line = out.strip().splitlines()[0]
    parts = line.rsplit(",", 3)  # kart adında virgül olabilir; sondan ayır
    if len(parts) != 4:
        return None
    name, driver, total_s, free_s = [p.strip() for p in parts]
    try:
        total_mb = int(float(total_s))
        free_mb = int(float(free_s))
    except ValueError:
        return None
    return GPUInfo(vendor="nvidia", name=name, vram_total_mb=total_mb, vram_free_mb=free_mb, driver=driver)


def _amd_info() -> Optional[GPUInfo]:
    """AMD GPU bilgisini rocm-smi üzerinden toplar (varsa)."""
    out = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram"])
    if not out:
        return None
    name = ""
    total_mb = 0
    used_mb = 0
    processed_total = False
    for raw in out.splitlines():
        line = raw.strip()
        m = re.match(r"GPU\[0\]:\s*(.+)", line)
        if m:
            name = m.group(1).strip()
        m = re.match(r"vram\s+used\s+\(MB\):\s*(\d+)", line)
        if m:
            used_mb = int(m.group(1))
        m = re.match(r"vram\s+total\s+\(MB\):\s*(\d+)", line)
        if m:
            total_mb = int(m.group(1))
            processed_total = True
    if not processed_total:
        return None
    return GPUInfo(
        vendor="amd",
        name=name or "AMD GPU",
        vram_total_mb=total_mb,
        vram_free_mb=max(0, total_mb - used_mb),
        driver="rocm",
    )


def _amd_sysfs_info() -> Optional[GPUInfo]:
    """sysfs üzerinden AMD (amdgpu sürücüsü) kartını bulur (rocm-smi yoksa)."""
    for card in sorted(Path("/sys/class/drm").glob("card*")):
        try:
            driver_link = (card / "device" / "driver").resolve()
            if driver_link.name != "amdgpu":
                continue
            total_mb = int((card / "device" / "mem_info_vram_total").read_text()) // (1024 * 1024)
            used_mb = int((card / "device" / "mem_info_vram_used").read_text()) // (1024 * 1024)
        except (FileNotFoundError, ValueError, OSError):
            continue
        return GPUInfo(
            vendor="amd",
            name=card.name,
            vram_total_mb=total_mb,
            vram_free_mb=max(0, total_mb - used_mb),
            driver="amdgpu",
        )
    return None


def _intel_info() -> Optional[GPUInfo]:
    """lspci üzerinden Intel GPU'sunu bulur (bellek paylaşımlı, VRAM bilinmez).

    Yalnızca Linux'ta anlamlıdır; Windows/macOS'ta "lspci" olmadığından _run
    None döner ve bu algılayıcı zararsızca atlanır.
    """
    out = _run(["lspci"])
    if not out:
        return None
    if not re.search(r"VGA compatible controller:\s*.*Intel|3D controller:\s*.*Intel", out, re.IGNORECASE):
        return None
    return GPUInfo(vendor="intel", name="Intel GPU", vram_total_mb=0, vram_free_mb=0, driver="i915")


def _apple_info() -> Optional[GPUInfo]:
    """macOS'ta Apple Silicon/Intel Mac'i algılar.

    llama.cpp Metal backend'i birleşik (unified) belleği kullanır; VRAM ayrı
    bir sayaç olmadığından 0 (bilinmiyor) döner. Bu, auto_gpu_layers'ın
    tüm katmanları GPU'ya almasına (-1) yol açar ve Metal için doğrudur
    (model + KV aynı sistem belleğinde yaşar, disk/VRAM taşması olmaz).
    """
    if sys.platform != "darwin":
        return None
    machine = platform.machine().lower()
    name = "Apple Silicon" if machine in ("arm64", "aarch64") else "Intel Mac"
    return GPUInfo(vendor="apple", name=name, vram_total_mb=0, vram_free_mb=0, driver="metal")


@lru_cache(maxsize=1)
def detect_hardware() -> GPUInfo:
    """Sistemdeki GPU'yu satıcı önceliğiyle algılar (NVIDIA -> AMD -> Apple).

    Apple algılaması yalnızca macOS'ta; sysfs/lspci algılayıcıları yalnızca
    Linux'ta çalışır (diğer platformlarda _run None döner ve atlanır).
    Hiçbir GPU algılanamazsa vendor="none" döner. Süreç başına bir kez
    çalışır (lru_cache); testlerde monkeypatch ile değiştirilebilir.
    """
    for detector in (_nvidia_info, _amd_info, _apple_info, _amd_sysfs_info, _intel_info):
        info = detector()
        if info is not None:
            logger.info("Donanım tespiti: %s (%s)", info.vendor, info.name or "-")
            return info
    logger.info("Donanım tespiti: GPU yok, CPU kullanılacak")
    return GPUInfo(vendor="none", name="", vram_total_mb=0, vram_free_mb=0, driver="")


# ─────────────────────────────────────────────
# Kurulu llama_cpp derlemesinin backend'leri
# ─────────────────────────────────────────────

@lru_cache(maxsize=1)
def compiled_backends() -> list:
    """Kurulu llama-cpp-python derlemesinin içerdiği backend listesini döner.

    Paketin lib/ klasöründeki backend kütüphane dosyalarına bakar. Dosya adı
    platforma göre değişir: Linux "libggml-cuda.so", macOS "libggml-cuda.dylib",
    Windows "ggml-cuda.dll". Metal yalnızca macOS'ta anlamlıdır (ggml.cpp'de
    GGML_METAL yolu Apple Silicon'a özgüdür). Hiçbir GPU backend dosyası yoksa
    ["cpu"] döner. import yerine dosya varlığına bakılır; böylece ağır
    kütüphane yüklenmez.
    """
    try:
        import llama_cpp
    except ImportError:
        return ["cpu"]
    lib_dir = Path(llama_cpp.__file__).parent / "lib"
    found = []
    for backend, names in BACKEND_LIB_PATTERNS.items():
        if backend == "metal" and not sys.platform.startswith("darwin"):
            continue
        if any((lib_dir / name).exists() for name in names):
            found.append(backend)
    if not found:
        return ["cpu"]
    logger.info("Derlenmiş llama_cpp backend'leri: %s", ", ".join(found))
    return found


# ─────────────────────────────────────────────
# GGUF meta verisinden katman sayısı (en iyi çaba)
# ─────────────────────────────────────────────

_VT_U8, _VT_I8, _VT_U16, _VT_I16, _VT_U32, _VT_I32 = 0, 1, 2, 3, 4, 5
_VT_F32, _VT_BOOL, _VT_STRING, _VT_ARRAY, _VT_U64, _VT_I64, _VT_F64 = 6, 7, 8, 9, 10, 11, 12


class _GGUFReadError(Exception):
    """GGUF meta verisi okunamadı (ör. bozuk/geçersiz dosya)."""

MAX_ARRAY_SKIP = 256  # array eleman sayısı bu sayıyı aşarsa parse durdurulur


def _read_verify(f, size: int) -> bytes:
    """f.read(size) yapar; saçma boyutları ve eksik veriyi ayıklar."""
    if size < 0 or size > 1 << 30:  # 1 GiB üstü okuyalım diyen veri geçersiz
        raise _GGUFReadError(f"geçersiz boyut: {size}")
    data = f.read(size)
    if len(data) != size:
        raise _GGUFReadError("dosya sonu (EOF)")
    return data


def _skip_value(f, value_type: int) -> None:
    """Bir GGUF metadata değerini türüne göre atlar (tensör verisine gitmeden)."""
    if value_type in (_VT_U8, _VT_I8, _VT_BOOL):
        size = 1
    elif value_type in (_VT_U16, _VT_I16):
        size = 2
    elif value_type in (_VT_U32, _VT_I32, _VT_F32):
        size = 4
    elif value_type in (_VT_U64, _VT_I64, _VT_F64):
        size = 8
    elif value_type == _VT_STRING:
        ln = struct.unpack("<Q", _read_verify(f, 8))[0]
        _read_verify(f, ln)
        return
    elif value_type == _VT_ARRAY:
        # GGUF array düzeni: element türü (u32) SONRA sayı (u64)
        elem_type = struct.unpack("<I", _read_verify(f, 4))[0]
        count = struct.unpack("<Q", _read_verify(f, 8))[0]
        if count > MAX_ARRAY_SKIP:
            raise _GGUFReadError(f"array çok büyük ({count} eleman)")
        for _ in range(count):
            _skip_value(f, elem_type)
        return
    else:
        raise _GGUFReadError(f"bilinmeyen değer türü: {value_type}")
    _read_verify(f, size)


def _read_value(f, value_type: int):
    """Bir GGUF metadata değerini okur; desteklenmeyen türde None döner."""
    if value_type in (_VT_U8, _VT_I8):
        return struct.unpack("<b", _read_verify(f, 1))[0] if value_type == _VT_I8 else _read_verify(f, 1)[0]
    if value_type == _VT_BOOL:
        return _read_verify(f, 1)[0] != 0
    if value_type in (_VT_U16, _VT_I16):
        return struct.unpack("<h", _read_verify(f, 2))[0] if value_type == _VT_I16 else struct.unpack("<H", _read_verify(f, 2))[0]
    if value_type == _VT_U32:
        return struct.unpack("<I", _read_verify(f, 4))[0]
    if value_type == _VT_I32:
        return struct.unpack("<i", _read_verify(f, 4))[0]
    if value_type == _VT_F32:
        return struct.unpack("<f", _read_verify(f, 4))[0]
    if value_type == _VT_U64:
        return struct.unpack("<Q", _read_verify(f, 8))[0]
    if value_type == _VT_I64:
        return struct.unpack("<q", _read_verify(f, 8))[0]
    if value_type == _VT_F64:
        return struct.unpack("<d", _read_verify(f, 8))[0]
    if value_type == _VT_STRING:
        ln = struct.unpack("<Q", _read_verify(f, 8))[0]
        return _read_verify(f, ln).decode("utf-8", errors="replace")
    return None  # array ve bilinmeyen türler -> sonraki anahtarlar atlanır


def gguf_meta(path, max_kv: int = 4096) -> dict:
    """GGUF dosyasının metadata başlığındaki anahtar-değer sözlüğünü okur.

    Yalnızca başlık bölümü okunur (tensör verisine dokunulmaz), bu yüzden
    büyük modellerde tüm dosya belleğe alınmaz. Okunamazsa kısmi sonuç
    (okunabilen anahtarlar) veya boş sözlük döner.
    """
    result: dict = {}
    try:
        with open(path, "rb") as f:
            magic, _version = struct.unpack("<II", _read_verify(f, 8))
            if magic != 0x46554747:  # "GGUF"
                return {}
            tensors = struct.unpack("<Q", _read_verify(f, 8))[0]  # tensör sayısı
            result["_tensor_count"] = tensors
            kv_count = struct.unpack("<Q", _read_verify(f, 8))[0]
            for _ in range(min(kv_count, max_kv)):
                kl = struct.unpack("<Q", _read_verify(f, 8))[0]
                key = _read_verify(f, kl).decode("utf-8", errors="replace")
                vt = struct.unpack("<I", _read_verify(f, 4))[0]
                value = _read_value(f, vt)
                if value is None:  # desteklenmeyen tür (ör. array) -> atla
                    _skip_value(f, vt)
                    continue
                result[key] = value
    except (OSError, struct.error, UnicodeDecodeError):
        return {}
    except _GGUFReadError:
        # Kısıtlı array veya eksik veri: şimdiye kadar okunanları koru
        pass
    return result


def gguf_n_layers(model_path) -> Optional[int]:
    """Modelin başlığındaki katman sayısını (ör. llama.block_count) okur."""
    meta = gguf_meta(model_path)
    if not meta:
        return None
    arch = meta.get("general.architecture")
    for key in (f"{arch}.block_count", "llama.block_count", "bert.block_count"):
        value = meta.get(key)
        if isinstance(value, int) and value > 0:
            return value
    return None


def gguf_kv_per_token_layer(model_path) -> Optional[float]:
    """Bir katmanın KV önbelleğinin token başına bayt giderini tahmin eder.

    kV hesabı: n_kv_heads x (key_length + value_length) bayt; varsayılan KV
    türü Q8_0 (1 bayt) kabul edilir. GGUF'ta doğrudan anahtarlar yoksa
    embedding_length / head_count üzerinden türetilir. Bilinemiyorsa None.
    """
    meta = gguf_meta(model_path)
    if not meta:
        return None
    arch = meta.get("general.architecture")
    suffix_sets = (
        (".attention.head_count_kv", ".attention.key_length", ".attention.value_length"),
        (".attention.head_count_kv", ".attention.key_length", ".attention.value_length"),
        (".attention.head_count_kv", ".attention.key_length", ".attention.value_length"),
    )

    def _find(*suffixes):
        for k in suffixes:
            for prefix in (arch, "llama", "bert"):
                value = meta.get(f"{prefix}{k}")
                if isinstance(value, int) and value > 0:
                    return value
        return None

    n_kv = _find(".attention.head_count_kv")
    k_len = _find(".attention.key_length")
    v_len = _find(".attention.value_length")
    if (not k_len or not v_len) and n_kv:
        emb = _find(".embedding_length")
        heads = _find(".attention.head_count")
        if emb and heads:
            per_head = max(1, emb // heads)
            k_len = k_len or per_head
            v_len = v_len or per_head
    if not n_kv or not k_len or not v_len:
        return None
    # KV başına bayt (Q8_0 varsayılanı); F16 KV için 2 kullanılmalıdır
    kv_bytes_per_elem = 1.0
    return kv_bytes_per_elem * n_kv * (k_len + v_len)


# ─────────────────────────────────────────────
# Katman sayısı otomatik hesabı (VRAM'e göre)
# ─────────────────────────────────────────────

def auto_gpu_layers(
    vram_free_mb: int,
    model_size_mb: int,
    n_layers: Optional[int] = None,
    kv_per_token_layer: Optional[float] = None,
    n_ctx: int = 0,
    ram_budget_mb: int = 0,
) -> int:
    """Boş VRAM'e göre güvenli gpu_layers değerini döndürür.

    Katman başına maliyet iki bölümden oluşur:
      - ağırlıklar   : model boyutu / katman sayısı
      - KV önbelleği : token başına katman KV gideri x context (n_ctx)
    KV bilgisi GGUF'tan (kv_per_token_layer) gelir; bilinmiyorsa genel tahmin
    (KV ~ model boyutu x n_ctx/65536) kullanılır. Compute/UB tamponu için
    sabit bir pay (max ~512 MB veya %12) ayrılır.

    ram_budget_mb > 0 ise sistem RAM bütçesi de gözetilir: GPU'ya taşınmayan
    katmanların (ağırlık + KV) bellek gideri bu bütçeyi aşmamalıdır. Aşılma
    ihtimalinde, VRAM'in yalnızca emniyet payı kadarını harcayarak daha çok
    katman GPU'ya taşınır (RAM'i bütçede tutmak için). VRAM imkân vermezse
    en iyi çaba uygulanır ve resolve_runtime bunu uyarıya çevirir.
    """
    if vram_free_mb <= 0 or model_size_mb <= 0:
        return -1
    total_layers = n_layers if n_layers and n_layers > 0 else DEFAULT_GGUF_LAYERS
    weights_per_layer_mb = model_size_mb / total_layers
    if kv_per_token_layer and n_ctx > 0:
        kv_per_layer_mb = kv_per_token_layer * n_ctx / 1_000_000
    else:
        # Genel tahmin: 65536 bağlamda KV yaklaşık model boyutu ölçeğindedir
        kv_per_layer_mb = weights_per_layer_mb * 1.25 * (n_ctx / 65536.0)
    per_layer_mb = weights_per_layer_mb + kv_per_layer_mb
    if per_layer_mb <= 0:
        return -1
    # Grafik/UB tamponları için kullanılabilir VRAM'in bir kısmı ayrılır.
    # MoE/modern modellerde compute grafikleri beklenenden büyüktür.
    graph_reserve_mb = max(1024, int(total_layers * 30), int(vram_free_mb * 0.10))
    # Tüm katmanlar sığabilir mi? Fazladan emniyet payı dikkate alınarak.
    total_weights_mb = total_layers * weights_per_layer_mb
    total_kv_mb = total_layers * kv_per_layer_mb
    total_needed_mb = total_weights_mb + total_kv_mb + graph_reserve_mb
    if total_needed_mb <= vram_free_mb:
        return -1  # tüm katmanlar VRAM'e sığıyor
    available_mb = vram_free_mb - graph_reserve_mb
    if available_mb <= 0:
        count = 0
    else:
        count = int(available_mb / per_layer_mb)

    # RAM bütçesi: CPU'da kalan katmanlar (ağırlık + KV) sistem RAM'inde yaşar.
    # Amaç: RAM %MEMORY_LIMIT_PCT'nin altında kalsın. Gerekirse GPU'ya daha çok
    # katman taşı — ama asla VRAM emniyet payının (RAM_VRAM_MARGIN_MB) altına inme.
    if ram_budget_mb > 0:
        baseline = max(process_rss_mb(), RAM_BASELINE_MB)
        room_mb = ram_budget_mb - baseline
        if room_mb > 0:
            cpu_by_ram = int(room_mb / per_layer_mb)  # RAM'de tutulabilecek katman
            n_ram = max(0, total_layers - cpu_by_ram)  # en az bu kadarını GPU'ya al
            if n_ram > count:
                # Grafik rezervinin tamamı yerine yalnızca küçük bir emniyet payı
                # bırakarak, RAM'i bütçede tutabilecek kadar katman daha taşınır.
                hard_available_mb = vram_free_mb - RAM_VRAM_MARGIN_MB
                n_vram_hard = int(hard_available_mb / per_layer_mb) if hard_available_mb > 0 else 0
                count = min(n_ram, n_vram_hard)
                if count >= total_layers:
                    return -1  # bütçe için tüm katmanlar GPU'ya taşınabilir
                if count <= 0 and n_ram > 0:
                    count = 0  # VRAM gerçekten yetmiyor; en iyi çaba eski kararda
        # RAM tarafı ek offload'a izin vermeyecek kadar darsa int(count) VRAM değeridir.

    # Anlamsız küçük offload'dan kaçın: katmanların ~%10'undan azı
    # sığacaksa CPU'da kal (gpu_layers=0)
    min_sensible = max(2, int(total_layers * 0.10))
    if count < min_sensible:
        return 0
    return count


# ─────────────────────────────────────────────
# Çalışma zamanı kararı (backend + gpu_layers)
# ─────────────────────────────────────────────

def resolve_runtime(
    gpu_mode: str = "auto",
    requested_layers: int = -1,
    model_path: Optional[str] = None,
    model_size_bytes: int = 0,
    n_ctx: int = 0,
    ram_budget_mb: int = 0,
) -> RuntimeInfo:
    """GPU_MODE/GPU_LAYERS ayarlarından nihai çalışma zamanı kararını üretir.

    gpu_mode değerleri:
      auto   -> donanıma göre; uyumlu derlenmiş backend varsa GPU, yoksa CPU.
      cuda | rocm | sycl | vulkan | metal -> zorlamalı; derlenmemişse RuntimeError.
      cpu    -> her zaman CPU (gpu_layers=0).
    Ayrıca requested_layers=0 ise CPU'ya, pozitif ise olduğu gibi GPU'ya gider.

    ram_budget_mb > 0 ise otomatik katman hesabı RAM bütçesini de gözetir ve
    tahmini RAM aşımında karara uyarı eklenir (yükleme engellenmez).
    """
    hw = detect_hardware()
    compiled = compiled_backends()

    if gpu_mode == "cpu" or requested_layers == 0:
        note = "CPU modu (GPU_LAYERS=0 veya GPU_MODE=cpu)"
        note += _ram_warning(model_path, model_size_bytes, n_ctx, ram_budget_mb) if ram_budget_mb > 0 else ""
        return RuntimeInfo(
            hardware=hw,
            backend="cpu",
            compiled_backends=compiled,
            gpu_layers=0,
            note=note,
        )

    if gpu_mode == "auto":
        preference = VENDOR_BACKEND_PREFERENCE.get(hw.vendor, [])
        chosen = next((b for b in preference if b in compiled), None)
        if chosen is None:
            note = (
                f"GPU algılandı ({hw.vendor}: {hw.name or 'bilinmiyor'}) ancak kurulu "
                f"llama_cpp derlemesi uyumlu değil (derlenen: {', '.join(compiled)}). "
                "CPU moduna geçiliyor. GPU için: bash install.sh --rebuild"
            )
            if ram_budget_mb > 0:
                note += _ram_warning(model_path, model_size_bytes, n_ctx, ram_budget_mb)
            return RuntimeInfo(hardware=hw, backend="cpu", compiled_backends=compiled, gpu_layers=0, note=note)
        backend = chosen
    elif gpu_mode in ("cuda", "rocm", "sycl", "vulkan", "metal"):
        if gpu_mode not in compiled:
            raise RuntimeError(
                f"GPU_MODE={gpu_mode} seçildi ancak kurulu llama_cpp derlemesi "
                f"{gpu_mode} içermiyor (derlenen: {', '.join(compiled)}). "
                "Çözüm: 'bash install.sh --rebuild' ile doğru backend derlenebilir "
                "(Windows'ta install.ps1 -Rebuild)."
            )
        backend = gpu_mode
    else:
        raise ValueError(f"Geçersiz GPU_MODE: {gpu_mode!r} (auto|cuda|rocm|sycl|vulkan|metal|cpu)")

    layers = requested_layers
    if layers == -1:
        n_layers = gguf_n_layers(model_path) if model_path else None
        kv_per_token_layer = gguf_kv_per_token_layer(model_path) if model_path else None
        layers = auto_gpu_layers(
            hw.vram_free_mb,
            model_size_bytes / 1_000_000,
            n_layers,
            kv_per_token_layer,
            n_ctx,
            ram_budget_mb=ram_budget_mb,
        )

    note = (
        f"{hw.vendor.upper()} GPU algılandı ({hw.name or 'bilinmiyor'}; boş VRAM "
        f"{hw.vram_free_mb} MB) -> backend={backend}, gpu_layers={layers}"
    )
    if ram_budget_mb > 0:
        note += _ram_warning(model_path, model_size_bytes, n_ctx, ram_budget_mb, layers=layers)
    return RuntimeInfo(hardware=hw, backend=backend, compiled_backends=compiled, gpu_layers=layers, note=note)


# Tahminî RAM kullanımını (uygulama + CPU'da kalan ağırlık/KV) bütçeyle karşılar.
def _ram_warning(
    model_path: Optional[str],
    model_size_bytes: int,
    n_ctx: int,
    ram_budget_mb: int,
    layers: int = 0,
) -> str:
    """RAM bütçesi aşılacaksa uyarı metni; aşılmıyorsa boş dize döner."""
    try:
        if not model_path or model_size_bytes <= 0:
            return ""
        total = gguf_n_layers(model_path) or DEFAULT_GGUF_LAYERS
        kv = gguf_kv_per_token_layer(model_path)
        model_mb = model_size_bytes / 1_000_000
        wpl = model_mb / total if total else model_mb
        kv_layer_mb = 0.0
        if kv and n_ctx > 0:
            kv_layer_mb = kv * n_ctx / 1_000_000
        elif wpl > 0:
            kv_layer_mb = wpl * 1.25 * (n_ctx / 65536.0)
        baseline = max(process_rss_mb(), RAM_BASELINE_MB)
        cpu_layers = max(0, total - max(0, layers))
        est = baseline + cpu_layers * (wpl + kv_layer_mb)
        if est <= ram_budget_mb:
            return ""
        return (
            f" | RAM bütçesi uyarısı: tahmini sistem RAM kullanımı ~{int(est)} MB "
            f"(bütçe {ram_budget_mb} MB). Model RAM'e zor sığabilir — daha küçük bir "
            "quant veya daha kısa CONTEXT_SIZE önerilir."
        )
    except Exception:  # noqa: BLE001
        return ""