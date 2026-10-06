# ─────────────────────────────────────────────────────────────
#  Bölüm:    Kurulum Scripti (Windows)
#  Dosya:    install.ps1
#  Amaç:     VProvider'ı Windows'a kurar: donanım tespiti -> venv ->
#            bağımlılıklar -> llama-cpp-python (CUDA derlemesi veya hazır
#            wheel) -> .env üretimi.
#  Platform: Windows PowerShell 5.1+ ve PowerShell 7 (üçü de uyumlu).
#  Kullanım: .\install.ps1            (otomatik tespit)
#            .\install.ps1 -CPU       (her zaman hazır CPU wheel)
#            .\install.ps1 -Rebuild   (donanıma göre zorla derle)
#            .\install.ps1 -SkipBuild (asla derleme; hazır wheel)
#  Not:      CUDA derlemesi için Visual Studio Build Tools + CUDA Toolkit
#            gerekir; yoksa betik otomatik olarak hazır CPU wheel'ine düşer.
# ─────────────────────────────────────────────────────────────
param(
  [switch]$CPU,
  [switch]$Rebuild,
  [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
  Write-Host "✗ python bulunamadı. python.org'dan Python 3.10+ kurun ve 'python' komutunu PATH'e ekleyin." -ForegroundColor Red
  exit 1
}

# ── 1. Sanal ortam ─────────────────────────────────────────────
$venv = Join-Path $ProjectRoot ".venv"
$py = Join-Path $venv "Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Host " • Sanal ortam oluşturuluyor (.venv)..." -ForegroundColor Cyan
  & python -m venv $venv
  if ($LASTEXITCODE -ne 0) { exit 1 }
}
Write-Host " ✓ Sanal ortam hazır" -ForegroundColor Green

& $py -m pip install --disable-pip-version-check --upgrade pip | Out-Null
& $py -m pip install --disable-pip-version-check -r requirements.txt -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { exit 1 }
Write-Host " ✓ Bağımlılıklar kuruldu (requirements*.txt)" -ForegroundColor Green

# ── 2. llama-cpp-python ────────────────────────────────────────
function Test-LlamaInstalled {
  & $py -c "import llama_cpp; print(llama_cpp.__version__)" 2>$null
  return ($LASTEXITCODE -eq 0)
}

# Kurulu derlemenin backend'i (llama_cpp/lib/*.dll dosyalarına bakılır)
function Get-CompiledBackend {
  $lib = Join-Path $venv "Lib\site-packages\llama_cpp\lib"
  if (-not (Test-Path $lib)) { return "cpu" }
  $patterns = @(
    @{ Name = "cuda";   File = "ggml-cuda.dll" },
    @{ Name = "rocm";   File = "ggml-hip.dll" },
    @{ Name = "sycl";   File = "ggml-sycl.dll" },
    @{ Name = "vulkan"; File = "ggml-vulkan.dll" }
  )
  foreach ($p in $patterns) {
    if (Test-Path (Join-Path $lib $p.File)) { return $p.Name }
  }
  return "cpu"
}

function Get-DesiredBackend {
  if ($CPU) { return "cpu" }
  $nv = Get-Command nvidia-smi -ErrorAction SilentlyContinue
  if ($nv) {
    & nvidia-smi -L 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { return "cuda" }
  }
  return "cpu"
}

function Install-LlamaBuild([string]$Flags) {
  $env:CMAKE_ARGS = $Flags
  $env:FORCE_CMAKE = "1"
  & $py -m pip install --no-cache-dir --force-reinstall llama-cpp-python
  Remove-Item Env:\CMAKE_ARGS -ErrorAction SilentlyContinue
  Remove-Item Env:\FORCE_CMAKE -ErrorAction SilentlyContinue
  return ($LASTEXITCODE -eq 0)
}

function Install-LlamaWheel {
  & $py -m pip install --no-cache-dir llama-cpp-python
  return ($LASTEXITCODE -eq 0)
}

$installed = Test-LlamaInstalled
$compiled = Get-CompiledBackend
$desired = Get-DesiredBackend

if ($Rebuild) {
  Write-Host " • llama-cpp-python yeniden derleniyor (-Rebuild)..." -ForegroundColor Cyan
  if (-not (Install-LlamaBuild "-DGGML_$($desired.ToUpper())=on")) {
    if ($desired -eq "cuda") {
      Write-Host " • CUDA derlemesi başarısız (Visual Studio Build Tools + CUDA Toolkit gerekir). Hazır wheel deneniyor..." -ForegroundColor Yellow
      $null = Install-LlamaWheel
    } else {
      $null = Install-LlamaBuild "-DGGML_CPU=on"
    }
  }
  Write-Host " ✓ Motor derlendi: $desired" -ForegroundColor Green
}
elseif ($SkipBuild) {
  Write-Host " • Derleme atlandı (-SkipBuild), hazır wheel kuruluyor..." -ForegroundColor Yellow
  $null = Install-LlamaWheel
  Write-Host " ✓ Motor: hazır wheel" -ForegroundColor Green
}
elseif (-not $installed) {
  Write-Host " • llama-cpp-python kuruluyor..." -ForegroundColor Cyan
  if ($desired -eq "cuda") {
    if (Install-LlamaBuild "-DGGML_CUDA=on") {
      Write-Host " ✓ Motor: NVIDIA CUDA (derlendi)" -ForegroundColor Green
    } else {
      Write-Host " • CUDA derlemesi başarısız, hazır wheel kuruluyor..." -ForegroundColor Yellow
      $null = Install-LlamaWheel
      Write-Host " ✓ Motor: hazır wheel" -ForegroundColor Green
    }
  } else {
    $null = Install-LlamaWheel
    Write-Host " ✓ Motor: hazır wheel (CPU)" -ForegroundColor Green
  }
}
elseif ($desired -ne $compiled -and -not $CPU) {
  Write-Host " • Motor yeniden derlenecek (kurulu: $compiled, hedef: $desired)..." -ForegroundColor Cyan
  if (-not (Install-LlamaBuild "-DGGML_$($desired.ToUpper())=on")) {
    Write-Host " • Derleme başarısız, hazır wheel'a düşülüyor..." -ForegroundColor Yellow
    $null = Install-LlamaWheel
  }
  Write-Host " ✓ Motor güncellendi" -ForegroundColor Green
}
else {
  Write-Host " ✓ llama-cpp-python zaten kurulu ve uyumlu (motor yeniden derlenmedi)" -ForegroundColor Green
}

# ── 3. .env üretimi ────────────────────────────────────────────
$envFile = Join-Path $ProjectRoot ".env"
if (-not (Test-Path $envFile)) {
  Copy-Item (Join-Path $ProjectRoot ".env.example") $envFile
  Write-Host " • .env oluşturuldu (gerekirse değerleri düzenleyin)" -ForegroundColor Green
} else {
  Write-Host " • Mevcut .env korundu" -ForegroundColor Green
}

# ── 4. Özet ───────────────────────────────────────────────────
$port = "9055"
$line = Select-String -Path $envFile -Pattern "^PORT=(.*)$" -ErrorAction SilentlyContinue | Select-Object -Last 1
if ($line) { $port = $line.Matches[0].Groups[1].Value }
$hostName = "127.0.0.1"
$line = Select-String -Path $envFile -Pattern "^HOST=(.*)$" -ErrorAction SilentlyContinue | Select-Object -Last 1
if ($line) { $hostName = $line.Matches[0].Groups[1].Value }

Write-Host ""
Write-Host " ✔ Kurulum tamamlandı!" -ForegroundColor Green
Write-Host "   - Panel:      http://${hostName}:${port}/     (ilk açılışta kurulum sihirbazı)"
Write-Host "   - API:        http://${hostName}:${port}/v1/models   (OpenAI uyumlu)"
Write-Host "   - Başlat:     .\scripts\start.ps1   (veya baslat.bat)"
Write-Host "   - Durdur:     .\scripts\stop.ps1"
Write-Host "   - Test:       .\.venv\Scripts\python.exe -m pytest tests -q"
Write-Host ""
Write-Host " İpucu: Güvenlik Duvarı (Windows Defender Firewall) LAN erişimi için"
Write-Host " uvicorn.exe'ye ağ izni sorarsa 'İzin Ver' deyin."