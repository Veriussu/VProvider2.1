# ─────────────────────────────────────────────────────────────
#  Bölüm:    ComfyUI Checkpoint İndirici (Windows)
#  Dosya:    scripts/comfyui-checkpoint.ps1
#  Amaç:     HuggingFace'teki checkpoint dosyasını ComfyUI'nin
#            models/checkpoints/ klasörüne indirir (SD/SDXL/FLUX).
#  Mekanik:  curl.exe ile sürdürülebilir (-C -) indirme yapar.
#  Kullanım: .\scripts\comfyui-checkpoint.ps1 <repo_id> <dosya_yolu> [hedef_adı]
#  Örnek:    .\scripts\comfyui-checkpoint.ps1 stabilityai/sd-1.5 v1-5-pruned-emaonly.safetensors
# ─────────────────────────────────────────────────────────────
param(
  [Parameter(Mandatory = $true)][string]$RepoId,
  [Parameter(Mandatory = $true)][string]$FilePath,
  [string]$TargetName = ""
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Get-EnvValue([string]$Key, [string]$Default) {
  $envFile = Join-Path $ProjectRoot ".env"
  if (Test-Path $envFile) {
    $m = Select-String -Path $envFile -Pattern "^$Key=(.*)$" -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($m) { return $m.Matches[0].Groups[1].Value }
  }
  return $Default
}

$comfyDir = Get-EnvValue "COMFYUI_DIR" (Join-Path $HOME "ComfyUI")
if ([string]::IsNullOrWhiteSpace($TargetName)) {
  $TargetName = Split-Path -Leaf $FilePath
}

$destDir = Join-Path $comfyDir "models\checkpoints"
New-Item -ItemType Directory -Force -Path $destDir | Out-Null
$dest = Join-Path $destDir $TargetName
$url = "https://huggingface.co/${RepoId}/resolve/main/${FilePath}"

$curl = Get-Command curl.exe -ErrorAction SilentlyContinue
if (-not $curl) {
  Write-Host "✗ curl.exe bulunamadı (Windows 10 1803+ ile gelir)." -ForegroundColor Red
  exit 1
}

Write-Host " ⇣ ${RepoId}/${FilePath}"
Write-Host "   → $dest"
& curl.exe -L -C - -o $dest $url
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host " ✓ İndirme tamam: $dest" -ForegroundColor Green
Write-Host "   Panelde Görsel Üretim sekmesinde checkpoint'i yeniden yükleyin."