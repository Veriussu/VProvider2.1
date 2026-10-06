# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Scripti (Windows)
#  Dosya:    scripts/download.ps1
#  Amaç:     HuggingFace'ten GGUF modeli indirir. Ad verilmezse repodaki
#            GGUF dosyalarını listeler; ad verilirse o dosyayı models/
#            klasörüne indirir (MODELS_DIR .env'den okunur).
#  Kullanım: .\scripts\download.ps1 org/model
#            .\scripts\download.ps1 org/model dosya.gguf [modeller_klasoru]
# ─────────────────────────────────────────────────────────────
param(
  [Parameter(Mandatory = $true)][string]$Repo,
  [string]$Filename = "",
  [string]$ModelsDir = ""
)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$py = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  Write-Host "✗ Sanal ortam bulunamadı: önce install.ps1 çalıştırılmalı" -ForegroundColor Red
  exit 1
}

if ([string]::IsNullOrWhiteSpace($ModelsDir)) {
  $ModelsDir = "models"
  $envFile = Join-Path $ProjectRoot ".env"
  if (Test-Path $envFile) {
    $m = Select-String -Path $envFile -Pattern "^MODELS_DIR=(.*)$" -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($m) { $ModelsDir = $m.Matches[0].Groups[1].Value }
  }
}

$pyCode = @'
import sys
import time

from app.hf_downloader import download_model, get_repo_files

repo_id, filename, models_dir = sys.argv[1], sys.argv[2], sys.argv[3]


def progress(received, total):
    if total:
        pct = received * 100 / total
        print("\r  %5.1f%%  %8.1f MB / %8.1f MB" %
              (pct, received / 1_048_576, total / 1_048_576), end="", flush=True)


if not filename:
    try:
        files = get_repo_files(repo_id)
    except Exception as exc:
        sys.exit(" ✗ Repo dosyaları alınamadı: %s" % exc)
    if not files:
        sys.exit(" ✗ '%s' içinde GGUF dosyası bulunamadı." % repo_id)
    print(" '%s' reposundaki GGUF dosyaları:" % repo_id)
    for i, f in enumerate(files):
        mark = "  <-- önerilen" if "Q4_K_M" in f.filename else ""
        print("  {0}) {1:60s} {2:8.1f} MB{3}".format(
            i + 1, f.filename, f.size_bytes / 1_048_576, mark))
    print(" Kullanım: .\\scripts\\download.ps1 org/model <dosya_adı>")
    sys.exit(0)

print(" İndiriliyor: {0} / {1}".format(repo_id, filename))
start = time.time()
result = download_model(repo_id, filename, models_dir=models_dir, progress_cb=progress)
elapsed = time.time() - start
print("\n ✓ Tamamlandı: {0}  ({1:.1f} sn)".format(result.path, elapsed))
sys.exit(0 if not getattr(result, "error", None) else 1)
'@

$tmp = Join-Path $env:TEMP "vp_download_$PID.py"
Set-Content -Path $tmp -Value $pyCode -Encoding UTF8
try {
  Push-Location $ProjectRoot
  & $py $tmp $Repo $Filename $ModelsDir
  $code = $LASTEXITCODE
  Pop-Location
  exit $code
} finally {
  Remove-Item $tmp -ErrorAction SilentlyContinue
}