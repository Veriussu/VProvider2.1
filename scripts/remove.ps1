# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Scripti (Windows)
#  Dosya:    scripts/remove.ps1
#  Amaç:     VProvider'ı Windows'tan kaldırır.
#  Mekanik:  - Sunucuyu durdurur, sanal ortamı ve önbellekleri siler
#            - data/ (veritabanı, kullanıcılar) ve runtime loglarını temizler
#            - Varsayılan: models/ ve kaynak kod korunur
#            - -All: modeller + kaynak kod + .env dahil proje dizini silinir
#  Kullanım: .\scripts\remove.ps1        (modeller ve kod korunur)
#            .\scripts\remove.ps1 -All   (projenin tamamı silinir)
# ─────────────────────────────────────────────────────────────
param([switch]$All)

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

if ($All) {
  Write-Host " ⚠  Tüm modeller, kaynak kod, .env ve veriler silinecek." -ForegroundColor Red
  Write-Host "   3 saniye içinde iptal için Ctrl+C."
  Start-Sleep -Seconds 3
}

# 1) Sunucuyu durdur
& (Join-Path $ProjectRoot "scripts\stop.ps1") 2>$null

# 2) Sanal ortam + Python önbellekleri
Remove-Item -Recurse -Force -Path (Join-Path $ProjectRoot ".venv") -ErrorAction SilentlyContinue
Get-ChildItem -Path $ProjectRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Write-Host " ✓ Sanal ortam ve önbellekler silindi" -ForegroundColor Green

# 3) Veri ve loglar
Remove-Item -Recurse -Force -Path (Join-Path $ProjectRoot "data\*") -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force -Path (Join-Path $ProjectRoot "runtime") -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "data") | Out-Null
New-Item -ItemType File -Force -Path (Join-Path $ProjectRoot "data\.gitkeep") | Out-Null
Write-Host " ✓ data/ ve runtime logları temizlendi" -ForegroundColor Green

# 4) Modeller ve proje dizini
if ($All) {
  Remove-Item -Recurse -Force -Path (Join-Path $ProjectRoot "models\*") -ErrorAction SilentlyContinue
  Write-Host " ✓ models/ içeriği silindi" -ForegroundColor Green
} else {
  Write-Host " • models/ içeriği korundu (tam silme için: .\scripts\remove.ps1 -All)" -ForegroundColor Yellow
}

if ($All) {
  Set-Location $env:TEMP
  Remove-Item -Recurse -Force -Path $ProjectRoot -ErrorAction SilentlyContinue
  Write-Host " ✓ Proje dizini silindi: $ProjectRoot" -ForegroundColor Green
} else {
  Write-Host " • Proje kaynak kodu korundu (kaldırmak için: .\scripts\remove.ps1 -All)" -ForegroundColor Yellow
}

Write-Host " ✓ VProvider kaldırma tamamlandı." -ForegroundColor Green