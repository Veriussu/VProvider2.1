# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Scripti (Windows)
#  Dosya:    scripts/restart.ps1
#  Amaç:     VProvider sunucusunu durdurup yeniden başlatır.
#  Kullanım: .\scripts\restart.ps1
# ─────────────────────────────────────────────────────────────

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

& (Join-Path $ProjectRoot "scripts\stop.ps1")
& (Join-Path $ProjectRoot "scripts\start.ps1")