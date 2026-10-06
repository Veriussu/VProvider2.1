# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Scripti (Windows)
#  Dosya:    scripts/stop.ps1
#  Amaç:     VProvider sunucusunu durdurur.
#  Mekanik:  .env'den okunan portu dinleyen süreci bulur ve sonlandırır.
#  Kullanım: .\scripts\stop.ps1
# ─────────────────────────────────────────────────────────────

$ProjectRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Get-EnvValue([string]$Key, [string]$Default) {
  $envFile = Join-Path $ProjectRoot ".env"
  if (Test-Path $envFile) {
    $m = Select-String -Path $envFile -Pattern "^$Key=(.*)$" -ErrorAction SilentlyContinue | Select-Object -Last 1
    if ($m) { return $m.Matches[0].Groups[1].Value }
  }
  return $Default
}

$port = Get-EnvValue "PORT" "9055"

$listener = Get-NetTCPConnection -State Listen -LocalPort ([int]$port) -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
  Stop-Process -Id $listener.OwningProcess -Force -ErrorAction SilentlyContinue
  Write-Host " ✓ VProvider durduruldu (pid $($listener.OwningProcess))" -ForegroundColor Green
} else {
  Write-Host " ✓ VProvider zaten çalışmıyor" -ForegroundColor Green
}

Remove-Item (Join-Path $ProjectRoot "runtime\vprovider.pid") -ErrorAction SilentlyContinue