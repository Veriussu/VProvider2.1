# ─────────────────────────────────────────────────────────────
#  Bölüm:    Yönetim Scripti (Windows)
#  Dosya:    scripts/start.ps1
#  Amaç:     VProvider sunucusunu Windows'ta başlatır.
#  Mekanik:  .env'den HOST/PORT okur, port zaten dinleniyorsa çıkış yapar;
#            değilse venv'deki uvicorn.exe'yi arka planda başlatır
#            (log: runtime/vprovider.log).
#  Kullanım: .\scripts\start.ps1
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

$listenHost = Get-EnvValue "HOST" "0.0.0.0"
$port = Get-EnvValue "PORT" "9055"

$uvicorn = Join-Path $ProjectRoot ".venv\Scripts\uvicorn.exe"
if (-not (Test-Path $uvicorn)) {
  Write-Host "✗ Sanal ortam bulunamadı: önce install.ps1 çalıştırılmalı" -ForegroundColor Red
  exit 1
}

$logDir = Join-Path $ProjectRoot "runtime"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$logFile = Join-Path $logDir "vprovider.log"
$errFile = Join-Path $logDir "vprovider.err.log"
$pidFile = Join-Path $logDir "vprovider.pid"

# Port dinleniyor mu? (uvicorn'un Python'u gibi süreç isimleri değil, port kesin)
$listener = Get-NetTCPConnection -State Listen -LocalPort ([int]$port) -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
  Write-Host " ✓ VProvider zaten çalışıyor (pid $($listener.OwningProcess))" -ForegroundColor Green
  Set-Content -Path $pidFile -Value $listener.OwningProcess
  exit 0
}

$p = Start-Process -FilePath $uvicorn `
  -ArgumentList @("app.main:app", "--host", $listenHost, "--port", $port) `
  -WorkingDirectory $ProjectRoot `
  -RedirectStandardOutput $logFile -RedirectStandardError $errFile `
  -WindowStyle Hidden -PassThru

Start-Sleep -Seconds 3
if ($p -and -not $p.HasExited) {
  Set-Content -Path $pidFile -Value $p.Id
  Write-Host " ✓ VProvider başlatıldı: http://${listenHost}:${port}  (pid $($p.Id))" -ForegroundColor Green
  Write-Host "   Log: $logFile"
} else {
  Write-Host "✗ Sunucu başlatılamadı; log dosyasına bakın:" -ForegroundColor Red
  if (Test-Path $errFile) { Get-Content $errFile -Tail 40 }
  exit 1
}