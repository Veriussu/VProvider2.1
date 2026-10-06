# ─────────────────────────────────────────────────────────────
#  Bölüm:    ComfyUI Yönetim Scripti (Windows)
#  Dosya:    scripts/comfyui.ps1
#  Amaç:     ComfyUI motorunu Windows'ta başlatır/durdurur/durum gösterir.
#  Mekanik:  ComfyUI'nin kendi .venv'ini (Scripts/python.exe) ve main.py'yi
#            arka planda başlatır; pid/log dosyaları runtime/ altındadır.
#            .env'deki COMFYUI_DIR ve COMFYUI_PORT değerlerini kullanır.
#  Kullanım: .\scripts\comfyui.ps1 {start|stop|status|restart}
# ─────────────────────────────────────────────────────────────
param([ValidateSet("start", "stop", "status", "restart")][string]$Action = "status")

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
$port = Get-EnvValue "COMFYUI_PORT" "8188"
$runtimeDir = Join-Path $ProjectRoot "runtime"
New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null
$pidFile = Join-Path $runtimeDir "comfyui.pid"
$logFile = Join-Path $runtimeDir "comfyui.log"
$errFile = Join-Path $runtimeDir "comfyui.err.log"
$py = Join-Path $comfyDir ".venv\Scripts\python.exe"

function Get-ComfyPid {
  if (Test-Path $pidFile) {
    $id = Get-Content $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($id -and (Get-Process -Id $id -ErrorAction SilentlyContinue)) { return [int]$id }
  }
  $listener = Get-NetTCPConnection -State Listen -LocalPort ([int]$port) -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($listener) { return $listener.OwningProcess }
  return $null
}

switch ($Action) {
  "start" {
    $existing = Get-ComfyPid
    if ($existing) { Write-Host " ✓ ComfyUI zaten çalışıyor (pid $existing)" -ForegroundColor Green; exit 0 }
    if (-not (Test-Path $py) -or -not (Test-Path (Join-Path $comfyDir "main.py"))) {
      Write-Host "✗ ComfyUI bulunamadı: $comfyDir" -ForegroundColor Red
      Write-Host "  Kurulum: deploy/comfyui-rehber.md adım 1-2'yi izleyin." -ForegroundColor Yellow
      exit 1
    }
    $p = Start-Process -FilePath $py `
      -ArgumentList @("main.py", "--listen", "127.0.0.1", "--port", $port) `
      -WorkingDirectory $comfyDir `
      -RedirectStandardOutput $logFile -RedirectStandardError $errFile `
      -WindowStyle Hidden -PassThru
    Start-Sleep -Seconds 3
    if (-not $p.HasExited) {
      Set-Content -Path $pidFile -Value $p.Id
      Write-Host " ✓ ComfyUI başlatıldı: http://127.0.0.1:$port  (pid $($p.Id))" -ForegroundColor Green
    } else {
      Write-Host "✗ ComfyUI başlatılamadı; log:" -ForegroundColor Red
      if (Test-Path $errFile) { Get-Content $errFile -Tail 30 }
      exit 1
    }
  }
  "stop" {
    $existing = Get-ComfyPid
    if ($existing) {
      Stop-Process -Id $existing -Force -ErrorAction SilentlyContinue
      Remove-Item $pidFile -ErrorAction SilentlyContinue
      Write-Host " ✓ ComfyUI durduruldu (pid $existing)" -ForegroundColor Green
    } else {
      Write-Host " ✓ ComfyUI zaten çalışmıyor" -ForegroundColor Green
    }
  }
  "status" {
    $existing = Get-ComfyPid
    if ($existing) { Write-Host " ✓ ComfyUI çalışıyor (pid $existing, port $port)" -ForegroundColor Green }
    else {
      Write-Host " ✗ ComfyUI çalışmıyor (COMFYUI_DIR=$comfyDir, port $port)" -ForegroundColor Yellow
      exit 1
    }
  }
  "restart" {
    & $MyInvocation.MyCommand.Path "stop"
    & $MyInvocation.MyCommand.Path "start"
  }
}