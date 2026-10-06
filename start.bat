@echo off
rem VProvider Windows başlatma (scripts/start.ps1'i çalıştırır)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\start.ps1" %*
pause