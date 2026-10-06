@echo off
rem VProvider Windows kurulumu (install.ps1'i çalıştırır)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause