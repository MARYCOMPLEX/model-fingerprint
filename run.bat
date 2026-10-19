@echo off
rem Windows launcher. Usage:
rem   run.bat --url https://api.example.com/v1 --key sk-... --model my-model
setlocal
cd /d "%~dp0"
python -m model_fingerprint %*
