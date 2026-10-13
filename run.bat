@echo off
cd /d "%~dp0"
python ollama_client.py
if errorlevel 1 pause
