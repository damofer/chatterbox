@echo off
chcp 65001 >nul
title Laris — Asistente de Voz IA
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python voice_chat_app.py
pause
