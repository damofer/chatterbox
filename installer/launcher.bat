@echo off
chcp 65001 >nul
title Laris — Asistente de Voz IA

:: Use the bundled Python and app directory
set "PYTHON_DIR=%~dp0python"
set "APP_DIR=%~dp0app"
set "PATH=%PYTHON_DIR%;%PYTHON_DIR%\Scripts;%PATH%"

cd /d "%APP_DIR%"
"%PYTHON_DIR%\python.exe" voice_chat_app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] La aplicacion se cerro con errores.
    echo         Verifica que tienes GPU NVIDIA con drivers actualizados.
    echo.
    pause
)
