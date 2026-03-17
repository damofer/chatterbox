@echo off
chcp 65001 >nul
title Instalador - Laris + Chatterbox TTS

echo ============================================
echo   Laris + Chatterbox TTS
echo   Instalador para Windows
echo ============================================
echo.

:: Check Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python no encontrado. Instala Python 3.11 desde python.org
    echo         Asegurate de marcar "Add Python to PATH" durante la instalacion.
    pause
    exit /b 1
)

:: Check Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python encontrado: %PYVER%

:: Check NVIDIA GPU
nvidia-smi >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ADVERTENCIA] nvidia-smi no encontrado. Se necesita GPU NVIDIA con CUDA.
    echo              La app puede funcionar en CPU pero sera muy lenta.
    echo.
)

:: Set install directory
set "INSTALL_DIR=%~dp0"
cd /d "%INSTALL_DIR%"
echo [INFO] Directorio de instalacion: %INSTALL_DIR%

:: Create virtual environment
if not exist ".venv" (
    echo.
    echo [1/4] Creando entorno virtual...
    python -m venv .venv
) else (
    echo [1/4] Entorno virtual ya existe.
)

:: Activate venv
call .venv\Scripts\activate.bat

:: Install PyTorch with CUDA
echo.
echo [2/4] Instalando PyTorch con soporte CUDA...
pip install --upgrade pip >nul 2>&1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

:: Install project and dependencies
echo.
echo [3/4] Instalando Chatterbox TTS y dependencias...
pip install -e . --no-deps
pip install numpy">=1.24.0,<1.26.0" librosa==0.11.0 s3tokenizer transformers==4.46.3
pip install diffusers==0.29.0 resemble-perth==1.0.1 conformer==0.3.2
pip install safetensors==0.5.3 spacy-pkuseg pykakasi==2.3.0
pip install gradio==5.44.1 pyloudnorm omegaconf soundfile
pip install google-genai ollama openai-whisper
pip install chromadb sentence-transformers PyMuPDF

:: Check ffmpeg
where ffmpeg >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ADVERTENCIA] ffmpeg no encontrado. Whisper lo necesita para audio.
    echo              Instala desde: https://www.gyan.dev/ffmpeg/builds/
    echo              O con: winget install Gyan.FFmpeg
)

:: Create launcher
echo.
echo [4/4] Creando acceso directo...
(
    echo @echo off
    echo chcp 65001 ^>nul
    echo title Laris — Asistente de Voz IA
    echo cd /d "%%~dp0"
    echo call .venv\Scripts\activate.bat
    echo python voice_chat_app.py
    echo pause
) > "Iniciar Laris.bat"

:: Create directories
if not exist "voices" mkdir voices
if not exist "knowledge" mkdir knowledge

echo.
echo ============================================
echo   Instalacion completada!
echo ============================================
echo.
echo   Para iniciar la app:
echo     - Doble clic en "Iniciar Laris.bat"
echo     - O ejecuta: python voice_chat_app.py
echo.
echo   Requisitos opcionales:
echo     - Ollama (LLM local): https://ollama.com
echo     - ffmpeg (para audio): winget install Gyan.FFmpeg
echo.
pause
