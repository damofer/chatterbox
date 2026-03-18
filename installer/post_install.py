"""
Post-installation script — runs after Inno Setup extracts files.
Installs Python packages (PyTorch, Chatterbox, etc.) using the bundled Python.
Shows a console window with progress for the user.
"""
import subprocess
import sys
import os

def main():
    app_dir = os.environ.get("LARIS_APP_DIR")
    python_dir = os.environ.get("LARIS_PYTHON_DIR")

    if not app_dir or not python_dir:
        print("ERROR: Variables de entorno no configuradas.")
        input("Presiona Enter para salir...")
        sys.exit(1)

    python_exe = os.path.join(python_dir, "python.exe")
    pip_exe = [python_exe, "-m", "pip"]

    print("=" * 50)
    print("  Laris — Instalando dependencias")
    print("=" * 50)
    print()
    print(f"  Python: {python_exe}")
    print(f"  App:    {app_dir}")
    print()

    def run(desc, cmd):
        print(f"[*] {desc}...")
        result = subprocess.run(cmd, cwd=app_dir)
        if result.returncode != 0:
            print(f"[ERROR] Fallo en: {desc}")
            print(f"        Comando: {' '.join(cmd)}")
            print()
            print("Puedes intentar ejecutar el comando manualmente.")
            input("Presiona Enter para continuar...")

    # Upgrade pip
    run("Actualizando pip",
        pip_exe + ["install", "--upgrade", "pip"])

    # PyTorch con CUDA 12.8
    run("Instalando PyTorch con CUDA (esto puede tardar varios minutos)",
        pip_exe + ["install", "torch", "torchaudio",
                   "--index-url", "https://download.pytorch.org/whl/cu128"])

    # Instalar proyecto en modo editable
    run("Instalando Chatterbox TTS",
        pip_exe + ["install", "-e", ".", "--no-deps"])

    # Dependencias del proyecto
    run("Instalando dependencias principales",
        pip_exe + ["install",
                   "setuptools<81",
                   "numpy>=1.24.0,<1.26.0", "librosa==0.11.0", "s3tokenizer",
                   "transformers==4.46.3", "diffusers==0.29.0",
                   "resemble-perth==1.0.1", "conformer==0.3.2",
                   "safetensors==0.5.3", "spacy-pkuseg", "pykakasi==2.3.0",
                   "gradio==5.44.1", "pyloudnorm", "omegaconf", "soundfile"])

    # Dependencias de la app de voz
    run("Instalando dependencias de voz (Whisper, Ollama, Gemini)",
        pip_exe + ["install",
                   "google-genai", "ollama", "openai-whisper",
                   "transformers", "accelerate",
                   "imageio-ffmpeg"])

    # RAG
    run("Instalando dependencias RAG",
        pip_exe + ["install",
                   "chromadb", "sentence-transformers", "PyMuPDF"])

    # Re-pin setuptools (some deps may have upgraded it past 81)
    run("Fijando setuptools compatible",
        pip_exe + ["install", "setuptools<81"])

    # --- Install Ollama (local LLM backend) ---
    install_ollama = os.environ.get("LARIS_INSTALL_OLLAMA", "0") == "1"
    if install_ollama:
        import shutil
        import urllib.request
        if not shutil.which("ollama"):
            ollama_installer = os.path.join(os.environ.get("TEMP", "."), "OllamaSetup.exe")
            print()
            print("[*] Descargando Ollama (backend LLM local)...")
            try:
                urllib.request.urlretrieve(
                    "https://ollama.com/download/OllamaSetup.exe",
                    ollama_installer,
                )
                print("[*] Instalando Ollama (silencioso)...")
                result = subprocess.run([ollama_installer, "/VERYSILENT", "/NORESTART"])
                if result.returncode == 0:
                    print("    ✅ Ollama instalado correctamente")
                else:
                    print(f"    ⚠️ Ollama instalador retornó código {result.returncode}")
            except Exception as e:
                print(f"    ⚠️ No se pudo instalar Ollama: {e}")
                print("    Puedes instalarlo manualmente desde https://ollama.com")
            finally:
                try:
                    os.remove(ollama_installer)
                except OSError:
                    pass
        else:
            print("[*] Ollama ya está instalado ✅")

        # Pull default model
        import shutil as _sh
        ollama_exe = _sh.which("ollama")
        if ollama_exe:
            print("[*] Descargando modelo Ollama por defecto (llama3.2, ~2 GB)...")
            print("    Esto puede tardar varios minutos...")
            result = subprocess.run([ollama_exe, "pull", "llama3.2"])
            if result.returncode == 0:
                print("    ✅ Modelo llama3.2 descargado")
            else:
                print("    ⚠️ No se pudo descargar el modelo. Puedes hacerlo después con: ollama pull llama3.2")
    else:
        print()
        print("[*] Ollama omitido (no seleccionado). Puedes instalarlo después desde https://ollama.com")

    print()
    print("=" * 50)
    print("  ¡Instalación completada exitosamente!")
    print("=" * 50)
    print()
    print("  Usa el acceso directo 'Laris' para iniciar.")
    print()
    input("Presiona Enter para cerrar...")


if __name__ == "__main__":
    main()
