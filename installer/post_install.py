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
                   "google-genai", "ollama", "openai-whisper"])

    # RAG
    run("Instalando dependencias RAG",
        pip_exe + ["install",
                   "chromadb", "sentence-transformers", "PyMuPDF"])

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
