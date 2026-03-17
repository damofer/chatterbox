"""
Voice Chat App — Hands-free Streaming Conversation
Conversational voice chatbot with continuous listening, VAD, and streaming TTS.

Features:
  - Hands-free conversation mode: mic stays open, auto-detects speech end
  - Streaming LLM text (word by word)
  - Streaming TTS audio (sentence-by-sentence)
  - Voice input via Whisper STT
  - Reference audio caching (processed once, reused)
  - Gemini (cloud) and Ollama (local, free) backends
"""

import json
import os
import re
import shutil
import time
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Ensure ffmpeg is discoverable (PATH may not be refreshed in spawned terminals)
if os.name == "nt" and not shutil.which("ffmpeg"):
    import winreg
    for _root, _subkey in [
        (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
        (winreg.HKEY_CURRENT_USER, r"Environment"),
    ]:
        try:
            with winreg.OpenKey(_root, _subkey) as _key:
                os.environ["PATH"] += ";" + winreg.QueryValueEx(_key, "Path")[0]
        except OSError:
            pass

import torch
import numpy as np
import soundfile as sf
import gradio as gr
from google import genai
from google.genai import errors as genai_errors
import ollama
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES

# RAG dependencies
import glob
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- CUDA performance optimizations ---
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# --- Default reference audio ---
_VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")
_LOCAL_DEFAULT_REF = os.path.join(_VOICES_DIR, "default.wav")
_DEFAULT_ES_REF_URL = "https://storage.googleapis.com/chatterbox-demo-samples/mtl_prompts/es_f1.flac"
_DEFAULT_ES_REF_PATH = os.path.join(tempfile.gettempdir(), "chatterbox_es_latam_ref.flac")

# ───────────────────────────────────────────
# RAG — Knowledge Base
# ───────────────────────────────────────────
_KNOWLEDGE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
_CHROMA_DIR = os.path.join(_KNOWLEDGE_DIR, ".chroma")
os.makedirs(_KNOWLEDGE_DIR, exist_ok=True)

_embed_model: SentenceTransformer | None = None
_chroma_collection = None
_rag_enabled = False


def _get_embed_model():
    """Lazy-load the embedding model (runs on CPU to keep GPU for TTS)."""
    global _embed_model
    if _embed_model is None:
        print("Cargando modelo de embeddings para RAG...")
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("✅ Modelo de embeddings cargado")
    return _embed_model


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path: str) -> str:
    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split text into overlapping chunks by character count."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def index_knowledge_base() -> str:
    """Scan knowledge/ for PDF and TXT files, chunk, embed, and store in ChromaDB."""
    global _chroma_collection, _rag_enabled

    files = glob.glob(os.path.join(_KNOWLEDGE_DIR, "**/*.pdf"), recursive=True) + \
            glob.glob(os.path.join(_KNOWLEDGE_DIR, "**/*.txt"), recursive=True)
    # Exclude README.txt
    files = [f for f in files if os.path.basename(f).lower() != "readme.txt"]

    if not files:
        _rag_enabled = False
        return "⚠️ No se encontraron archivos PDF o TXT en knowledge/"

    embed_model = _get_embed_model()

    # Create/reset ChromaDB
    client = chromadb.PersistentClient(path=_CHROMA_DIR)
    # Delete old collection if exists
    try:
        client.delete_collection("knowledge")
    except Exception:
        pass
    collection = client.create_collection(
        name="knowledge",
        metadata={"hnsw:space": "cosine"},
    )

    all_chunks = []
    all_ids = []
    all_metas = []
    doc_count = 0

    for fpath in files:
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].lower()
        try:
            if ext == ".pdf":
                raw = _read_pdf(fpath)
            else:
                raw = _read_txt(fpath)
        except Exception as e:
            print(f"⚠️ Error leyendo {fname}: {e}")
            continue

        if not raw.strip():
            continue

        chunks = _chunk_text(raw)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{fname}_{i}")
            all_metas.append({"source": fname, "chunk_idx": i})
        doc_count += 1
        print(f"  📄 {fname}: {len(chunks)} fragmentos")

    if not all_chunks:
        _rag_enabled = False
        return "⚠️ Los archivos no contienen texto extraíble."

    # Embed all chunks
    print(f"Generando embeddings para {len(all_chunks)} fragmentos...")
    embeddings = embed_model.encode(all_chunks, show_progress_bar=True, batch_size=64)

    # Add to ChromaDB in batches of 5000 (API limit)
    batch_size = 5000
    for i in range(0, len(all_chunks), batch_size):
        collection.add(
            ids=all_ids[i:i+batch_size],
            documents=all_chunks[i:i+batch_size],
            embeddings=embeddings[i:i+batch_size].tolist(),
            metadatas=all_metas[i:i+batch_size],
        )

    _chroma_collection = collection
    _rag_enabled = True
    msg = f"✅ Indexados {doc_count} documentos → {len(all_chunks)} fragmentos"
    print(msg)
    return msg


def retrieve_context(query: str, n_results: int = 3) -> str:
    """Retrieve relevant document chunks for a user query."""
    if not _rag_enabled or _chroma_collection is None:
        return ""

    embed_model = _get_embed_model()
    query_embedding = embed_model.encode([query]).tolist()

    results = _chroma_collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
    )

    if not results["documents"] or not results["documents"][0]:
        return ""

    context_parts = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        source = meta.get("source", "desconocido")
        context_parts.append(f"[{source}]: {doc}")

    return "\n---\n".join(context_parts)


def _build_rag_system_prompt(user_message: str, custom_prompt: str = "") -> str:
    """Build system prompt with RAG context injected."""
    base_prompt = custom_prompt.strip() if custom_prompt and custom_prompt.strip() else SYSTEM_PROMPT
    context = retrieve_context(user_message)
    if not context:
        return base_prompt
    return (
        base_prompt + "\n\n"
        "A continuación tienes información relevante de la base de conocimiento. "
        "Úsala para responder con precisión. Si la información no es relevante "
        "a la pregunta, ignórala.\n\n"
        f"--- CONTEXTO ---\n{context}\n--- FIN CONTEXTO ---"
    )


def _count_knowledge_files() -> str:
    """Count PDF and TXT files in knowledge/."""
    files = _list_knowledge_files()
    if not files:
        return "No hay documentos en knowledge/"
    names = [os.path.basename(f) for f in files]
    return f"{len(files)} doc(s): {', '.join(names[:10])}{'...' if len(names) > 10 else ''}"


def _list_knowledge_files() -> list[str]:
    """List all PDF and TXT files in knowledge/."""
    files = glob.glob(os.path.join(_KNOWLEDGE_DIR, "**/*.pdf"), recursive=True) + \
            glob.glob(os.path.join(_KNOWLEDGE_DIR, "**/*.txt"), recursive=True)
    return [f for f in files if os.path.basename(f).lower() != "readme.txt"]


def _get_doc_choices():
    """Return gr.update for the document dropdown."""
    choices = [os.path.basename(f) for f in _list_knowledge_files()]
    return gr.update(choices=choices, value=None)


def upload_knowledge_files(file_paths) -> str:
    """Copy uploaded files to knowledge/ folder."""
    if not file_paths:
        return "⚠️ No se seleccionaron archivos."
    added = []
    for fpath in file_paths:
        fname = os.path.basename(fpath)
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".pdf", ".txt"):
            continue
        dest = os.path.join(_KNOWLEDGE_DIR, fname)
        shutil.copy2(fpath, dest)
        added.append(fname)
    if not added:
        return "⚠️ Solo se aceptan archivos .pdf y .txt"
    return f"✅ Subidos: {', '.join(added)}"


def delete_knowledge_file(filename: str) -> str:
    """Delete a document from knowledge/ folder."""
    if not filename:
        return "⚠️ Selecciona un documento para eliminar."
    fpath = os.path.join(_KNOWLEDGE_DIR, os.path.basename(filename))
    if not os.path.isfile(fpath):
        return f"⚠️ No encontrado: {filename}"
    os.remove(fpath)
    return f"🗑️ Eliminado: {filename}"


def _ensure_default_es_ref() -> str:
    """Download the default Spanish reference audio if not already cached."""
    if os.path.exists(_DEFAULT_ES_REF_PATH):
        return _DEFAULT_ES_REF_PATH
    print(f"Descargando voz de referencia español por defecto...")
    urllib.request.urlretrieve(_DEFAULT_ES_REF_URL, _DEFAULT_ES_REF_PATH)
    print(f"Guardada en: {_DEFAULT_ES_REF_PATH}")
    return _DEFAULT_ES_REF_PATH


def _resolve_ref_audio(ref_audio, tts_language):
    """Return the ref audio path: user upload > voices/default.wav > download."""
    if ref_audio:
        return ref_audio
    if os.path.exists(_LOCAL_DEFAULT_REF):
        print(f"Usando voz de referencia local: {_LOCAL_DEFAULT_REF}")
        return _LOCAL_DEFAULT_REF
    if tts_language == "es":
        return _ensure_default_es_ref()
    return None


# --- TTS Model ---
tts_model = None
_cached_ref_key = None
_current_model_type = None  # "english" or "multilingual"


def load_tts(model_type="multilingual"):
    global tts_model, _current_model_type, _cached_ref_key
    if tts_model is None or _current_model_type != model_type:
        print(f"Loading Chatterbox {'Multilingual' if model_type == 'multilingual' else 'English'} TTS on {DEVICE}...")
        if model_type == "multilingual":
            tts_model = ChatterboxMultilingualTTS.from_pretrained(DEVICE)
        else:
            tts_model = ChatterboxTTS.from_pretrained(DEVICE)
        _current_model_type = model_type
        _cached_ref_key = None  # reset cache when model changes

        # --- Optimization: convert to bfloat16 on CUDA ---
        if DEVICE == "cuda" and torch.cuda.is_bf16_supported():
            print("Aplicando optimización bfloat16...")
            tts_model.t3.to(dtype=torch.bfloat16)
            # S3Gen: only convert flow model + HiFiGAN vocoder
            # Keep speaker_encoder and tokenizer in fp32 (FFT doesn't support bf16)
            tts_model.s3gen.flow.to(dtype=torch.bfloat16)
            tts_model.s3gen.mel2wav.to(dtype=torch.bfloat16)
            print("✅ bfloat16 activado para T3 + S3Gen (flow + vocoder)")

        # --- Optimization: torch.compile key modules ---
        # Skipped: mode="reduce-overhead" requires Triton which is not available on Windows.
        # torch.compile with default backend (inductor) also needs Triton on CUDA.
        # If Triton becomes available, uncomment below:
        # try:
        #     tts_model.t3.tfmr = torch.compile(tts_model.t3.tfmr, mode="reduce-overhead")
        #     tts_model.s3gen.mel2wav = torch.compile(tts_model.s3gen.mel2wav, mode="reduce-overhead")
        #     print("✅ torch.compile activado")
        # except Exception as e:
        #     print(f"⚠️ torch.compile no disponible: {e}")

    return tts_model


def prepare_voice_cached(model, ref_audio, exaggeration):
    """Only re-process reference audio if the file or exaggeration changed."""
    global _cached_ref_key
    key = (ref_audio, exaggeration)
    if key != _cached_ref_key:
        print(f"Preparing voice conditionals for: {ref_audio}")
        model.prepare_conditionals(ref_audio, exaggeration=exaggeration)
        _cached_ref_key = key


SYSTEM_PROMPT = (
    "Eres un asistente de voz conversacional amigable. "
    "Responde de forma concisa (1-3 oraciones), natural y conversacional. "
    "Usa español latinoamericano: tuteo (tú/ustedes, NUNCA vosotros ni vos), "
    "sin ceceo (usa 's' en vez de 'z/c' al pronunciar), "
    "vocabulario neutro de Latinoamérica (computadora, celular, carro, platicar, ahorita). "
    "No uses markdown, viñetas, emojis ni formato especial — "
    "tu texto será leído en voz alta por un sistema text-to-speech. "
    "Responde en el mismo idioma que el usuario escribe."
)

# --- Persistent configuration ---
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

_CONFIG_DEFAULTS = {
    "backend": "Ollama",
    "api_key": os.environ.get("GEMINI_API_KEY", ""),
    "gemini_model": "gemini-2.0-flash-lite",
    "ollama_model": "",
    "exaggeration": 0.5,
    "cfg_weight": 0.5,
    "speed_factor": 1.0,
    "cfm_steps": 4,
    "tts_model_type": "multilingual",
    "tts_language": "es",
    "stt_language": "es",
    "system_prompt": SYSTEM_PROMPT,
}


def _load_config() -> dict:
    """Load saved config from disk, falling back to defaults."""
    cfg = dict(_CONFIG_DEFAULTS)
    if os.path.isfile(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg.update({k: v for k, v in saved.items() if k in _CONFIG_DEFAULTS})
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def _save_config(cfg: dict):
    """Persist config to disk."""
    try:
        with open(_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


# Load once at import time for UI defaults
_saved_cfg = _load_config()


# --- Spanish text normalization ---

def normalize_spanish_text(text):
    """Normalize text for better Spanish TTS pronunciation."""
    # Number words for Spanish
    _units = ['', 'uno', 'dos', 'tres', 'cuatro', 'cinco', 'seis', 'siete', 'ocho', 'nueve']
    _teens = ['diez', 'once', 'doce', 'trece', 'catorce', 'quince',
              'dieciséis', 'diecisiete', 'dieciocho', 'diecinueve']
    _tens = ['', 'diez', 'veinte', 'treinta', 'cuarenta', 'cincuenta',
             'sesenta', 'setenta', 'ochenta', 'noventa']

    def _num_to_words(n):
        if n < 0:
            return 'menos ' + _num_to_words(-n)
        if n == 0:
            return 'cero'
        if n < 10:
            return _units[n]
        if n < 20:
            return _teens[n - 10]
        if n < 100:
            t, u = divmod(n, 10)
            if n == 21:
                return 'veintiuno'
            if 21 < n < 30:
                return 'veinti' + _units[u]
            return _tens[t] + (' y ' + _units[u] if u else '')
        if n < 1000:
            c, r = divmod(n, 100)
            if c == 1 and r == 0:
                return 'cien'
            if c == 1:
                return 'ciento ' + _num_to_words(r)
            if c == 5:
                return 'quinientos' + (' ' + _num_to_words(r) if r else '')
            if c == 7:
                return 'setecientos' + (' ' + _num_to_words(r) if r else '')
            if c == 9:
                return 'novecientos' + (' ' + _num_to_words(r) if r else '')
            return _units[c] + 'cientos' + (' ' + _num_to_words(r) if r else '')
        if n < 1000000:
            t, r = divmod(n, 1000)
            prefix = 'mil' if t == 1 else _num_to_words(t) + ' mil'
            return prefix + (' ' + _num_to_words(r) if r else '')
        return str(n)  # fallback for very large numbers

    # Replace standalone numbers with words (up to 999999)
    text = re.sub(r'\b(\d{1,6})\b', lambda m: _num_to_words(int(m.group(1))), text)

    # Common abbreviations (Latin American conventions)
    abbrevs = {
        'Sr.': 'Señor', 'Sra.': 'Señora', 'Srta.': 'Señorita',
        'Dr.': 'Doctor', 'Dra.': 'Doctora',
        'Lic.': 'Licenciado', 'Ing.': 'Ingeniero',
        'Ud.': 'Usted', 'Uds.': 'Ustedes',
        'etc.': 'etcétera', 'Etc.': 'Etcétera',
        'vs.': 'versus', 'aprox.': 'aproximadamente',
        'Col.': 'Colonia', 'Depto.': 'Departamento',
        'Av.': 'Avenida', 'Blvd.': 'Bulevar',
        'No.': 'Número', 'Tel.': 'Teléfono',
    }
    for abbr, full in abbrevs.items():
        text = text.replace(abbr, full)

    # Replace Spain-specific terms with LATAM equivalents (in case LLM slips)
    latam_replacements = {
        'ordenador': 'computadora',
        'móvil': 'celular',
        'coche': 'carro',
        'vale': 'ok',
        'mola': 'está genial',
        'guay': 'genial',
        'tío': 'amigo',
        'habéis': 'han',
        'tenéis': 'tienen',
        'vosotros': 'ustedes',
        'vosotras': 'ustedes',
    }
    for es_spain, es_latam in latam_replacements.items():
        text = re.sub(r'\b' + re.escape(es_spain) + r'\b', es_latam, text, flags=re.IGNORECASE)

    return text


# --- Sentence splitter ---

def split_sentences(text):
    """Split text into sentences for chunk-by-chunk streaming TTS."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


# --- Streaming LLM Backends ---

def _build_gemini_contents(chat_history, user_message):
    contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})
    return contents


def stream_gemini(api_key, model_name, chat_history, user_message, custom_prompt=""):
    """Stream text from Gemini API."""
    client = genai.Client(api_key=api_key)
    contents = _build_gemini_contents(chat_history, user_message)
    config = genai.types.GenerateContentConfig(
        system_instruction=_build_rag_system_prompt(user_message, custom_prompt),
        temperature=0.7,
        max_output_tokens=256,
    )
    for attempt in range(3):
        try:
            for chunk in client.models.generate_content_stream(
                model=model_name, contents=contents, config=config,
            ):
                if chunk.text:
                    yield chunk.text
            return
        except genai_errors.ClientError as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(10)
                continue
            raise gr.Error(
                f"Gemini API error: {e}\n"
                f"Try gemini-2.0-flash-lite or switch to Ollama (local, free)."
            )


def _build_ollama_messages(chat_history, user_message, custom_prompt=""):
    messages = [{"role": "system", "content": _build_rag_system_prompt(user_message, custom_prompt)}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def stream_ollama(model_name, chat_history, user_message, custom_prompt=""):
    """Stream text from local Ollama server."""
    messages = _build_ollama_messages(chat_history, user_message, custom_prompt)
    try:
        for chunk in ollama.chat(model=model_name, messages=messages, stream=True):
            if chunk.message.content:
                yield chunk.message.content
    except Exception as e:
        raise gr.Error(
            f"Ollama error: {e}\n"
            f"Make sure Ollama is running and model is pulled: ollama pull {model_name}"
        )


def get_ollama_models():
    try:
        models = ollama.list()
        return [m.model for m in models.models] if models.models else []
    except Exception:
        return []


# --- Transcription (Speech-to-Text) ---

_whisper_pipe = None

def transcribe(audio_path, language="es"):
    """Transcribe audio using Whisper via transformers pipeline."""
    global _whisper_pipe
    if audio_path is None:
        return ""
    if _whisper_pipe is None:
        from transformers import pipeline
        print("Loading Whisper model for speech-to-text...")
        _whisper_pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-base",
            device=DEVICE,
        )
    generate_kwargs = {"language": language} if language else {}
    result = _whisper_pipe(audio_path, generate_kwargs=generate_kwargs, return_timestamps=True)
    return result["text"].strip()


def transcribe_numpy(audio_np, sr, language="es"):
    """Transcribe audio from numpy array."""
    tmp = tempfile.mktemp(suffix=".wav")
    try:
        sf.write(tmp, audio_np, sr)
        return transcribe(tmp, language=language)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --- VAD (Voice Activity Detection) using Silero VAD ---

_vad_model = None

def _get_vad_model():
    global _vad_model
    if _vad_model is None:
        print("Loading Silero VAD model...")
        _vad_model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad', trust_repo=True)
        _vad_model = _vad_model.to("cpu")  # VAD runs on CPU (tiny model)
    return _vad_model


def _detect_speech(audio_np, sr, energy_gate=None):
    """Return speech probability (0.0 to 1.0) using Silero VAD.
    Includes a minimum energy gate to reject digital silence/noise."""
    import torchaudio
    vad = _get_vad_model()
    if energy_gate is None:
        energy_gate = 0.03

    audio_t = torch.from_numpy(audio_np.astype(np.float32))
    if audio_t.dim() > 1:
        audio_t = audio_t.mean(dim=-1)

    # Normalize int16 to float [-1, 1]
    if audio_t.abs().max() > 1.0:
        audio_t = audio_t / 32768.0

    # Energy gate: if RMS is below noise floor, skip VAD entirely
    rms = torch.sqrt(torch.mean(audio_t ** 2)).item()
    if rms < energy_gate:
        return 0.0

    # Soft noise gate: attenuate samples below threshold to reduce
    # background rumble that might confuse Silero VAD
    gate_threshold = 0.02
    mask = audio_t.abs() < gate_threshold
    audio_t = audio_t.clone()
    audio_t[mask] *= 0.1  # reduce low-amplitude noise by 20dB

    # Resample to 16kHz if needed
    if sr != 16000:
        audio_t = torchaudio.functional.resample(audio_t, sr, 16000)

    # Silero needs at least 512 samples
    if len(audio_t) < 512:
        return 0.0

    # Process in 512-sample windows, return max probability
    max_prob = 0.0
    for i in range(0, len(audio_t) - 511, 512):
        chunk = audio_t[i:i+512]
        prob = vad(chunk, 16000).item()
        max_prob = max(max_prob, prob)
    return max_prob


def _tts_generate(model, sentence, model_type, tts_language, exaggeration, cfg_weight):
    """Generate TTS audio, handling both model types."""
    # Normalize Spanish text if language is Spanish
    if tts_language == "es":
        sentence = normalize_spanish_text(sentence)

    if model_type == "multilingual":
        return model.generate(
            sentence,
            language_id=tts_language,
            audio_prompt_path=None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
    else:
        return model.generate(
            sentence,
            audio_prompt_path=None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )


# --- Pipelined TTS: overlap T3 (token gen) of sentence N+1 with S3Gen (vocoder) of sentence N ---
_tts_lock = threading.Lock()
TTS_WORKERS = 3


def _prepare_tts_input(model, sentence, model_type, tts_language, exaggeration, cfg_weight):
    """CPU work: normalize text + tokenize. Returns args needed for GPU inference."""
    import torch.nn.functional as F

    if tts_language == "es":
        sentence = normalize_spanish_text(sentence)

    if model_type == "multilingual":
        # Replicate what model.generate() does before GPU inference
        from chatterbox.mtl_tts import punc_norm
        text = punc_norm(sentence)
        text_tokens = model.tokenizer.text_to_tokens(
            text, language_id=tts_language.lower() if tts_language else None
        ).to(model.device)
        text_tokens = torch.cat([text_tokens, text_tokens], dim=0)
        sot = model.t3.hp.start_text_token
        eot = model.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)
        return {"text_tokens": text_tokens, "model_type": model_type}
    else:
        # English model — just pass through; can't easily split its pipeline
        return {"sentence": sentence, "model_type": model_type}


def _gpu_generate(model, prepared, exaggeration, cfg_weight, n_cfm_steps=10):
    """GPU work: run T3 inference + S3Gen vocoder. Must hold _tts_lock."""
    from chatterbox.models.s3tokenizer import drop_invalid_tokens

    if prepared["model_type"] == "multilingual":
        text_tokens = prepared["text_tokens"]

        # Update exaggeration if needed
        if float(exaggeration) != float(model.conds.t3.emotion_adv[0, 0, 0].item()):
            from chatterbox.models.t3.modules.cond_enc import T3Cond
            _cond = model.conds.t3
            model.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=model.device)

        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
            speech_tokens = model.t3.inference(
                t3_cond=model.conds.t3,
                text_tokens=text_tokens,
                max_new_tokens=1000,
                temperature=0.8,
                cfg_weight=cfg_weight,
                repetition_penalty=2.0,
                min_p=0.05,
                top_p=1.0,
            )
            speech_tokens = speech_tokens[0]
            speech_tokens = drop_invalid_tokens(speech_tokens)
            speech_tokens = speech_tokens.to(model.device)

            wav, _ = model.s3gen.inference(
                speech_tokens=speech_tokens,
                ref_dict=model.conds.gen,
                n_cfm_timesteps=n_cfm_steps,
            )
            wav = wav.squeeze(0).detach().cpu().numpy()
            watermarked_wav = model.watermarker.apply_watermark(wav, sample_rate=model.sr)
        return torch.from_numpy(watermarked_wav).unsqueeze(0)
    else:
        # English model — use standard generate
        return model.generate(
            prepared["sentence"],
            audio_prompt_path=None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )


def _tts_worker(model, sentence, model_type, tts_language, exaggeration, cfg_weight, speed_factor, sr, n_cfm_steps=10):
    """Thread worker: prepare (CPU, parallel) → generate (GPU, locked) → post-process (CPU, parallel)."""
    # Phase 1: CPU — text normalization + tokenization (runs in parallel)
    prepared = _prepare_tts_input(model, sentence, model_type, tts_language, exaggeration, cfg_weight)

    # Phase 2: GPU — inference (serialized)
    with _tts_lock:
        wav = _gpu_generate(model, prepared, exaggeration, cfg_weight, n_cfm_steps)

    # Phase 3: CPU — numpy conversion + speed adjustment (runs in parallel)
    wav_np = wav.squeeze(0).numpy()
    wav_int16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
    wav_int16, out_sr = _apply_speed(wav_int16, sr, speed_factor)
    return out_sr, wav_int16


def _extract_complete_sentences(pending):
    """Split pending text into (complete_sentences_list, remaining_text)."""
    parts = re.split(r'(?<=[.!?])\s+', pending.strip())
    if len(parts) <= 1:
        return [], pending
    # All but last are complete (they had punctuation + space after them)
    return parts[:-1], parts[-1]


def _apply_speed(wav_np, sr, speed_factor):
    """Change audio speed by resampling."""
    if speed_factor == 1.0:
        return wav_np, sr
    # Resample to change speed without pitch shift would need librosa;
    # simple approach: change sample rate interpretation
    new_sr = int(sr * speed_factor)
    return wav_np, new_sr


# Conversation state dict stored in gr.State
def make_conv_state():
    return {
        "audio_buffer": [],       # list of numpy chunks
        "speech_detected": False, # did we detect speech?
        "silence_chunks": 0,      # consecutive silent chunks since last speech
        "sample_rate": 16000,
        "processing": False,      # True while LLM+TTS are running (ignore mic)
        "prev_audio_len": 0,      # track cumulative audio length
    }

# VAD parameters
SPEECH_PROB_THRESHOLD = 0.75  # Silero probability — very strict, only clear speech
SILENCE_CHUNKS_TO_STOP = 3   # ~1.5s of silence at stream_every=0.5
MIN_SPEECH_CHUNKS = 3         # need at least ~1.5s of speech to process


# --- Hands-free conversation: streaming mic handler (generator) ---

def conversation_stream(
    chunk,          # (sr, audio_np) from streaming mic — cumulative audio
    conv_state,     # gr.State dict
    chat_history,
    backend,
    api_key,
    ref_audio,
    exaggeration,
    cfg_weight,
    gemini_model,
    ollama_model,
    stt_language,
    vad_energy_gate,
    vad_speech_threshold,
    vad_silence_chunks,
    vad_min_speech_chunks,
):
    """
    Called every ~0.5s with the cumulative mic audio.
    Implements VAD — when speech-end detected, runs full pipeline as generator.
    """
    if conv_state is None:
        conv_state = make_conv_state()

    # If we're currently processing a response, ignore mic input
    if conv_state["processing"]:
        yield chat_history, None, conv_state, "⏳ Procesando respuesta..."
        return

    # No audio yet
    if chunk is None:
        yield chat_history, None, conv_state, "⏸️ Esperando audio..."
        return

    sr, audio_full = chunk
    conv_state["sample_rate"] = sr

    current_len = len(audio_full)
    prev_len = conv_state["prev_audio_len"]

    # No new audio
    if current_len <= prev_len:
        yield chat_history, None, conv_state, "⏸️ Sin audio nuevo..."
        return

    # Extract only the new audio since last call
    new_audio = audio_full[prev_len:]
    conv_state["prev_audio_len"] = current_len

    speech_prob = _detect_speech(new_audio, sr, energy_gate=vad_energy_gate)

    # Build real-time VAD status indicator
    rms_val = torch.sqrt(torch.mean(torch.from_numpy(new_audio.astype(np.float32)) ** 2)).item()
    if new_audio.max() > 1.0:  # int16 normalization
        rms_val = rms_val / 32768.0
    status_color = "🟢" if speech_prob >= vad_speech_threshold else "🔴"
    blocked = " ⛔ BLOQUEADO (RMS < gate)" if rms_val < vad_energy_gate else ""
    vad_status = (
        f"{status_color} **RMS**: `{rms_val:.4f}` (gate: {vad_energy_gate:.3f}){blocked} &nbsp;|&nbsp; "
        f"**Prob voz**: `{speech_prob:.2f}` (umbral: {vad_speech_threshold:.2f}) &nbsp;|&nbsp; "
        f"**Hablando**: {'SÍ ✅' if conv_state['speech_detected'] else 'NO'}"
    )

    if speech_prob >= vad_speech_threshold:
        # Speech detected
        conv_state["audio_buffer"].append(new_audio)
        conv_state["speech_detected"] = True
        conv_state["silence_chunks"] = 0
        yield chat_history, None, conv_state, vad_status
        return

    if conv_state["speech_detected"]:
        # We had speech, now silence — count silent chunks
        conv_state["audio_buffer"].append(new_audio)  # keep for context
        conv_state["silence_chunks"] += 1

        if conv_state["silence_chunks"] < int(vad_silence_chunks):
            # Not enough silence yet, keep waiting
            yield chat_history, None, conv_state, vad_status
            return

        # Check minimum speech duration (reject very short accidental triggers)
        speech_chunk_count = len(conv_state["audio_buffer"]) - conv_state["silence_chunks"]
        if speech_chunk_count < int(vad_min_speech_chunks):
            print(f"  (too short: {speech_chunk_count} chunks, ignoring)")
            conv_state["audio_buffer"] = []
            conv_state["speech_detected"] = False
            conv_state["silence_chunks"] = 0
            yield chat_history, None, conv_state, vad_status
            return

        # === SPEECH ENDED — PROCESS ===
        conv_state["processing"] = True
        full_audio = np.concatenate(conv_state["audio_buffer"])

        # Reset buffer immediately
        conv_state["audio_buffer"] = []
        conv_state["speech_detected"] = False
        conv_state["silence_chunks"] = 0

        print(f"🎤 Speech detected ({len(full_audio)/sr:.1f}s), processing...")

        # Transcribe
        user_message = transcribe_numpy(full_audio, sr, language=stt_language)
        if not user_message or len(user_message.strip()) < 2:
            print("  (empty transcription, ignoring)")
            conv_state["processing"] = False
            conv_state["prev_audio_len"] = 0  # reset for next recording cycle
            yield chat_history, None, conv_state, "⚠️ Transcripción vacía, ignorando"
            return

        print(f"  📝 STT: {user_message}")

        # Prepare TTS voice
        model = load_tts()
        if ref_audio:
            prepare_voice_cached(model, ref_audio, exaggeration)
        elif model.conds is None:
            conv_state["processing"] = False
            yield chat_history, None, conv_state, "⚠️ Sube audio de referencia"
            return

        # Get LLM stream
        if backend == "Gemini":
            if not api_key:
                conv_state["processing"] = False
                yield chat_history, None, conv_state, "⚠️ Falta API key"
                return
            llm_stream = stream_gemini(api_key, gemini_model, chat_history, user_message)
        else:
            if not ollama_model:
                conv_state["processing"] = False
                yield chat_history, None, conv_state, "⚠️ Falta modelo Ollama"
                return
            llm_stream = stream_ollama(ollama_model, chat_history, user_message)

        # Phase 1: Stream LLM text
        full_reply = ""
        for text_chunk in llm_stream:
            full_reply += text_chunk
            streaming_h = chat_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": full_reply},
            ]
            yield streaming_h, None, conv_state, "💬 Generando texto..."

        print(f"  💬 LLM: {full_reply[:80]}...")

        final_h = chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": full_reply},
        ]

        # Phase 2: Streaming TTS sentence-by-sentence
        sentences = split_sentences(full_reply) or [full_reply]
        for i, sentence in enumerate(sentences):
            print(f"  🔊 TTS [{i+1}/{len(sentences)}]: {sentence[:60]}...")
            wav = model.generate(
                sentence,
                audio_prompt_path=None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            wav_np = wav.squeeze(0).numpy()
            wav_int16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
            yield final_h, (model.sr, wav_int16), conv_state, f"🔊 TTS [{i+1}/{len(sentences)}]..."

        # Done — ready to listen again
        conv_state["processing"] = False
        conv_state["prev_audio_len"] = 0  # reset for fresh recording
        # Update chat_history state for next round
        yield final_h, None, conv_state, "✅ Listo — escuchando..."
        return

    # No speech detected yet — just waiting
    conv_state["audio_buffer"] = []  # don't accumulate noise
    yield chat_history, None, conv_state, vad_status


# --- Conversation mode: auto-process recorded audio ---

def auto_conversation(
    audio_path,
    chat_history,
    backend,
    api_key,
    ref_audio,
    exaggeration,
    cfg_weight,
    gemini_model,
    ollama_model,
    stt_language,
    tts_model_type,
    tts_language,
    speed_factor,
    cfm_steps,
    custom_prompt="",
):
    """When user stops recording, auto-transcribe → LLM → TTS."""
    if audio_path is None:
        return chat_history, None, None

    # Transcribe
    user_message = transcribe(audio_path, language=stt_language)
    if not user_message or len(user_message.strip()) < 2:
        return chat_history, None, None

    print(f"🎤 Conversación: {user_message}")

    model = load_tts(tts_model_type)

    resolved_ref = _resolve_ref_audio(ref_audio, tts_language)
    if resolved_ref:
        prepare_voice_cached(model, resolved_ref, exaggeration)
    elif model.conds is None:
        raise gr.Error("Sube un audio de referencia primero.")

    if backend == "Gemini" and not api_key:
        raise gr.Error("Ingresa tu Gemini API key.")
    if backend == "Ollama" and not ollama_model:
        raise gr.Error("Selecciona un modelo Ollama (o descarga uno: ollama pull llama3.2)")

    if backend == "Gemini":
        llm_stream = stream_gemini(api_key, gemini_model, chat_history, user_message, custom_prompt)
    else:
        llm_stream = stream_ollama(ollama_model, chat_history, user_message, custom_prompt)

    # Pipeline: LLM streaming + parallel TTS generation
    full_reply = ""
    pending_text = ""
    tts_futures = []
    pool = ThreadPoolExecutor(max_workers=TTS_WORKERS)

    try:
        for text_chunk in llm_stream:
            full_reply += text_chunk
            pending_text += text_chunk
            streaming_h = chat_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": full_reply},
            ]
            yield streaming_h, None, None

            # Submit complete sentences to TTS pool immediately
            complete, pending_text = _extract_complete_sentences(pending_text)
            for sent in complete:
                if sent.strip():
                    print(f"  🔊 TTS submit: {sent[:60]}...")
                    fut = pool.submit(
                        _tts_worker, model, sent,
                        tts_model_type, tts_language,
                        exaggeration, cfg_weight, speed_factor, model.sr, int(cfm_steps),
                    )
                    tts_futures.append(fut)

        # Submit remaining text
        if pending_text.strip():
            print(f"  🔊 TTS submit (final): {pending_text[:60]}...")
            fut = pool.submit(
                _tts_worker, model, pending_text.strip(),
                tts_model_type, tts_language,
                exaggeration, cfg_weight, speed_factor, model.sr, int(cfm_steps),
            )
            tts_futures.append(fut)

        print(f"  💬 LLM: {full_reply[:80]}... | {len(tts_futures)} TTS chunks en pipeline")

        final_h = chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": full_reply},
        ]

        # Yield audio in order as futures complete
        for i, fut in enumerate(tts_futures):
            out_sr, wav_int16 = fut.result()
            print(f"  ✅ TTS [{i+1}/{len(tts_futures)}] listo")
            yield final_h, (out_sr, wav_int16), None
    finally:
        pool.shutdown(wait=False)


# --- Manual chat (text + record button, kept as fallback) ---

def chat_and_speak(
    backend,
    api_key,
    text_message,
    chat_history,
    ref_audio,
    exaggeration,
    cfg_weight,
    gemini_model,
    ollama_model,
    tts_model_type,
    tts_language,
    speed_factor,
    cfm_steps,
    custom_prompt="",
):
    user_message = text_message.strip() if text_message else ""
    if not user_message:
        raise gr.Error("Escribe un mensaje.")

    model = load_tts(tts_model_type)

    resolved_ref = _resolve_ref_audio(ref_audio, tts_language)
    if resolved_ref:
        prepare_voice_cached(model, resolved_ref, exaggeration)
    elif model.conds is None:
        raise gr.Error("Sube un audio de referencia primero.")

    if backend == "Gemini" and not api_key:
        raise gr.Error("Ingresa tu Gemini API key.")
    if backend == "Ollama" and not ollama_model:
        raise gr.Error("Selecciona un modelo Ollama (o descarga uno: ollama pull llama3.2)")

    if backend == "Gemini":
        llm_stream = stream_gemini(api_key, gemini_model, chat_history, user_message, custom_prompt)
    else:
        llm_stream = stream_ollama(ollama_model, chat_history, user_message, custom_prompt)

    full_reply = ""
    pending_text = ""
    tts_futures = []
    pool = ThreadPoolExecutor(max_workers=TTS_WORKERS)

    try:
        for text_chunk in llm_stream:
            full_reply += text_chunk
            pending_text += text_chunk
            streaming_history = chat_history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": full_reply},
            ]
            yield streaming_history, None, ""

            # Submit complete sentences to TTS pool immediately
            complete, pending_text = _extract_complete_sentences(pending_text)
            for sent in complete:
                if sent.strip():
                    fut = pool.submit(
                        _tts_worker, model, sent,
                        tts_model_type, tts_language,
                        exaggeration, cfg_weight, speed_factor, model.sr, int(cfm_steps),
                    )
                    tts_futures.append(fut)

        # Submit remaining text
        if pending_text.strip():
            fut = pool.submit(
                _tts_worker, model, pending_text.strip(),
                tts_model_type, tts_language,
                exaggeration, cfg_weight, speed_factor, model.sr, int(cfm_steps),
            )
            tts_futures.append(fut)

        final_history = chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": full_reply},
        ]

        # Yield audio in order as futures complete
        for fut in tts_futures:
            out_sr, wav_int16 = fut.result()
            yield final_history, (out_sr, wav_int16), ""
    finally:
        pool.shutdown(wait=False)


def refresh_ollama_models():
    models = get_ollama_models()
    if not models:
        return gr.update(choices=[], value=None)
    return gr.update(choices=models, value=models[0])


def toggle_backend(choice):
    is_gemini = choice == "Gemini"
    return (
        gr.update(visible=is_gemini),
        gr.update(visible=is_gemini),
        gr.update(visible=not is_gemini),
        gr.update(visible=not is_gemini),
    )


# --- Gradio UI ---
with gr.Blocks(title="Chat de Voz — IA + Chatterbox") as demo:
   

    # Conversation state (VAD buffer, flags, etc.)
    conv_state = gr.State(make_conv_state)

    with gr.Tabs():
        # ── Tab 1: Chat ──────────────────────────────────────
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(label="Conversación", height=450, type="messages")
            audio_output = gr.Audio(
                label="Respuesta de voz",
                streaming=True,
                autoplay=True,
            )

            gr.Markdown("### 🎙️ Modo Conversación (grabar y procesar)")
            gr.Markdown(
                "Pulsa para grabar, vuelve a pulsar para parar. "
                "Se procesa automáticamente al terminar la grabación."
            )
            conversation_mic = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Grabar mensaje",
            )

            gr.Markdown("---")
            gr.Markdown("### ✏️ Modo Manual (escribir)")
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Escribe algo...",
                    scale=4,
                    show_label=False,
                )
                send_btn = gr.Button("Enviar", variant="primary", scale=1)
            clear_btn = gr.Button("Limpiar conversación")

        # ── Tab 2: Configuración ─────────────────────────────
        with gr.Tab("⚙️ Configuración"):
            with gr.Row():
                save_config_btn = gr.Button("💾 Guardar configuración", variant="primary", size="lg")
                save_config_status = gr.Textbox(show_label=False, interactive=False, scale=2)

            with gr.Group():
                gr.Markdown("#### Backend LLM")
                backend = gr.Radio(
                    choices=["Gemini", "Ollama"],
                    value=_saved_cfg["backend"],
                    label="Backend LLM",
                )
                _is_gemini = _saved_cfg["backend"] == "Gemini"
                api_key = gr.Textbox(
                    label="Gemini API Key",
                    type="password",
                    placeholder="AIza...",
                    value=_saved_cfg["api_key"],
                    visible=_is_gemini,
                )
                gemini_model = gr.Dropdown(
                    choices=[
                        "gemini-2.0-flash",
                        "gemini-2.0-flash-lite",
                        "gemini-1.5-flash",
                        "gemini-1.5-pro",
                    ],
                    value=_saved_cfg["gemini_model"],
                    label="Modelo Gemini",
                    visible=_is_gemini,
                )
                ollama_available = get_ollama_models()
                _saved_ollama = _saved_cfg["ollama_model"]
                _ollama_val = _saved_ollama if _saved_ollama in ollama_available else (
                    ollama_available[0] if ollama_available else None
                )
                ollama_model = gr.Dropdown(
                    choices=ollama_available,
                    value=_ollama_val,
                    label="Modelo Ollama",
                    visible=not _is_gemini,
                )
                refresh_btn = gr.Button("Actualizar modelos Ollama", visible=True, size="sm")

            with gr.Group():
                gr.Markdown(
                    "#### Voz de referencia\n"
                    "Prioridad: **1)** audio subido aquí → **2)** `voices/default.wav` → "
                    "**3)** voz descargada automáticamente.\n\n"
                    "Para usar tu propia voz LATAM por defecto, coloca un `.wav` de ~10s "
                    "en la carpeta `voices/default.wav` del proyecto."
                )
                ref_audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    label="Voz de referencia (WAV/FLAC, ~10s) — opcional para español",
                )
                exaggeration = gr.Slider(0.25, 2, step=0.05, value=_saved_cfg["exaggeration"], label="Exageración")
                cfg_weight = gr.Slider(0.0, 1.0, step=0.05, value=_saved_cfg["cfg_weight"], label="CFG / Ritmo")
                speed_factor = gr.Slider(0.5, 2.0, step=0.05, value=_saved_cfg["speed_factor"], label="Velocidad de voz")
                cfm_steps = gr.Slider(
                    2, 10, step=1, value=_saved_cfg["cfm_steps"],
                    label="Pasos CFM (calidad vs velocidad)",
                    info="Menos pasos = más rápido. 4 = buen balance, 10 = máxima calidad.",
                )

            with gr.Group():
                gr.Markdown("#### Modelo y Idioma TTS")
                tts_model_type = gr.Radio(
                    choices=[
                        ("Multilingüe (español nativo + 22 idiomas)", "multilingual"),
                        ("Inglés (original, solo EN)", "english"),
                    ],
                    value=_saved_cfg["tts_model_type"],
                    label="Modelo TTS",
                )
                tts_language = gr.Dropdown(
                    choices=[
                        ("Español (Latinoamérica)", "es"),
                        ("English", "en"),
                        ("Français", "fr"),
                        ("Deutsch", "de"),
                        ("Italiano", "it"),
                        ("Português", "pt"),
                        ("中文", "zh"),
                        ("日本語", "ja"),
                        ("한국어", "ko"),
                        ("Русский", "ru"),
                        ("العربية", "ar"),
                        ("Hindi", "hi"),
                        ("Türkçe", "tr"),
                        ("Nederlands", "nl"),
                        ("Polski", "pl"),
                        ("Svenska", "sv"),
                        ("Suomi", "fi"),
                        ("Norsk", "no"),
                        ("Dansk", "da"),
                        ("Ελληνικά", "el"),
                        ("עברית", "he"),
                        ("Melayu", "ms"),
                        ("Kiswahili", "sw"),
                    ],
                    value=_saved_cfg["tts_language"],
                    label="Idioma TTS (voz de salida)",
                    info="Idioma en el que el modelo genera la voz. Usa Multilingüe para idiomas != inglés.",
                )

            with gr.Group():
                gr.Markdown("#### Idioma de reconocimiento de voz")
                stt_language = gr.Dropdown(
                    choices=[
                        ("Español (Latinoamérica)", "es"),
                        ("English", "en"),
                        ("Français", "fr"),
                        ("Deutsch", "de"),
                        ("Italiano", "it"),
                        ("Português", "pt"),
                        ("Auto-detectar", ""),
                    ],
                    value=_saved_cfg["stt_language"],
                    label="Idioma de voz (STT — reconocimiento)",
                )

            with gr.Group():
                gr.Markdown("#### 📚 Base de Conocimiento (RAG)")
                gr.Markdown(
                    "Sube archivos **PDF** o **TXT** para que el asistente "
                    "use esa información al responder."
                )
                rag_upload = gr.File(
                    label="Subir documentos (PDF / TXT)",
                    file_types=[".pdf", ".txt"],
                    file_count="multiple",
                    type="filepath",
                )
                rag_upload_result = gr.Textbox(label="Resultado", interactive=False)
                rag_status = gr.Textbox(
                    label="Documentos cargados",
                    value=_count_knowledge_files(),
                    interactive=False,
                )
                with gr.Row():
                    rag_doc_select = gr.Dropdown(
                        choices=[os.path.basename(f) for f in _list_knowledge_files()],
                        label="Seleccionar documento",
                        allow_custom_value=False,
                        scale=3,
                    )
                    rag_delete_btn = gr.Button("🗑️ Eliminar", size="sm", variant="stop", scale=1)
                with gr.Row():
                    rag_index_btn = gr.Button("📚 Indexar documentos", variant="primary", size="sm")
                rag_result = gr.Textbox(label="Resultado de indexación", interactive=False)

            with gr.Group():
                gr.Markdown("#### 🧠 System Prompt")
                system_prompt_input = gr.Textbox(
                    label="Instrucciones del sistema",
                    value=_saved_cfg["system_prompt"],
                    lines=6,
                    placeholder="Ingresa las instrucciones para el asistente...",
                    info="Define la personalidad y comportamiento del asistente. Los cambios se aplican al siguiente mensaje.",
                )

    # --- Event wiring ---

    # Save all config with button
    def _save_all_config(
        _backend, _api_key, _gemini_model, _ollama_model,
        _exaggeration, _cfg_weight, _speed_factor, _cfm_steps,
        _tts_model_type, _tts_language, _stt_language, _system_prompt,
    ):
        cfg = {
            "backend": _backend,
            "api_key": _api_key,
            "gemini_model": _gemini_model,
            "ollama_model": _ollama_model or "",
            "exaggeration": _exaggeration,
            "cfg_weight": _cfg_weight,
            "speed_factor": _speed_factor,
            "cfm_steps": _cfm_steps,
            "tts_model_type": _tts_model_type,
            "tts_language": _tts_language,
            "stt_language": _stt_language,
            "system_prompt": _system_prompt,
        }
        _save_config(cfg)
        return "✅ Configuración guardada"

    save_config_btn.click(
        fn=_save_all_config,
        inputs=[
            backend, api_key, gemini_model, ollama_model,
            exaggeration, cfg_weight, speed_factor, cfm_steps,
            tts_model_type, tts_language, stt_language, system_prompt_input,
        ],
        outputs=[save_config_status],
    )

    # Toggle backend visibility
    backend.change(
        fn=toggle_backend,
        inputs=[backend],
        outputs=[api_key, gemini_model, ollama_model, refresh_btn],
    )

    refresh_btn.click(fn=refresh_ollama_models, outputs=[ollama_model])

    # RAG: upload files → update status → update dropdown
    rag_upload.change(
        fn=upload_knowledge_files,
        inputs=[rag_upload],
        outputs=[rag_upload_result],
    ).then(
        fn=_count_knowledge_files,
        outputs=[rag_status],
    ).then(
        fn=_get_doc_choices,
        outputs=[rag_doc_select],
    )

    # RAG: delete file → update status → update dropdown
    rag_delete_btn.click(
        fn=delete_knowledge_file,
        inputs=[rag_doc_select],
        outputs=[rag_upload_result],
    ).then(
        fn=_count_knowledge_files,
        outputs=[rag_status],
    ).then(
        fn=_get_doc_choices,
        outputs=[rag_doc_select],
    )

    # RAG: indexing
    rag_index_btn.click(
        fn=index_knowledge_base,
        outputs=[rag_result],
    ).then(
        fn=_count_knowledge_files,
        outputs=[rag_status],
    ).then(
        fn=_get_doc_choices,
        outputs=[rag_doc_select],
    )

    # Conversation mode: record → stop → auto-process
    conv_inputs = [
        conversation_mic, chatbot,
        backend, api_key, ref_audio, exaggeration, cfg_weight,
        gemini_model, ollama_model, stt_language,
        tts_model_type, tts_language, speed_factor, cfm_steps,
        system_prompt_input,
    ]
    conversation_mic.stop_recording(
        fn=auto_conversation,
        inputs=conv_inputs,
        outputs=[chatbot, audio_output, conversation_mic],
    )

    # Manual mode
    manual_inputs = [
        backend, api_key, user_input, chatbot,
        ref_audio, exaggeration, cfg_weight, gemini_model, ollama_model,
        tts_model_type, tts_language, speed_factor, cfm_steps,
        system_prompt_input,
    ]
    manual_outputs = [chatbot, audio_output, user_input]

    send_btn.click(fn=chat_and_speak, inputs=manual_inputs, outputs=manual_outputs)
    user_input.submit(fn=chat_and_speak, inputs=manual_inputs, outputs=manual_outputs)
    clear_btn.click(
        lambda: ([], None, ""),
        outputs=[chatbot, audio_output, user_input],
    )

    # --- Restore saved config on every page load/refresh ---
    def _restore_config():
        cfg = _load_config()
        is_gemini = cfg["backend"] == "Gemini"
        ollama_models = get_ollama_models()
        saved_ollama = cfg["ollama_model"]
        ollama_val = saved_ollama if saved_ollama in ollama_models else (
            ollama_models[0] if ollama_models else None
        )
        return (
            gr.update(value=cfg["backend"]),
            gr.update(value=cfg["api_key"], visible=is_gemini),
            gr.update(value=cfg["gemini_model"], visible=is_gemini),
            gr.update(value=ollama_val, choices=ollama_models, visible=not is_gemini),
            gr.update(visible=not is_gemini),  # refresh_btn
            gr.update(value=cfg["exaggeration"]),
            gr.update(value=cfg["cfg_weight"]),
            gr.update(value=cfg["speed_factor"]),
            gr.update(value=cfg["cfm_steps"]),
            gr.update(value=cfg["tts_model_type"]),
            gr.update(value=cfg["tts_language"]),
            gr.update(value=cfg["stt_language"]),
            gr.update(value=cfg["system_prompt"]),
        )

    demo.load(
        fn=_restore_config,
        outputs=[
            backend, api_key, gemini_model, ollama_model, refresh_btn,
            exaggeration, cfg_weight, speed_factor, cfm_steps,
            tts_model_type, tts_language, stt_language, system_prompt_input,
        ],
    )


if __name__ == "__main__":
    load_tts("multilingual")  # Pre-load multilingual model at startup
    # Auto-index knowledge base if documents exist
    _kb_files = glob.glob(os.path.join(_KNOWLEDGE_DIR, "**/*.pdf"), recursive=True) + \
                glob.glob(os.path.join(_KNOWLEDGE_DIR, "**/*.txt"), recursive=True)
    _kb_files = [f for f in _kb_files if os.path.basename(f).lower() != "readme.txt"]
    if _kb_files:
        print(f"Indexando {len(_kb_files)} documentos de knowledge/...")
        index_knowledge_base()
    demo.queue(max_size=20, default_concurrency_limit=1).launch(share=True)
