"""
Voice Chat App — Streaming TTS with Gemini / Ollama LLM + Chatterbox TTS
Conversational voice chatbot: speak or type, get streamed voice responses in a cloned voice.

Features:
  - Streaming LLM text (appears word by word in chatbot)
  - Streaming TTS audio (sentence-by-sentence, first audio in ~3-5s)
  - Voice input via Whisper STT
  - Reference audio caching (processed once, reused)
  - Gemini (cloud) and Ollama (local, free) backends
"""

import os
import re
import shutil
import time

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
import gradio as gr
from google import genai
from google.genai import errors as genai_errors
import ollama
from chatterbox.tts import ChatterboxTTS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- TTS Model ---
tts_model = None
_cached_ref_key = None


def load_tts():
    global tts_model
    if tts_model is None:
        print(f"Loading Chatterbox TTS on {DEVICE}...")
        tts_model = ChatterboxTTS.from_pretrained(DEVICE)
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
    "You are a friendly conversational voice assistant. "
    "Keep your responses concise (1-3 sentences), natural, and conversational. "
    "Do not use markdown, bullet points, emojis, or special formatting — "
    "your text will be spoken aloud by a text-to-speech system. "
    "Respond in the same language the user writes in."
)


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


def stream_gemini(api_key, model_name, chat_history, user_message):
    """Stream text from Gemini API."""
    client = genai.Client(api_key=api_key)
    contents = _build_gemini_contents(chat_history, user_message)
    config = genai.types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
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


def _build_ollama_messages(chat_history, user_message):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})
    return messages


def stream_ollama(model_name, chat_history, user_message):
    """Stream text from local Ollama server."""
    messages = _build_ollama_messages(chat_history, user_message)
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
    result = _whisper_pipe(audio_path, generate_kwargs=generate_kwargs)
    return result["text"].strip()


# --- Main Streaming Chat Function (generator) ---

def chat_and_speak(
    backend,
    api_key,
    text_message,
    voice_message,
    chat_history,
    ref_audio,
    exaggeration,
    cfg_weight,
    gemini_model,
    ollama_model,
    stt_language,
):
    # Determine user message: typed text takes priority, otherwise transcribe voice
    user_message = text_message.strip() if text_message and text_message.strip() else ""
    if not user_message and voice_message is not None:
        user_message = transcribe(voice_message, language=stt_language)
    if not user_message:
        raise gr.Error("Please type or speak a message.")

    model = load_tts()

    # Prepare voice conditionals (cached)
    if ref_audio:
        prepare_voice_cached(model, ref_audio, exaggeration)
    elif model.conds is None:
        raise gr.Error("Please upload a reference voice audio first.")

    # Validate backend
    if backend == "Gemini" and not api_key:
        raise gr.Error("Please enter your Gemini API key.")
    if backend == "Ollama" and not ollama_model:
        raise gr.Error("Select an Ollama model (or pull one: ollama pull llama3.2)")

    # --- Phase 1: Stream LLM text into chatbot ---
    if backend == "Gemini":
        llm_stream = stream_gemini(api_key, gemini_model, chat_history, user_message)
    else:
        llm_stream = stream_ollama(ollama_model, chat_history, user_message)

    full_reply = ""
    for text_chunk in llm_stream:
        full_reply += text_chunk
        streaming_history = chat_history + [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": full_reply},
        ]
        # Yield chatbot update, no audio yet
        yield streaming_history, None, "", None

    # --- Phase 2: Streaming TTS sentence-by-sentence ---
    final_history = chat_history + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": full_reply},
    ]

    sentences = split_sentences(full_reply) or [full_reply]
    for i, sentence in enumerate(sentences):
        print(f"  TTS [{i+1}/{len(sentences)}]: {sentence[:60]}...")
        wav = model.generate(
            sentence,
            audio_prompt_path=None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        wav_np = wav.squeeze(0).numpy()
        # Convert to int16 to avoid Gradio float32 warning
        wav_int16 = (np.clip(wav_np, -1.0, 1.0) * 32767).astype(np.int16)
        yield final_history, (model.sr, wav_int16), "", None


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
    gr.Markdown("# Chat de Voz — Streaming IA + Chatterbox TTS")
    gr.Markdown(
        "Habla o escribe → el texto del LLM aparece en tiempo real → el TTS reproduce oración por oración.\n\n"
        "Soporta **Gemini** (nube) y **Ollama** (local, gratis)."
    )

    with gr.Row():
        # Left column — settings
        with gr.Column(scale=1):
            backend = gr.Radio(
                choices=["Gemini", "Ollama"],
                value="Ollama",
                label="Backend LLM",
            )
            api_key = gr.Textbox(
                label="Gemini API Key",
                type="password",
                placeholder="AIza...",
                value=os.environ.get("GEMINI_API_KEY", ""),
                visible=False,
            )
            gemini_model = gr.Dropdown(
                choices=[
                    "gemini-2.0-flash",
                    "gemini-2.0-flash-lite",
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                ],
                value="gemini-2.0-flash-lite",
                label="Modelo Gemini",
                visible=False,
            )
            ollama_available = get_ollama_models()
            ollama_model = gr.Dropdown(
                choices=ollama_available,
                value=ollama_available[0] if ollama_available else None,
                label="Modelo Ollama",
                visible=True,
            )
            refresh_btn = gr.Button("Actualizar modelos Ollama", visible=True, size="sm")

            gr.Markdown("---")
            ref_audio = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Voz de referencia (WAV, ~10s) — se cachea",
            )
            exaggeration = gr.Slider(0.25, 2, step=0.05, value=0.5, label="Exageración")
            cfg_weight = gr.Slider(0.0, 1.0, step=0.05, value=0.5, label="CFG / Ritmo")

            gr.Markdown("---")
            stt_language = gr.Dropdown(
                choices=[
                    ("Español", "es"),
                    ("English", "en"),
                    ("Français", "fr"),
                    ("Deutsch", "de"),
                    ("Italiano", "it"),
                    ("Português", "pt"),
                    ("Auto-detectar", ""),
                ],
                value="es",
                label="Idioma de voz (STT)",
            )

        # Right column — chat
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="Conversación", height=400, type="messages")
            audio_output = gr.Audio(
                label="Respuesta de voz",
                streaming=True,
                autoplay=True,
            )

            gr.Markdown("### Envía un mensaje (escribe o habla)")
            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="Escribe algo...",
                    scale=3,
                    show_label=False,
                )
                voice_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="O habla",
                    scale=2,
                )
                send_btn = gr.Button("Enviar", variant="primary", scale=1)
            clear_btn = gr.Button("Limpiar conversación")

    # Toggle backend visibility
    backend.change(
        fn=toggle_backend,
        inputs=[backend],
        outputs=[api_key, gemini_model, ollama_model, refresh_btn],
    )
    refresh_btn.click(fn=refresh_ollama_models, outputs=[ollama_model])

    # Chat events (generator function → streaming)
    chat_inputs = [
        backend, api_key, user_input, voice_input, chatbot,
        ref_audio, exaggeration, cfg_weight, gemini_model, ollama_model,
        stt_language,
    ]
    chat_outputs = [chatbot, audio_output, user_input, voice_input]

    send_btn.click(fn=chat_and_speak, inputs=chat_inputs, outputs=chat_outputs)
    user_input.submit(fn=chat_and_speak, inputs=chat_inputs, outputs=chat_outputs)
    clear_btn.click(lambda: ([], None, "", None), outputs=chat_outputs)


if __name__ == "__main__":
    load_tts()  # Pre-load model at startup
    demo.queue(max_size=20, default_concurrency_limit=1).launch(share=True)
