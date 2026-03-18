"""
Agent Orchestrator — Recursive tool-calling loop for Laris.

The agent loop:
  1. Send message + tool descriptions to LLM
  2. If LLM responds with a tool call → execute it → feed result back → goto 1
  3. If LLM responds with plain text → return it (final answer)

Supports both Gemini (native function calling) and Ollama (JSON-in-prompt).
Streams text tokens during the final answer for TTS pipeline compatibility.
"""

import json
import os
import re
import time
from google import genai
from google.genai import errors as genai_errors
import ollama as ollama_client

from agent_tools import (
    TOOLS,
    execute_tool,
    get_tools_description,
    get_tools_json_schema,
    set_last_tool_result,
    set_last_response,
)
from agent_contexts import build_context_prompt, detect_intents

MAX_AGENT_STEPS = 50  # prevent infinite loops


def _build_system_context() -> str:
    """Build dynamic system context with OS info and common paths."""
    import platform
    import subprocess
    home = os.path.expanduser("~")
    username = os.path.basename(home)

    # Detect real Desktop path (may be OneDrive-synced or localized)
    desktop = ""
    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[Environment]::GetFolderPath('Desktop')"],
                capture_output=True, text=True, timeout=5,
            )
            desktop = r.stdout.strip()
        except Exception:
            pass
    if not desktop or not os.path.isdir(desktop):
        # Fallback: try common names
        for name in ("Desktop", "Escritorio"):
            candidate = os.path.join(home, name)
            if os.path.isdir(candidate):
                desktop = candidate
                break
        else:
            desktop = os.path.join(home, "Desktop")

    # Detect Documents path
    documents = ""
    if platform.system() == "Windows":
        try:
            r = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "[Environment]::GetFolderPath('MyDocuments')"],
                capture_output=True, text=True, timeout=5,
            )
            documents = r.stdout.strip()
        except Exception:
            pass
    if not documents or not os.path.isdir(documents):
        documents = os.path.join(home, "Documents")

    downloads = os.path.join(home, "Downloads")

    return (
        f"\nINFORMACIÓN DEL SISTEMA:\n"
        f"- Sistema operativo: {platform.system()} {platform.release()}\n"
        f"- Usuario: {username}\n"
        f"- Home: {home}\n"
        f"- Escritorio (Desktop): {desktop}\n"
        f"- Documentos: {documents}\n"
        f"- Descargas: {downloads}\n"
        f"IMPORTANTE: Usa SIEMPRE rutas absolutas de Windows (con \\\\ o /). "
        f"NUNCA uses rutas Linux como /home/. "
        f"Cuando el usuario diga 'escritorio' o 'desktop', usa: {desktop}\n"
    )


# Strong instruction to force the LLM to use tools instead of hallucinating
# (Now minimal — most rules moved to agent_contexts.py modules)
_AGENT_SYSTEM_SUFFIX = ""


# ─── Gemini Agent (native function calling) ──────────────────────────────────

def agent_stream_gemini(api_key, model_name, chat_history, user_message, system_prompt=""):
    """
    Agent loop with Gemini's native function calling.
    Yields text chunks only for the final response (compatible with TTS pipeline).
    Also yields status updates as special markers: ⟨tool:name⟩ and ⟨result:preview⟩
    """
    client = genai.Client(api_key=api_key)
    context_prompt = build_context_prompt(user_message)
    full_system = system_prompt + "\n" + context_prompt + "\n" + _build_system_context()

    # Build contents from history
    contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append({"role": role, "parts": [{"text": msg["content"]}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    # Build tool declarations for Gemini
    tool_declarations = get_tools_json_schema()
    tools_config = genai.types.Tool(function_declarations=[
        genai.types.FunctionDeclaration(**decl) for decl in tool_declarations
    ])

    config = genai.types.GenerateContentConfig(
        system_instruction=full_system,
        temperature=0.7,
        max_output_tokens=1024,
        tools=[tools_config],
    )

    for step in range(MAX_AGENT_STEPS):
        # Non-streaming call to detect tool calls
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model_name, contents=contents, config=config,
                )
                break
            except genai_errors.ClientError as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(10)
                    continue
                raise

        # Check if the response has a function call
        part = response.candidates[0].content.parts[0]

        if hasattr(part, "function_call") and part.function_call and part.function_call.name:
            fc = part.function_call
            tool_name = fc.name
            tool_args = dict(fc.args) if fc.args else {}

            # Yield status update so UI can show what's happening
            yield f"\n🔧 Ejecutando: {tool_name}({json.dumps(tool_args, ensure_ascii=False)[:100]})\n"

            # Execute the tool
            print(f"  🤖 Agent step {step+1}: {tool_name}({tool_args})")
            result = execute_tool(tool_name, tool_args)
            set_last_tool_result(result)
            print(f"  📋 Result: {result[:200]}...")

            yield f"📋 Resultado: {result[:150]}{'...' if len(result) > 150 else ''}\n"

            # Add the function call and result to contents for next iteration
            contents.append({"role": "model", "parts": [part]})
            contents.append({
                "role": "user",
                "parts": [genai.types.Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )]
            })
            continue  # next iteration — LLM will see the result

        # No function call — this is the final text response. Stream it.
        # Re-do as streaming for TTS compatibility
        config_no_tools = genai.types.GenerateContentConfig(
            system_instruction=full_system,
            temperature=0.7,
            max_output_tokens=1024,
        )
        # Rebuild contents with the full conversation including tool results
        for attempt in range(3):
            try:
                for chunk in client.models.generate_content_stream(
                    model=model_name, contents=contents, config=config_no_tools,
                ):
                    if chunk.text:
                        yield chunk.text
                return
            except genai_errors.ClientError as e:
                if "429" in str(e) and attempt < 2:
                    time.sleep(10)
                    continue
                raise
        return

    # Max steps reached — yield a notice
    yield "\n(Se alcanzó el límite de pasos del agente)"


# ─── Ollama Agent (JSON-in-prompt approach) ──────────────────────────────────

_OLLAMA_TOOL_PROMPT = """
Herramientas disponibles:

{tools_description}

FORMATO para usar herramientas (sin texto antes ni después):
```tool_call
{{"tool": "nombre", "params": {{"param1": "valor1"}}}}
```

Siempre usa EXACTAMENTE este bloque. NUNCA escribas "Ejecutando: herramienta(...)" como texto.
"""

_TOOL_CALL_PATTERN = re.compile(
    r"```tool_call\s*\n?\s*(\{.*?\})\s*\n?\s*```",
    re.DOTALL,
)

# Fallback: detect raw JSON with "tool" key
_TOOL_CALL_RAW_PATTERN = re.compile(
    r'\{\s*"tool"\s*:\s*"(\w+)"\s*,\s*"params"\s*:\s*(\{.*?\})\s*\}',
    re.DOTALL,
)

# Fallback 2: detect function-call style like  open_application({"name": "steam"})
_TOOL_CALL_FUNC_PATTERN = re.compile(
    r'(?:Ejecutando:?\s*)?\b([a-z_]+)\(\s*(\{.*?\})\s*\)',
    re.DOTALL | re.IGNORECASE,
)


def _parse_tool_call(text: str) -> tuple[str, dict] | None:
    """Try to extract a tool call from LLM response text."""
    # Try fenced block first
    m = _TOOL_CALL_PATTERN.search(text)
    if m:
        try:
            data = json.loads(m.group(1))
            if "tool" in data:
                return data["tool"], data.get("params", {})
        except json.JSONDecodeError:
            pass

    # Try raw JSON pattern
    m = _TOOL_CALL_RAW_PATTERN.search(text)
    if m:
        try:
            params = json.loads(m.group(2))
            return m.group(1), params
        except json.JSONDecodeError:
            return m.group(1), {}

    # Try function-call style: tool_name({"param": "value"})
    from agent_tools import _TOOLS_BY_NAME
    m = _TOOL_CALL_FUNC_PATTERN.search(text)
    if m:
        name = m.group(1)
        if name in _TOOLS_BY_NAME:  # only match actual tool names
            try:
                params = json.loads(m.group(2))
                return name, params
            except json.JSONDecodeError:
                return name, {}

    return None


def agent_stream_ollama(model_name, chat_history, user_message, system_prompt=""):
    """
    Agent loop for Ollama using JSON-in-prompt tool calling.
    Yields text chunks for the final response (compatible with TTS pipeline).
    """
    tools_desc = get_tools_description()
    context_prompt = build_context_prompt(user_message)
    full_system = (system_prompt + "\n" + context_prompt + "\n" 
                   + _build_system_context() + "\n"
                   + _OLLAMA_TOOL_PROMPT.format(tools_description=tools_desc))

    messages = [{"role": "system", "content": full_system}]
    for msg in chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_message})

    # Track tool calls so the LLM can see what's done vs pending
    tool_history = []
    intents = detect_intents(user_message)
    save_prompted = False

    for step in range(MAX_AGENT_STEPS):
        # Collect full response first (need to detect tool calls)
        full_response = ""
        try:
            for chunk in ollama_client.chat(model=model_name, messages=messages, stream=True):
                if chunk.message.content:
                    full_response += chunk.message.content
        except Exception as e:
            yield f"Error Ollama: {e}"
            return

        # Check for tool call
        tool_call = _parse_tool_call(full_response)

        if tool_call:
            tool_name, tool_params = tool_call

            # --- Auto-redirect: open_application("youtube") → youtube_search when user wants to search ---
            import re as _re
            if tool_name == "open_application" and tool_params.get("name", "").lower().strip() in ("youtube", "yt"):
                yt_search_match = _re.search(
                    r'\bbusca(?:r|me)?\s+(.+?)(?:\s+y\s+(?:abre|pon|reproduce|ponme|abrelo|ponlo|mete)\b|$)',
                    user_message, _re.IGNORECASE
                )
                if not yt_search_match:
                    # Try broader: anything after "busca" up to end
                    yt_search_match = _re.search(r'\bbusca(?:r|me)?\s+(.+)', user_message, _re.IGNORECASE)
                if yt_search_match:
                    query = yt_search_match.group(1).strip().rstrip('.')
                    print(f"  🔄 Redirect: open_application(youtube) → youtube_search({query})")
                    tool_name = "youtube_search"
                    tool_params = {"query": query}

            yield f"\n🔧 Ejecutando: {tool_name}({json.dumps(tool_params, ensure_ascii=False)[:100]})\n"

            print(f"  🤖 Agent step {step+1}: {tool_name}({tool_params})")
            result = execute_tool(tool_name, tool_params)
            set_last_tool_result(result)
            print(f"  📋 Result: {result[:200]}...")

            yield f"📋 Resultado: {result[:150]}{'...' if len(result) > 150 else ''}\n"

            # Track this step
            param_summary = next(iter(tool_params.values()), "") if tool_params else ""
            tool_history.append(f"{tool_name}({param_summary})")

            # Build feedback message with history so LLM knows what's done
            feedback = (
                f"[RESULTADO '{tool_name}']:\n"
                f"{result}\n\n"
            )
            if len(result) > 500:
                feedback += (
                    "(Resultado largo. Para guardarlo en archivo usa save_last_result.)\n\n"
                )

            # Auto-execute open_url after youtube_search if user wanted to play/open
            if tool_name == "youtube_search" and not result.startswith("Error"):
                import re as _re
                wants_open = _re.search(
                    r'\b(pon|abre|reproduce|reproducir|primer enlace|primer resultado|primer video|ponlo|abrelo)\b',
                    user_message, _re.IGNORECASE
                )
                first_url_m = _re.search(r'(https://www\.youtube\.com/watch\?v=[a-zA-Z0-9_-]+)', result)
                if wants_open and first_url_m:
                    url = first_url_m.group(1)
                    yield f"\n🔧 Ejecutando: open_url({{\"url\": \"{url}\"}})\n"
                    open_result = execute_tool("open_url", {"url": url})
                    tool_history.append(f"open_url({url})")
                    yield f"📋 Resultado: {open_result}\n"
                    # Done — don't let the LLM talk over the video
                    return

            history_str = ", ".join(tool_history)
            feedback += (
                f"Pasos completados: {history_str}\n"
                f"Peticion original: \"{user_message}\"\n"
                f"Revisa: completaste TODOS los pasos que pide el usuario? Si falta algo -> usa otra herramienta. Si ya terminaste TODO -> responde.\n"
                f"NO inventes datos. NO hagas cosas que no te pidieron."
            )

            # Add assistant response and tool result to conversation
            messages.append({"role": "assistant", "content": full_response})
            messages.append({"role": "user", "content": feedback})
            continue  # next iteration

        # No tool call — store response for potential save_last_response
        set_last_response(full_response)

        # Check if user wanted to save content and we haven't saved yet
        if not save_prompted and 'save' in intents:
            has_saved = any('write_file' in h or 'save_last' in h for h in tool_history)
            if not has_saved:
                save_prompted = True
                # Yield the generated text so the user sees it
                yield full_response + "\n\n"
                messages.append({"role": "assistant", "content": full_response})
                feedback = (
                    "Generaste el texto. Ahora GUARDALO en un archivo usando save_last_response.\n"
                    "save_last_response guarda tu respuesta completa. Solo da la ruta del archivo.\n"
                    "Las carpetas se crean automaticamente.\n"
                    f"Peticion original: \"{user_message}\"\n"
                )
                messages.append({"role": "user", "content": feedback})
                continue

        # This is the final answer. Stream it for TTS.
        words = full_response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
        return

    yield "\n(Se alcanzó el límite de pasos del agente)"
