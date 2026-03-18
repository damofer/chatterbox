"""
Agent Tools — Extensible tool system for Laris agent mode.

Each tool is a dict with:
  - name: unique identifier
  - description: what the tool does (sent to LLM)
  - parameters: JSON-schema-style description of parameters
  - function: callable(params_dict) -> str
"""

import datetime
import json
import os
import shutil
import subprocess
import re
import urllib.request
import urllib.parse
import urllib.error


# ─── Safety ──────────────────────────────────────────────────────────────────
# Allowed directories for file operations (expanded at runtime)
_WORKSPACE_DIR = os.path.dirname(os.path.abspath(__file__))
_ALLOWED_ROOTS = [_WORKSPACE_DIR, os.path.expanduser("~")]

# Blocked commands that could cause damage
_BLOCKED_CMD_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bformat\b",
    r"\bdel\s+/s\s+/q\s+[A-Z]:\\",
    r"\brmdir\s+/s\s+/q\s+[A-Z]:\\",
    r"\b(Invoke-Expression|iex)\b.*\bdownload",
]


def _is_path_safe(path: str) -> bool:
    """Check that a path is within allowed directories."""
    resolved = os.path.realpath(os.path.expanduser(path))
    return any(resolved.startswith(os.path.realpath(root)) for root in _ALLOWED_ROOTS)


def _is_command_safe(cmd: str) -> bool:
    """Block obviously dangerous commands."""
    for pattern in _BLOCKED_CMD_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return False
    return True


# ─── Tool implementations ────────────────────────────────────────────────────

def tool_run_command(params: dict) -> str:
    """Execute a shell command and return stdout+stderr."""
    cmd = params.get("command", "").strip()
    if not cmd:
        return "Error: no se proporcionó un comando."
    if not _is_command_safe(cmd):
        return "Error: comando bloqueado por seguridad."

    timeout = min(params.get("timeout", 30), 60)  # cap at 60s
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_WORKSPACE_DIR,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += ("\n--- stderr ---\n" + result.stderr) if output else result.stderr
        if result.returncode != 0:
            output += f"\n(exit code: {result.returncode})"
        return output.strip()[:4000] or "(sin salida)"
    except subprocess.TimeoutExpired:
        return f"Error: el comando excedió el tiempo límite de {timeout}s."
    except Exception as e:
        return f"Error ejecutando comando: {e}"


def tool_read_file(params: dict) -> str:
    """Read a file's contents."""
    path = params.get("path", "").strip()
    if not path:
        return "Error: no se proporcionó una ruta."    # Strip trailing semicolons/garbage the LLM sometimes appends
    path = path.rstrip(";,.")    # Expand ~ to user home directory first
    path = os.path.expanduser(path)
    # Resolve relative to workspace
    if not os.path.isabs(path):
        path = os.path.join(_WORKSPACE_DIR, path)
    if not _is_path_safe(path):
        return "Error: ruta fuera del directorio permitido."
    if not os.path.isfile(path):
        return f"Error: archivo no encontrado: {path}"
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(10000)  # cap at 10KB
        if len(content) == 10000:
            content += "\n... (truncado a 10KB)"
        return content
    except Exception as e:
        return f"Error leyendo archivo: {e}"


def tool_write_file(params: dict) -> str:
    """Write content to a file."""
    path = params.get("path", "").strip()
    content = params.get("content", "")
    if not path:
        return "Error: no se proporcionó una ruta."
    # Expand ~ to user home directory first
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(_WORKSPACE_DIR, path)
    if not _is_path_safe(path):
        return "Error: ruta fuera del directorio permitido."
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Archivo escrito: {path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error escribiendo archivo: {e}"


def tool_move_file(params: dict) -> str:
    """Move or rename a file or folder."""
    source = params.get("source", "").strip()
    destination = params.get("destination", "").strip()
    if not source or not destination:
        return "Error: se necesitan 'source' y 'destination'."
    source = os.path.expanduser(source)
    destination = os.path.expanduser(destination)
    if not os.path.isabs(source):
        source = os.path.join(_WORKSPACE_DIR, source)
    if not os.path.isabs(destination):
        destination = os.path.join(_WORKSPACE_DIR, destination)
    if not _is_path_safe(source) or not _is_path_safe(destination):
        return "Error: ruta fuera del directorio permitido."
    if not os.path.exists(source):
        return f"Error: no existe: {source}"
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.move(source, destination)
        return f"Movido: {source} -> {destination}"
    except Exception as e:
        return f"Error moviendo: {e}"


def tool_copy_file(params: dict) -> str:
    """Copy a file or folder."""
    source = params.get("source", "").strip()
    destination = params.get("destination", "").strip()
    if not source or not destination:
        return "Error: se necesitan 'source' y 'destination'."
    source = os.path.expanduser(source)
    destination = os.path.expanduser(destination)
    if not os.path.isabs(source):
        source = os.path.join(_WORKSPACE_DIR, source)
    if not os.path.isabs(destination):
        destination = os.path.join(_WORKSPACE_DIR, destination)
    if not _is_path_safe(source) or not _is_path_safe(destination):
        return "Error: ruta fuera del directorio permitido."
    if not os.path.exists(source):
        return f"Error: no existe: {source}"
    try:
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        if os.path.isdir(source):
            shutil.copytree(source, destination)
        else:
            shutil.copy2(source, destination)
        return f"Copiado: {source} -> {destination}"
    except Exception as e:
        return f"Error copiando: {e}"


def tool_list_directory(params: dict) -> str:
    """List contents of a directory."""
    path = params.get("path", ".").strip()
    # Expand ~ to user home directory first
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(_WORKSPACE_DIR, path)
    if not _is_path_safe(path):
        return "Error: ruta fuera del directorio permitido."
    if not os.path.isdir(path):
        # Fuzzy match: try to find a similar folder name in the parent
        parent = os.path.dirname(path)
        target_name = os.path.basename(path).lower()
        # Strip accents for comparison
        import unicodedata
        def _normalize(s):
            return "".join(
                c for c in unicodedata.normalize("NFD", s.lower())
                if unicodedata.category(c) != "Mn"
            )
        target_norm = _normalize(target_name)
        target_words = set(target_norm.split())
        if os.path.isdir(parent):
            candidates = []
            for entry in os.listdir(parent):
                full = os.path.join(parent, entry)
                if not os.path.isdir(full):
                    continue
                entry_norm = _normalize(entry)
                entry_words = set(entry_norm.split())
                # Match if: substring match, or significant word overlap
                overlap = target_words & entry_words
                if (target_norm in entry_norm or entry_norm in target_norm
                        or len(overlap) >= max(1, len(target_words) - 1)):
                    candidates.append(entry)
            if len(candidates) == 1:
                path = os.path.join(parent, candidates[0])
            elif candidates:
                return (f"Error: directorio no encontrado: {os.path.basename(params.get('path', ''))}\n"
                        f"Carpetas similares en {parent}:\n" + "\n".join(f"  • {c}" for c in candidates[:10]))
            else:
                return f"Error: directorio no encontrado: {path}"
        else:
            return f"Error: directorio no encontrado: {path}"
    try:
        entries = sorted(os.listdir(path))
        result = []
        for e in entries[:100]:
            full = os.path.join(path, e)
            if os.path.isdir(full):
                result.append(f"[DIR] {e}/ -> {full}")
            else:
                size = os.path.getsize(full)
                result.append(f"[FILE] {e} -> {full} ({size:,} bytes)")
        if len(entries) > 100:
            result.append(f"... y {len(entries) - 100} más")
        return "\n".join(result)
    except Exception as e:
        return f"Error: {e}"


def tool_get_datetime(params: dict) -> str:
    """Get current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S (%A)")


def tool_web_search(params: dict) -> str:
    """Simple web fetch — retrieves text from a URL."""
    url = params.get("url", "").strip()
    if not url:
        return "Error: no se proporcionó URL."
    # Basic URL validation
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "Error: solo se permiten URLs http/https."
    if not parsed.hostname:
        return "Error: URL inválida."
    # Block private/internal IPs (SSRF protection)
    hostname = parsed.hostname.lower()
    _blocked = ["localhost", "127.0.0.1", "0.0.0.0", "169.254.", "10.", "192.168.", "172.16."]
    if any(hostname.startswith(b) or hostname == b.rstrip(".") for b in _blocked):
        return "Error: no se permiten URLs a redes internas."
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Laris/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(20000).decode("utf-8", errors="replace")
        # Strip HTML tags for readability
        if "html" in content_type.lower():
            raw = re.sub(r"<script[^>]*>.*?</script>", "", raw, flags=re.DOTALL | re.IGNORECASE)
            raw = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL | re.IGNORECASE)
            raw = re.sub(r"<[^>]+>", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
        return raw[:4000]
    except urllib.error.HTTPError as e:
        return f"Error HTTP {e.code}: {e.reason}"
    except Exception as e:
        return f"Error: {e}"


def tool_python_eval(params: dict) -> str:
    """Evaluate a Python expression safely (math, string ops, etc.)."""
    expr = params.get("expression", "").strip()
    if not expr:
        return "Error: no se proporcionó expresión."
    # Only allow safe builtins
    safe_builtins = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "len": len, "int": int, "float": float,
        "str": str, "bool": bool, "list": list, "dict": dict,
        "range": range, "sorted": sorted, "reversed": reversed,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "True": True, "False": False, "None": None,
        "pow": pow, "divmod": divmod, "hex": hex, "oct": oct, "bin": bin,
    }
    import math
    safe_builtins.update({k: getattr(math, k) for k in dir(math) if not k.startswith("_")})
    try:
        result = eval(expr, {"__builtins__": safe_builtins}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ─── Application discovery (Windows) ─────────────────────────────────────────

def _resolve_lnk(lnk_path: str) -> str | None:
    """Resolve a .lnk shortcut to its target path using PowerShell."""
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             f"(New-Object -ComObject WScript.Shell).CreateShortcut('{lnk_path}').TargetPath"],
            capture_output=True, text=True, timeout=5,
        )
        target = result.stdout.strip()
        if target and os.path.exists(target):
            return target
    except Exception:
        pass
    return None


def _search_start_menu(query: str) -> list[dict]:
    """Search Start Menu shortcuts for apps matching the query."""
    results = []
    query_lower = query.lower()
    start_dirs = [
        os.path.join(os.environ.get("ProgramData", "C:\\ProgramData"),
                      "Microsoft", "Windows", "Start Menu", "Programs"),
        os.path.join(os.path.expanduser("~"),
                      "AppData", "Roaming", "Microsoft", "Windows", "Start Menu", "Programs"),
    ]
    for start_dir in start_dirs:
        if not os.path.isdir(start_dir):
            continue
        for root, _dirs, files in os.walk(start_dir):
            for fname in files:
                if not fname.lower().endswith((".lnk", ".url")):
                    continue
                name_no_ext = os.path.splitext(fname)[0].lower()
                if query_lower in name_no_ext or name_no_ext in query_lower:
                    full_path = os.path.join(root, fname)
                    target = _resolve_lnk(full_path) if fname.lower().endswith(".lnk") else full_path
                    results.append({
                        "name": os.path.splitext(fname)[0],
                        "shortcut": full_path,
                        "executable": target or "(no se pudo resolver)",
                    })
    return results


def _search_registry(query: str) -> list[dict]:
    """Search Windows registry for installed applications matching the query."""
    if os.name != "nt":
        return []
    import winreg
    results = []
    query_lower = query.lower()
    reg_paths = [
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
    ]
    for hive, path in reg_paths:
        try:
            with winreg.OpenKey(hive, path) as key:
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        with winreg.OpenKey(key, subkey_name) as subkey:
                            try:
                                display_name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                            except OSError:
                                continue
                            if query_lower not in display_name.lower():
                                continue
                            install_loc = ""
                            exe_path = ""
                            try:
                                install_loc = winreg.QueryValueEx(subkey, "InstallLocation")[0]
                            except OSError:
                                pass
                            try:
                                exe_path = winreg.QueryValueEx(subkey, "DisplayIcon")[0]
                                # DisplayIcon often has ",0" suffix
                                exe_path = exe_path.split(",")[0].strip('"')
                            except OSError:
                                pass
                            results.append({
                                "name": display_name,
                                "install_location": install_loc,
                                "executable": exe_path if exe_path and os.path.isfile(exe_path) else "",
                            })
                    except OSError:
                        continue
        except OSError:
            continue
    return results


def _search_desktop(query: str) -> list[dict]:
    """Search Desktop for shortcuts matching the query."""
    results = []
    query_lower = query.lower()
    # Get real Desktop path
    desktop = ""
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
        home = os.path.expanduser("~")
        for name in ("Desktop", "Escritorio"):
            candidate = os.path.join(home, name)
            if os.path.isdir(candidate):
                desktop = candidate
                break
    if not desktop or not os.path.isdir(desktop):
        return results
    for fname in os.listdir(desktop):
        name_lower = fname.lower()
        name_no_ext = os.path.splitext(fname)[0].lower()
        if query_lower not in name_no_ext and name_no_ext not in query_lower:
            continue
        full_path = os.path.join(desktop, fname)
        if name_lower.endswith(".lnk"):
            target = _resolve_lnk(full_path)
            results.append({
                "name": os.path.splitext(fname)[0],
                "shortcut": full_path,
                "executable": target or "(no se pudo resolver)",
            })
        elif name_lower.endswith(".url"):
            results.append({
                "name": os.path.splitext(fname)[0],
                "shortcut": full_path,
                "executable": "(acceso directo URL)",
            })
        elif name_lower.endswith(".exe"):
            results.append({
                "name": os.path.splitext(fname)[0],
                "shortcut": full_path,
                "executable": full_path,
            })
    return results


def tool_find_application(params: dict) -> str:
    """Find an installed application by name, scanning registry, Start Menu, and Desktop."""
    query = params.get("name", "").strip()
    if not query:
        return "Error: no se proporcionó nombre de aplicación."

    all_results = []

    # 1. Desktop shortcuts
    desktop_results = _search_desktop(query)
    for r in desktop_results:
        r["source"] = "Escritorio"
    all_results.extend(desktop_results)

    # 2. Start Menu
    start_results = _search_start_menu(query)
    for r in start_results:
        r["source"] = "Menú Inicio"
    all_results.extend(start_results)

    # 3. Windows Registry
    reg_results = _search_registry(query)
    for r in reg_results:
        r["source"] = "Registro Windows"
    all_results.extend(reg_results)

    # 4. PATH lookup (where command)
    try:
        r = subprocess.run(
            ["where", query], capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            for line in r.stdout.strip().splitlines():
                all_results.append({
                    "name": query,
                    "source": "PATH del sistema",
                    "executable": line.strip(),
                })
    except Exception:
        pass

    if not all_results:
        return f"No se encontró ninguna aplicación que coincida con '{query}'."

    # Deduplicate by executable
    seen = set()
    unique = []
    for r in all_results:
        exe = r.get("executable", "")
        key = exe.lower() if exe else r.get("shortcut", "").lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(r)

    lines = [f"Encontradas {len(unique)} coincidencia(s) para '{query}':\n"]
    for i, r in enumerate(unique[:10], 1):
        lines.append(f"{i}. {r['name']} [{r['source']}]")
        if r.get("executable"):
            lines.append(f"   Ejecutable: {r['executable']}")
        if r.get("shortcut"):
            lines.append(f"   Acceso directo: {r['shortcut']}")
        if r.get("install_location"):
            lines.append(f"   Instalación: {r['install_location']}")
    return "\n".join(lines)


def tool_open_application(params: dict) -> str:
    """Find and open an application by name. Scans the system automatically."""
    import webbrowser
    query = params.get("name", "").strip()
    if not query:
        return "Error: no se proporcion\u00f3 nombre de aplicaci\u00f3n."

    # Known web apps/services — open in browser if no desktop app found
    _WEB_APPS = {
        "youtube": "https://www.youtube.com",
        "gmail": "https://mail.google.com",
        "google maps": "https://maps.google.com",
        "maps": "https://maps.google.com",
        "google drive": "https://drive.google.com",
        "drive": "https://drive.google.com",
        "netflix": "https://www.netflix.com",
        "twitter": "https://twitter.com",
        "x": "https://x.com",
        "facebook": "https://www.facebook.com",
        "instagram": "https://www.instagram.com",
        "linkedin": "https://www.linkedin.com",
        "reddit": "https://www.reddit.com",
        "twitch": "https://www.twitch.tv",
        "github": "https://github.com",
        "chatgpt": "https://chat.openai.com",
        "whatsapp web": "https://web.whatsapp.com",
        "google": "https://www.google.com",
        "amazon": "https://www.amazon.com",
        "tiktok": "https://www.tiktok.com",
        "pinterest": "https://www.pinterest.com",
        "canva": "https://www.canva.com",
        "notion": "https://www.notion.so",
        "figma": "https://www.figma.com",
        "trello": "https://trello.com",
        "slack": "https://app.slack.com",
    }

    # Search in order: Desktop → Start Menu → Registry → PATH
    launch_target = None
    source = ""

    # Desktop
    for r in _search_desktop(query):
        if r.get("shortcut"):
            launch_target = r["shortcut"]
            source = f"Escritorio: {r['name']}"
            break

    # Start Menu
    if not launch_target:
        for r in _search_start_menu(query):
            if r.get("executable") and os.path.isfile(r["executable"]):
                launch_target = r["executable"]
                source = f"Menú Inicio: {r['name']}"
                break
            elif r.get("shortcut"):
                launch_target = r["shortcut"]
                source = f"Menú Inicio: {r['name']}"
                break

    # Registry
    if not launch_target:
        for r in _search_registry(query):
            if r.get("executable") and os.path.isfile(r["executable"]):
                launch_target = r["executable"]
                source = f"Registro: {r['name']}"
                break

    # PATH
    if not launch_target:
        try:
            r = subprocess.run(
                ["where", query], capture_output=True, text=True, timeout=5,
            )
            if r.returncode == 0:
                launch_target = r.stdout.strip().splitlines()[0]
                source = f"PATH: {launch_target}"
        except Exception:
            pass

    if not launch_target:
        # Check if it's a known web app/service
        query_lower = query.lower()
        for web_name, web_url in _WEB_APPS.items():
            if query_lower in web_name or web_name in query_lower:
                try:
                    webbrowser.open(web_url)
                    return f"Abriendo {web_name} en el navegador: {web_url}"
                except Exception as e:
                    return f"Error abriendo {web_url}: {e}"
        return f"No se encontró '{query}' instalado en el sistema."

    # Launch it
    try:
        subprocess.Popen(
            ["cmd", "/c", "start", "", launch_target],
            cwd=os.path.expanduser("~"),
        )
        return f"Abriendo {source} ({launch_target})"
    except Exception as e:
        return f"Error al abrir {launch_target}: {e}"


def tool_open_url(params: dict) -> str:
    """Open a URL in the default browser."""
    import webbrowser
    url = params.get("url", "").strip()
    if not url:
        return "Error: no se proporcionó URL."
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return "Error: solo se permiten URLs http/https."
    try:
        webbrowser.open(url)
        return f"Abriendo en navegador: {url}"
    except Exception as e:
        return f"Error abriendo URL: {e}"


def tool_youtube_search(params: dict) -> str:
    """Search YouTube and return the first video results with titles and URLs."""
    query = params.get("query", "").strip()
    if not query:
        return "Error: no se proporcionó búsqueda."
    search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)
    try:
        req = urllib.request.Request(search_url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        })
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Try to extract from ytInitialData JSON (includes titles)
        m = re.search(r'var ytInitialData\s*=\s*(\{.+?\});\s*</script>', html, re.DOTALL)
        if m:
            import json as _json
            data = _json.loads(m.group(1))
            contents = data["contents"]["twoColumnSearchResultsRenderer"]["primaryContents"]["sectionListRenderer"]["contents"]
            items = contents[0]["itemSectionRenderer"]["contents"]
            results = []
            for item in items:
                v = item.get("videoRenderer", {})
                if v.get("videoId"):
                    title = v.get("title", {}).get("runs", [{}])[0].get("text", "Sin titulo")
                    vid_url = f"https://www.youtube.com/watch?v={v['videoId']}"
                    results.append(f"{len(results)+1}. {title}\n   {vid_url}")
                    if len(results) >= 5:
                        break
            if results:
                return "\n".join(results)

        # Fallback: extract video IDs via regex
        ids = re.findall(r'"videoId":"([a-zA-Z0-9_-]{11})"', html)
        unique = list(dict.fromkeys(ids))[:5]
        if unique:
            results = [f"{i+1}. https://www.youtube.com/watch?v={vid}" for i, vid in enumerate(unique)]
            return "\n".join(results)

        return "No se encontraron resultados."
    except Exception as e:
        return f"Error buscando en YouTube: {e}"


# ─── Last tool result buffer (for save_last_result) ─────────────────────────

_last_tool_result = ""


def set_last_tool_result(result: str):
    """Called by the orchestrator to store the last tool result."""
    global _last_tool_result
    _last_tool_result = result


# ─── Last response buffer (for save_last_response) ──────────────────────────

_last_response = ""


def set_last_response(text: str):
    """Called by the orchestrator to store the last LLM text response."""
    global _last_response
    _last_response = text


def tool_save_last_response(params: dict) -> str:
    """Save the last assistant text response to a file."""
    global _last_response
    path = params.get("path", "").strip()
    if not path:
        return "Error: no se proporcion\u00f3 ruta."
    if not _last_response:
        return "Error: no hay respuesta previa para guardar."
    return tool_write_file({"path": path, "content": _last_response})


def tool_save_last_result(params: dict) -> str:
    """Save the last tool result to a file without the LLM needing to copy it."""
    global _last_tool_result
    path = params.get("path", "").strip()
    if not path:
        return "Error: no se proporcionó ruta."
    if not _last_tool_result:
        return "Error: no hay resultado previo para guardar."
    # Reuse write_file for path safety and creation
    return tool_write_file({"path": path, "content": _last_tool_result})


# ─── Tool registry ───────────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "run_command",
        "description": (
            "Ejecuta un comando en la terminal del sistema (PowerShell en Windows, bash en Linux). "
            "Úsalo para instalar paquetes, ejecutar scripts, compilar código, ver procesos, etc. "
            "Para abrir aplicaciones en Windows usa: start \"\" \"ruta\\al\\programa.exe\" o start \"\" \"acceso.lnk\""
        ),
        "parameters": {
            "command": {"type": "string", "description": "El comando a ejecutar", "required": True},
            "timeout": {"type": "integer", "description": "Tiempo límite en segundos (máx 60)", "default": 30},
        },
        "function": tool_run_command,
    },
    {
        "name": "read_file",
        "description": "Lee el contenido de un archivo. Útil para inspeccionar código, configuraciones, logs, etc.",
        "parameters": {
            "path": {"type": "string", "description": "Ruta del archivo (absoluta o relativa al proyecto)", "required": True},
        },
        "function": tool_read_file,
    },
    {
        "name": "write_file",
        "description": "Escribe contenido en un archivo. Crea el archivo y directorios intermedios si no existen.",
        "parameters": {
            "path": {"type": "string", "description": "Ruta del archivo", "required": True},
            "content": {"type": "string", "description": "Contenido a escribir", "required": True},
        },
        "function": tool_write_file,
    },
    {
        "name": "move_file",
        "description": "Mueve o renombra un archivo o carpeta. Crea directorios intermedios automaticamente.",
        "parameters": {
            "source": {"type": "string", "description": "Ruta origen del archivo o carpeta", "required": True},
            "destination": {"type": "string", "description": "Ruta destino (incluir nombre del archivo)", "required": True},
        },
        "function": tool_move_file,
    },
    {
        "name": "copy_file",
        "description": "Copia un archivo o carpeta a otra ubicacion. Crea directorios intermedios automaticamente.",
        "parameters": {
            "source": {"type": "string", "description": "Ruta origen del archivo o carpeta", "required": True},
            "destination": {"type": "string", "description": "Ruta destino (incluir nombre del archivo)", "required": True},
        },
        "function": tool_copy_file,
    },
    {
        "name": "list_directory",
        "description": "Lista el contenido de un directorio mostrando archivos y carpetas.",
        "parameters": {
            "path": {"type": "string", "description": "Ruta del directorio (default: directorio del proyecto)", "default": "."},
        },
        "function": tool_list_directory,
    },
    {
        "name": "datetime",
        "description": "Obtiene la fecha y hora actual del sistema.",
        "parameters": {},
        "function": tool_get_datetime,
    },
    {
        "name": "web_fetch",
        "description": "Descarga y muestra el contenido de texto de una URL. Útil para consultar APIs o páginas web.",
        "parameters": {
            "url": {"type": "string", "description": "URL a consultar (http/https)", "required": True},
        },
        "function": tool_web_search,
    },
    {
        "name": "python_eval",
        "description": "Evalúa una expresión Python (matemáticas, operaciones con strings, etc.). No puede importar módulos ni acceder al sistema de archivos.",
        "parameters": {
            "expression": {"type": "string", "description": "Expresión Python a evaluar", "required": True},
        },
        "function": tool_python_eval,
    },
    {
        "name": "find_application",
        "description": (
            "Busca una aplicación instalada en el sistema por nombre. "
            "Escanea el Escritorio, Menú Inicio, Registro de Windows y el PATH. "
            "Devuelve la ruta del ejecutable (.exe) y su origen. "
            "Úsalo ANTES de intentar abrir una aplicación para verificar que existe y obtener su ruta correcta."
        ),
        "parameters": {
            "name": {"type": "string", "description": "Nombre de la aplicación a buscar (ej: 'steam', 'chrome', 'discord')", "required": True},
        },
        "function": tool_find_application,
    },
    {
        "name": "open_application",
        "description": (
            "Busca y abre una aplicación instalada en el sistema por nombre. "
            "Escanea automáticamente Escritorio, Menú Inicio, Registro de Windows y PATH, "
            "luego lanza la primera coincidencia encontrada. "
            "Úsalo cuando el usuario te pida abrir un programa."
        ),
        "parameters": {
            "name": {"type": "string", "description": "Nombre de la aplicación a abrir (ej: 'steam', 'chrome', 'discord')", "required": True},
        },
        "function": tool_open_application,
    },
    {
        "name": "open_url",
        "description": "Abre una URL en el navegador predeterminado del usuario.",
        "parameters": {
            "url": {"type": "string", "description": "URL a abrir (http/https)", "required": True},
        },
        "function": tool_open_url,
    },
    {
        "name": "youtube_search",
        "description": (
            "Busca videos en YouTube y devuelve los primeros 5 resultados con titulo y URL. "
            "Usalo cuando el usuario pida buscar algo en YouTube o reproducir un video."
        ),
        "parameters": {
            "query": {"type": "string", "description": "Texto a buscar en YouTube", "required": True},
        },
        "function": tool_youtube_search,
    },
    {
        "name": "save_last_result",
        "description": (
            "Guarda el resultado COMPLETO de la última herramienta ejecutada en un archivo. "
            "Úsalo cuando necesites guardar una lista, contenido largo o resultado de otra herramienta en un archivo. "
            "SIEMPRE usa esta herramienta en vez de copiar datos manualmente con write_file, "
            "ya que preserva el contenido completo sin truncar."
        ),
        "parameters": {
            "path": {"type": "string", "description": "Ruta del archivo donde guardar el resultado", "required": True},
        },
        "function": tool_save_last_result,
    },
    {
        "name": "save_last_response",
        "description": (
            "Guarda tu ultima respuesta de texto en un archivo. "
            "Usalo cuando generas contenido (resumenes, textos, informacion) y luego necesitas guardarlo. "
            "Solo necesitas dar la ruta. El contenido se guarda automaticamente completo. "
            "Las carpetas se crean automaticamente si no existen."
        ),
        "parameters": {
            "path": {"type": "string", "description": "Ruta del archivo donde guardar la respuesta", "required": True},
        },
        "function": tool_save_last_response,
    },
]

# Quick lookup by name
_TOOLS_BY_NAME = {t["name"]: t for t in TOOLS}


def get_tool(name: str) -> dict | None:
    return _TOOLS_BY_NAME.get(name)


def execute_tool(name: str, params: dict) -> str:
    """Execute a tool by name and return the result string."""
    tool = get_tool(name)
    if not tool:
        return f"Error: herramienta desconocida '{name}'"
    try:
        return tool["function"](params)
    except Exception as e:
        return f"Error ejecutando {name}: {e}"


def get_tools_description() -> str:
    """Build a text description of all tools for the system prompt."""
    lines = []
    for t in TOOLS:
        params_desc = ""
        if t["parameters"]:
            parts = []
            for pname, pinfo in t["parameters"].items():
                req = " (obligatorio)" if pinfo.get("required") else f" (default: {pinfo.get('default', 'null')})"
                parts.append(f"    - {pname} ({pinfo['type']}): {pinfo['description']}{req}")
            params_desc = "\n" + "\n".join(parts)
        lines.append(f"• {t['name']}: {t['description']}{params_desc}")
    return "\n".join(lines)


def get_tools_json_schema() -> list[dict]:
    """Return tools in Gemini function-calling format."""
    declarations = []
    for t in TOOLS:
        properties = {}
        required = []
        for pname, pinfo in t["parameters"].items():
            ptype = {"string": "STRING", "integer": "INTEGER", "number": "NUMBER", "boolean": "BOOLEAN"}.get(pinfo["type"], "STRING")
            properties[pname] = {"type": ptype, "description": pinfo["description"]}
            if pinfo.get("required"):
                required.append(pname)
        schema = {"type": "OBJECT", "properties": properties}
        if required:
            schema["required"] = required
        declarations.append({
            "name": t["name"],
            "description": t["description"],
            "parameters": schema,
        })
    return declarations
