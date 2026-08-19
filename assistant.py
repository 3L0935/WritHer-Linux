"""llama-server assistant with function calling for web search and vault search."""

import json
import re
import subprocess
import threading
import time as _time
from datetime import datetime
from pathlib import Path
from json_repair import repair_json
from logger import log
import config
import locales
from llm_backend import LlamaServerBackend
from llm_manager import manager as _llm_manager
import app_launcher
import url_shortcuts
import folders
import file_search

def _get_backend():
    """Create the appropriate backend based on DB provider settings.

    Reads the DB directly (like llm_manager does) so config is not the source
    of truth for the LLM stack. Previously this read config.LLAMA_MODEL /
    config.LLM_PROVIDER, which were constants never populated from the DB,
    so the model name sent to llama-server was a dead placeholder.
    """
    import database as db
    provider = db.get_setting("llm_provider", "llama_cpp")
    if provider == "ollama_local":
        return LlamaServerBackend(
            db.get_setting("ollama_local_url", "http://localhost:11434"),
            db.get_setting("ollama_model", ""),
        )
    if provider == "ollama_cloud":
        return LlamaServerBackend(
            db.get_setting("ollama_cloud_url", "https://ollama.com"),
            db.get_setting("ollama_model", ""),
            db.get_setting("ollama_api_key", ""),
        )
    return LlamaServerBackend(
        db.get_setting("llama_server_url", "http://localhost:8081"),
        db.get_setting("llama_model", ""),
    )

_backend = _get_backend()

# ── Multi-turn conversation context ───────────────────────────────────────

_CONTEXT_TIMEOUT  = 30.0       # seconds of inactivity before auto-reset
_context_lock     = threading.Lock()
_conversation_history: list[dict] = []
_last_interaction: float = 0.0
_waiting_for_reply: bool = False
_pending_candidates: list[str] = []
_pending_action: str = ""      # "launch" | "close" | "open_file"

# True when the most recent process() result came from the synthesis pass
# (raw tool result re-fed to the LLM for a spoken-language paraphrase).
# Lets main.py decide whether to TTS while in waiting state — search_files
# results are paraphrased and worth speaking, app_candidates lists are raw
# and would sound bad.
_last_was_synthesised: bool = False


def was_last_synthesised() -> bool:
    return _last_was_synthesised

_WORD_TO_NUM: dict[str, int] = {
    # cardinals
    "one": 1, "un": 1, "uno": 1,
    "two": 2, "deux": 2, "due": 2,
    "three": 3, "trois": 3, "tre": 3,
    "four": 4, "quatre": 4, "quattro": 4,
    "five": 5, "cinq": 5, "cinque": 5,
    # ordinals (user often replies "la première" / "the first")
    "first":  1, "premier": 1, "premiere": 1, "première": 1, "primo": 1, "prima": 1,
    "second": 2, "deuxieme": 2, "deuxième": 2, "seconde": 2, "secondo": 2, "seconda": 2,
    "third":  3, "troisieme": 3, "troisième": 3, "terzo": 3, "terza": 3,
    "fourth": 4, "quatrieme": 4, "quatrième": 4, "quarto": 4, "quarta": 4,
    "fifth":  5, "cinquieme": 5, "cinquième": 5, "quinto": 5, "quinta": 5,
}

# Phonetic Whisper-FR mistranscriptions of the digit "1" — interjections that
# the user may not actually have said but that come out when they say "1".
# Ambiguous (could just mean "huh?"), so only accepted as a number reply when
# the whole transcript is short (≤ 2 words). Avoids treating "hein attends
# c'est quoi" as choice #1.
_PHONETIC_ONE = frozenset({"hein", "han", "ein", "in"})

# Hardcoded keywords that trigger a context reset without going through the
# LLM. Faster than a tool call, frees one tool slot for the small model, and
# the regex match is unambiguous on these phrasings (fr/en/it). Match is on
# the full transcript lowercased+stripped — substring match so things like
# "non, nettoie la conv" still work.
_CLEAR_CONTEXT_PATTERNS = (
    "nettoie la conv", "nettoie la conversation", "efface la conv",
    "efface la conversation", "repart à zéro", "repart a zero",
    "recommence", "remet à zéro", "remet a zero",
    "clear context", "reset context", "start over", "start fresh",
    "pulisci la conversazione", "ricomincia", "azzera",
)


def _is_clear_context_request(text: str) -> bool:
    t = text.strip().lower()
    return any(p in t for p in _CLEAR_CONTEXT_PATTERNS)


def reset_context() -> None:
    global _conversation_history, _last_interaction, _waiting_for_reply
    global _pending_candidates, _pending_action
    with _context_lock:
        _conversation_history = []
        _last_interaction     = 0.0
        _waiting_for_reply    = False
        _pending_candidates   = []
        _pending_action       = ""


def is_waiting() -> bool:
    with _context_lock:
        return _waiting_for_reply


def context_level() -> int:
    """Number of completed turns (each turn = 1 user + 1 assistant message)."""
    with _context_lock:
        return len(_conversation_history) // 2


def _parse_number(text: str) -> int | None:
    """Tolerant number parser for voice replies. Handles:
       - bare digits with trailing punctuation: "1", "1.", "2 !"
       - cardinals fr/en/it: "un", "deux", "two", "due"
       - ordinals fr/en/it: "première", "second", "troisième", "primo"
       - phonetic Whisper-FR misreads of "1" (hein, han, ein, in) when the
         transcript is short — Whisper sometimes hears "1" as the interjection.
    Returns None if nothing parseable is found."""
    t = text.strip().lower().rstrip(".,!?;: ")
    try:
        return int(t)
    except ValueError:
        pass
    words = re.findall(r"\w+", t, flags=re.UNICODE)
    short = len(words) <= 2
    for w in words:
        if w in _WORD_TO_NUM:
            return _WORD_TO_NUM[w]
        if w in _PHONETIC_ONE and short:
            return 1
    return None


# ── Action callbacks (registered by main.py) ──────────────────────────────

_action_callbacks: dict = {}


def register_action(name: str, fn) -> None:
    _action_callbacks[name] = fn


# ── Tool definitions ──────────────────────────────────────────────────────

_SEARCH_TOOLS = {"search_web", "search_obsidian_vault", "search_files"}

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": (
            "Search the web via DuckDuckGo. Use for facts, news, prices, or real-time "
            "information the model does not know internally. "
            "Example: \"cherche sur internet\" or \"what is X\" >> calls with query parameter."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max results (1-10)", "default": 5},
            },
            "required": ["query"],
        },
    },
}

_OBSIDIAN_TOOL = {
    "type": "function",
    "function": {
        "name": "search_obsidian_vault",
        "description": (
            "Search the user's Obsidian vault (.md notes) for content matching a query. "
            "Use when the user asks about information in their notes or saved documents. "
            "Example: \"cherche dans ma vault\" or \"look in my notes\" >> search the vault."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "description": "Max notes to return", "default": 5},
            },
            "required": ["query"],
        },
    },
}


_OPEN_SETTINGS_TOOL = {
    "type": "function",
    "function": {
        "name": "open_settings",
        "description": (
            "Open Vigil's own settings/preferences window. Use ONLY when the user asks to "
            "configure Vigil itself. If the user says \"parametres\" without naming a target, "
            "assume Vigil's settings. "
            "Example: \"ouvre les paramètres de Vigil\" >> open_settings()."
            "Do NOT use for OS/system settings -- use app_action('System Settings', 'launch')."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_CLOSE_SETTINGS_TOOL = {
    "type": "function",
    "function": {
        "name": "close_settings",
        "description": (
            "Close Vigil's own settings window."
            "Example: \"ferme les paramètres\" >> close_settings()."
            "Do NOT use for OS/system settings -- use app_action('System Settings', 'close')."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
}

_APP_ACTION_TOOL = {
    "type": "function",
    "function": {
        "name": "app_action",
        "description": (
            "Launch or close an INSTALLED desktop application (not a website). "
            "The 'action' parameter specifies 'launch' or 'close'. If the exact name "
            "is uncertain, give your best guess -- the system will suggest matches. "
            "Examples: \"lance Firefox\" >> app_action('Firefox', 'launch'), "
            "            \"ferme VLC\" >> app_action('VLC', 'close')."
            "Do NOT use for: websites (use open_url), local folders (use open_folder)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Installed application name.",
                },
                "action": {
                    "type": "string",
                    "enum": ["launch", "close"],
                    "description": "What to do: 'launch' to start, 'close' to stop a running instance.",
                },
            },
            "required": ["name", "action"],
        },
    },
}

_OPEN_URL_TOOL = {
    "type": "function",
    "function": {
        "name": "open_url",
        "description": (
            "Open a website in the default browser. Pass the site keyword as the user "
            "spoke it (e.g. 'youtube', 'github', 'gmail'). ~50 sites are recognized. "
            "Example: \"ouvre YouTube\" >> open_url('youtube')."
            "Do NOT use for: desktop apps (use app_action), folders (use open_folder)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "description": "Site keyword the user spoke (e.g. 'youtube', 'github', 'gmail').",
                },
            },
            "required": ["target"],
        },
    },
}

_OPEN_FOLDER_TOOL = {
    "type": "function",
    "function": {
        "name": "open_folder",
        "description": (
            "Open one of the user's standard folders (Documents, Downloads, Pictures, "
            "Videos, Music, Desktop, Templates, Public) in the file manager. "
            "Pass a short folder keyword the user mentioned (in any language). "
            "Example: \"ouvre mes téléchargements\" >> open_folder('downloads')."
            "Do NOT use for: apps (use app_action), websites (use open_url)."
            "Do NOT invent paths or use slashes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Folder keyword the user spoke (e.g. 'downloads', 'documents', 'musique').",
                },
            },
            "required": ["name"],
        },
    },
}

_SEARCH_FILES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_files",
        "description": (
            "Search standard folders for a specific file by name or keywords. "
            "After the results, the user can reply with a number to open the chosen file. "
            "Set include_date=true for time-sensitive files (invoices, screenshots). "
            "Set include_size=true ONLY if the user explicitly asks for file size. "
            "Example: \"cherche facture de mars dans downloads\" >> "
            "        search_files(folder='downloads', query='facture mars')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "folder": {
                    "type": "string",
                    "description": "Standard folder keyword (downloads, documents, pictures, music, videos, desktop, ...).",
                },
                "query": {
                    "type": "string",
                    "description": "What to look for, in natural language (e.g. 'facture mars', 'photo Paris', 'CV').",
                },
                "include_date": {
                    "type": "boolean",
                    "description": "Set true when the file type makes the modification date useful (invoices, dated documents). Default false.",
                    "default": False,
                },
                "include_size": {
                    "type": "boolean",
                    "description": "Set true ONLY when the user explicitly asked for the file size. Default false.",
                    "default": False,
                },
            },
            "required": ["folder", "query"],
        },
    },
}

_ASK_USER_CHOICE_TOOL = {
    "type": "function",
    "function": {
        "name": "ask_user_choice",
        "description": (
            "Ask the user to pick between 2-4 similar installed apps when the request is "
            "genuinely ambiguous. The user replies with a number (1, 2, 3, ...) and the "
            "chosen app is then launched or closed. Use this ONLY after an app_action retry "
            "when several candidates fit the user's intent equally well. "
            "Do NOT use for single-best-match cases — call app_action directly instead."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-4 app names to present to the user, copied exactly from the installed-apps list.",
                },
                "action": {
                    "type": "string",
                    "enum": ["launch", "close"],
                    "description": "What to do with the chosen app.",
                },
            },
            "required": ["options", "action"],
        },
    },
}


def _get_tools() -> list[dict]:
    """Return tool definitions, cached after first build."""
    if not hasattr(_get_tools, "_cache"):
        tools = [_WEB_SEARCH_TOOL, _OPEN_SETTINGS_TOOL, _CLOSE_SETTINGS_TOOL,
                 _APP_ACTION_TOOL, _OPEN_URL_TOOL, _OPEN_FOLDER_TOOL,
                 _SEARCH_FILES_TOOL, _ASK_USER_CHOICE_TOOL]
        if config.OBSIDIAN_VAULT_PATH:
            tools.append(_OBSIDIAN_TOOL)
        _get_tools._cache = tools
    return _get_tools._cache


# ── System prompt ─────────────────────────────────────────────────────────

def _system_prompt() -> str:
    return locales.get("system_prompt", name=getattr(config, "ASSISTANT_NAME", "Vigil"))


# ── Function dispatcher ───────────────────────────────────────────────────

_RETRY_STOPWORDS = frozenset({
    # action verbs / glue (fr/en/it) — ignored when mining keywords from user_text
    "lance", "lances", "lancer", "ouvre", "ouvres", "ouvrir", "démarre", "démarrer",
    "ferme", "fermer", "quitte", "quitter", "arrête", "arrêter", "peux", "peux-tu",
    "tu", "me", "moi", "stp", "svp",
    "launch", "open", "start", "run", "close", "quit", "stop", "please", "could", "can",
    "you", "me", "my",
    "apri", "avvia", "chiudi", "lancia", "ferma", "puoi",
    "le", "la", "les", "un", "une", "des", "du", "de", "ton", "ta", "tes", "mon", "ma",
    "mes", "ce", "cet", "cette", "ces",
    "the", "a", "an", "your", "my", "this", "that",
    "il", "lo", "gli", "i", "le", "uno", "una",
})


def _extract_intent_tokens(text: str) -> list[str]:
    """Strip stopwords from user_text so the remaining tokens are the ones
    that actually identify the app (e.g. 'musique', 'navigateur')."""
    if not text:
        return []
    words = re.findall(r"[\w'’-]+", text.lower())
    return [w for w in words if w and w not in _RETRY_STOPWORDS and len(w) > 2]


def _build_launch_retry_context(app_name: str, user_text: str) -> str:
    """Build retry feedback for the LLM after a launch_app failure.

    Mines intent tokens from user_text (minus stopwords) and fuzzy-matches
    each against installed apps so short significant words like 'musique'
    surface the right candidates instead of being drowned by the full
    mistranscribed sentence. Language follows config.LANGUAGE so the model's
    plain-text fallback ends up in the user's language.
    """
    seen: set[str] = set()
    cands: list[str] = []
    # 1. Intent tokens from user's raw utterance (most semantic)
    for token in _extract_intent_tokens(user_text):
        for c in app_launcher.find_candidates(token, n=4, cutoff=0.5):
            if c not in seen:
                seen.add(c)
                cands.append(c)
    # 2. Fallback on the failed name itself (handles near-typo cases)
    for c in app_launcher.find_candidates(app_name, n=5, cutoff=0.3):
        if c not in seen:
            seen.add(c)
            cands.append(c)
    cands = cands[:8]
    if not cands:
        return locales.get("retry_launch_ctx_empty", name=app_name)
    meta = {a["name"]: a for a in app_launcher.list_all_apps()}
    lines = []
    for c in cands:
        m = meta.get(c, {})
        generic = m.get("generic", "")
        kws = [k.strip() for k in m.get("keywords", "").split(";") if k.strip()][:5]
        parts = [f"- {c}"]
        if generic:
            parts.append(f"({generic})")
        if kws:
            parts.append(f"[{', '.join(kws)}]")
        lines.append(" ".join(parts))
    return locales.get("retry_launch_ctx", name=app_name, list="\n".join(lines))


def _dispatch(name: str, args: dict, user_text: str = "") -> tuple[str, str | None]:
    """Execute a tool and return (user_facing_text, retry_context_or_None).

    If retry_context is non-None, the caller (process) should re-query the LLM
    with this as a tool result so it can correct a failed launch_app call.
    """
    global _waiting_for_reply, _pending_candidates, _pending_action
    log.info("Assistant dispatch: %s(%s)", name, args)

    try:
        if name == "search_web":
            from ddgs import DDGS
            query = args.get("query", "")
            max_results = min(int(args.get("max_results", 5)), 10)
            results = list(DDGS().text(query, max_results=max_results))
            if not results:
                return (locales.get("web_no_results", query=query), None)
            lines = []
            for r in results:
                title = r.get("title", "")
                body  = r.get("body", "")
                href  = r.get("href", "")
                lines.append(f"**{title}**\n{body}\n{href}")
            return ("\n\n".join(lines), None)

        elif name == "open_settings":
            cb = _action_callbacks.get("open_settings")
            if cb:
                cb()
            return (locales.get("settings_opened"), None)

        elif name == "close_settings":
            cb = _action_callbacks.get("close_settings")
            if cb:
                cb()
            return (locales.get("settings_closed"), None)

        elif name == "search_obsidian_vault":
            if not config.OBSIDIAN_VAULT_PATH or not Path(config.OBSIDIAN_VAULT_PATH).is_dir():
                return (locales.get("vault_not_configured"), None)
            from obsidian import search_vault
            results = search_vault(
                query=args.get("query", ""),
                vault_path=config.OBSIDIAN_VAULT_PATH,
                max_results=args.get("max_results", 5),
            )
            if not results:
                return (locales.get("vault_no_results", query=args.get("query", "")), None)
            lines = []
            for r in results:
                lines.append(f"**{r['title']}**\n{r['excerpt']}")
            return ("\n\n".join(lines), None)

        elif name == "app_action":
            app_name = (args.get("name") or "").strip()
            action   = args.get("action", "launch")
            if action not in ("launch", "close"):
                action = "launch"
            do = app_launcher.launch if action == "launch" else app_launcher.close
            ok, label = do(app_name)
            if ok:
                key = "app_launched" if action == "launch" else "app_closed"
                return (locales.get(key, name=label), None)
            # Exact match failed — try fuzzy
            candidates = app_launcher.find_candidates(app_name)
            if len(candidates) == 1:
                ok2, label2 = do(candidates[0])
                if action == "launch":
                    text = locales.get("app_launched", name=label2) if ok2 else locales.get("app_not_found", name=label2)
                else:
                    text = locales.get("app_closed", name=label2) if ok2 else locales.get("app_close_failed", name=label2)
                return (text, None)
            if len(candidates) > 1:
                with _context_lock:
                    _waiting_for_reply  = True
                    _pending_candidates = candidates
                    _pending_action     = action
                lines = "\n".join(f"{i + 1}: {c}" for i, c in enumerate(candidates))
                return (locales.get("app_candidates", list=lines), None)
            # Zero candidates. For launch, retry with hints (and URL hint if
            # the name matches a known web shortcut). For close, just report.
            if action == "close":
                return (locales.get("app_close_failed", name=app_name), None)
            retry_ctx = _build_launch_retry_context(app_name, user_text)
            if url_shortcuts.is_known(app_name):
                retry_ctx = (
                    retry_ctx + "\n\n"
                    + locales.get("retry_launch_url_hint", name=app_name)
                )
            return (locales.get("app_not_found", name=app_name), retry_ctx)

        elif name == "search_files":
            folder = (args.get("folder") or "").strip()
            query = (args.get("query") or "").strip()
            include_date = bool(args.get("include_date", False))
            include_size = bool(args.get("include_size", False))
            r = file_search.search(folder, query,
                                   include_size=include_size,
                                   include_date=include_date)
            if r["folder_resolved"] is None:
                return (locales.get("folder_unknown", name=folder), None)
            results = r["found"] or r["similar"]
            if not results:
                return (locales.get("file_no_results",
                                    folder=r["folder_resolved"],
                                    query=query), None)
            # Set multi-turn state so a numbered reply opens a file.
            paths = [item["path"] for item in results]
            with _context_lock:
                _waiting_for_reply  = True
                _pending_candidates = paths
                _pending_action     = "open_file"
            # Build a compact human-readable list. Skip the absolute path in
            # the LLM's view — paths are noisy tokens. The LLM only needs
            # name + optional date/size + matched (for similar) to formulate.
            lines: list[str] = []
            for i, item in enumerate(results, 1):
                bits = [item["name"]]
                extras = []
                if "mtime" in item:
                    extras.append(item["mtime"])
                if "size_kb" in item:
                    extras.append(f"{item['size_kb']} KB")
                if "matched" in item:
                    extras.append("partial: " + ", ".join(item["matched"]))
                if extras:
                    bits.append(f"({'; '.join(extras)})")
                lines.append(f"{i}. {' '.join(bits)}")
            list_str = "\n".join(lines)
            if r["found"]:
                msg = locales.get("file_results_found",
                                  folder=r["folder_resolved"],
                                  list=list_str)
            else:
                msg = locales.get("file_results_similar",
                                  folder=r["folder_resolved"],
                                  query=query, list=list_str)
            return (msg, None)

        elif name == "ask_user_choice":
            options = args.get("options") or []
            action  = args.get("action", "launch")
            if action not in ("launch", "close"):
                action = "launch"
            opts = [str(o) for o in options if o][:4]
            if not opts:
                return (locales.get("app_not_found", name=""), None)
            with _context_lock:
                _waiting_for_reply  = True
                _pending_candidates = opts
                _pending_action     = action
            lines = "\n".join(f"{i + 1}: {c}" for i, c in enumerate(opts))
            return (locales.get("app_candidates", list=lines), None)

        elif name == "open_url":
            target = (args.get("target") or "").strip()
            url = url_shortcuts.resolve(target)
            if not url:
                return (locales.get("url_invalid", target=target), None)
            try:
                subprocess.Popen(
                    ["xdg-open", url],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception as exc:
                return (locales.get("url_invalid", target=target), None)
            # TTS-friendly: return the spoken keyword if it's a known shortcut,
            # else extract just the hostname stem ("https://www.example.com/x"
            # → "example"). Avoids reading "https colon slash slash..." aloud.
            if url_shortcuts.is_known(target):
                display = target
            else:
                from urllib.parse import urlparse
                host = urlparse(url).netloc.removeprefix("www.")
                display = host.split(".")[0] if host else target
            return (locales.get("url_opened", url=display), None)

        elif name == "open_folder":
            folder_name = (args.get("name") or "").strip()
            path = folders.resolve(folder_name)
            if path is None:
                return (locales.get("folder_unknown", name=folder_name), None)
            if not path.exists():
                return (locales.get("folder_missing", path=str(path)), None)
            try:
                subprocess.Popen(
                    ["xdg-open", str(path)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
            except Exception:
                return (locales.get("folder_missing", path=str(path)), None)
            # TTS-friendly: pass the keyword the user actually said rather than
            # the absolute filesystem path (which would be read literally).
            return (locales.get("folder_opened", path=folder_name), None)

        else:
            return (locales.get("unknown_command", name=name), None)

    except Exception as exc:
        log.error("Dispatch error: %s", exc)
        return (locales.get("error", detail=str(exc)), None)


# ── Public API ────────────────────────────────────────────────────────────

def ping_llama_server() -> bool:
    """Quick connectivity check. Returns True if LLM backend is reachable."""
    return _backend.ping()


def reload_backend():
    global _backend
    _backend = _get_backend()
    log.info("LLM backend reloaded (%s)", type(_backend).__name__)


def process(text: str) -> str:
    """Process transcribed text through llama-server. Returns answer string."""
    global _conversation_history, _last_interaction, _waiting_for_reply
    global _pending_candidates, _pending_action

    _llm_manager.ensure_running()
    log.info("Assistant input: %r", text)

    global _last_was_synthesised
    _last_was_synthesised = False

    # Hardcoded shortcut: clear context request bypasses the LLM entirely.
    # Saves a tool slot for the small model and is unambiguous on the listed
    # phrasings.
    if _is_clear_context_request(text):
        reset_context()
        return locales.get("context_cleared")

    # Snapshot mutable state under the lock
    with _context_lock:
        waiting        = _waiting_for_reply
        candidates     = list(_pending_candidates)
        action         = _pending_action
        history        = list(_conversation_history)
        last_time      = _last_interaction

    # Auto-reset on timeout
    now = _time.monotonic()
    if history and (now - last_time) > _CONTEXT_TIMEOUT:
        log.info("Context timeout — resetting conversation")
        reset_context()
        with _context_lock:
            history  = []
            waiting  = False

    # Resolve pending numbered reply
    if waiting:
        n = _parse_number(text)
        if n is not None and 1 <= n <= len(candidates):
            chosen = candidates[n - 1]
            log.info("Resolving candidate %d: %s (action=%s)", n, chosen, action)
            with _context_lock:
                _waiting_for_reply  = False
                _pending_candidates = []
                _pending_action     = ""
            if action == "open_file":
                # `chosen` is an absolute file path here, not an app name.
                from pathlib import Path as _Path
                p = _Path(chosen)
                if not p.exists():
                    result = locales.get("file_open_failed", path=chosen)
                else:
                    try:
                        subprocess.Popen(
                            ["xdg-open", chosen],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            start_new_session=True,
                        )
                        result = locales.get("file_opened", name=p.name)
                    except Exception:
                        result = locales.get("file_open_failed", path=chosen)
            elif action == "launch":
                ok, label = app_launcher.launch(chosen)
                result = locales.get("app_launched", name=label) if ok else locales.get("app_not_found", name=label)
            else:
                ok, label = app_launcher.close(chosen)
                result = locales.get("app_closed", name=label) if ok else locales.get("app_close_failed", name=label)
            with _context_lock:
                _conversation_history.append({"role": "user",      "content": text})
                _conversation_history.append({"role": "assistant",  "content": result})
                if len(_conversation_history) > 16:
                    _conversation_history = _conversation_history[-16:]
                _last_interaction = _time.monotonic()
            return result
        else:
            # Not a valid number — drop waiting state and process normally
            with _context_lock:
                _waiting_for_reply  = False
                _pending_candidates = []
                _pending_action     = ""

    # Normal LLM flow (with history for multi-turn)
    messages = [{"role": "system", "content": _system_prompt()}] + history + [
        {"role": "user", "content": text},
    ]
    data = _backend.chat(messages=messages, tools=_get_tools())

    if data is None:
        return locales.get("not_understood")

    result = locales.get("not_understood")
    _context_cleared_flag = False
    try:
        choices = data.get("choices", [])
        if not choices:
            return locales.get("not_understood")

        msg = choices[0].get("message", {})
        tool_calls = msg.get("tool_calls")

        if tool_calls:
            tc = tool_calls[0]
            fn_name = tc["function"]["name"]
            _context_cleared_flag = (fn_name == "clear_context")
            raw_args = tc["function"].get("arguments", "{}")
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = repair_json(raw_args) or {}
            else:
                args = raw_args

            raw_result, retry_ctx = _dispatch(fn_name, args, user_text=text)

            # Launch retry: LLM hit a not-found — give it candidate apps and
            # let it try again once. No further retry from that second call.
            if retry_ctx is not None:
                log.info("launch_app retry: feeding %d chars of context back to LLM", len(retry_ctx))
                tool_call_id = tc.get("id", "tc_0")
                retry_messages = messages + [
                    {"role": "assistant", "tool_calls": [tc]},
                    {"role": "tool", "tool_call_id": tool_call_id, "content": retry_ctx},
                ]
                retry_data = _backend.chat(messages=retry_messages, tools=_get_tools())
                if retry_data:
                    try:
                        retry_msg = retry_data.get("choices", [{}])[0].get("message", {})
                        retry_tcs = retry_msg.get("tool_calls")
                        if retry_tcs:
                            rtc = retry_tcs[0]
                            retry_fn = rtc["function"]["name"]
                            retry_raw = rtc["function"].get("arguments", "{}")
                            if isinstance(retry_raw, str):
                                try:
                                    retry_args = json.loads(retry_raw)
                                except json.JSONDecodeError:
                                    retry_args = repair_json(retry_raw) or {}
                            else:
                                retry_args = retry_raw
                            # Execute once, ignore any further retry_ctx to avoid loops
                            retry_text, _ = _dispatch(retry_fn, retry_args, user_text=text)
                            if retry_text:
                                raw_result = retry_text
                        else:
                            retry_content = retry_msg.get("content", "").strip()
                            if retry_content:
                                log.info("Retry text response: %s", retry_content[:120])
                                raw_result = retry_content
                    except (KeyError, IndexError, TypeError) as exc:
                        log.error("Retry parsing error: %s", exc)
                else:
                    log.warning("Retry returned None — using original not-found message")

            # Search tools: feed results back to LLM for a concise synthesis
            elif fn_name in _SEARCH_TOOLS:
                tool_call_id = tc.get("id", "tc_0")
                synthesis_messages = messages + [
                    {"role": "assistant", "tool_calls": [tc]},
                    {"role": "tool", "tool_call_id": tool_call_id, "content": raw_result},
                ]
                syn_data = _backend.chat(messages=synthesis_messages, tools=None)
                if syn_data:
                    syn_choices = syn_data.get("choices", [])
                    if syn_choices:
                        syn_content = (syn_choices[0]
                                       .get("message", {})
                                       .get("content", "")
                                       .strip())
                        if syn_content:
                            log.info("Synthesis: %s", syn_content[:120])
                            raw_result = syn_content
                            _last_was_synthesised = True
                        else:
                            log.warning("Synthesis empty — using raw result")
                else:
                    log.warning("Synthesis returned None — using raw result")

            result = raw_result

        else:
            # No tool call: plain text response
            content = msg.get("content", "").strip()
            if content:
                log.info("LLM text response: %s", content)
                result = content

    except (KeyError, IndexError, TypeError) as exc:
        log.error("Response parsing error: %s", exc)
        return locales.get("not_understood")

    # Append this turn to history (skip if clear_context was called)
    with _context_lock:
        if not _context_cleared_flag:
            _conversation_history.append({"role": "user",      "content": text})
            _conversation_history.append({"role": "assistant",  "content": result})
            if len(_conversation_history) > 16:
                _conversation_history = _conversation_history[-16:]
        _last_interaction = _time.monotonic()

    return result
