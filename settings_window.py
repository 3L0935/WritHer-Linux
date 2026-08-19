"""Settings window — CustomTkinter + Pandora Blackboard theme.

Allows the user to configure:
  - Whisper model selection (tiny / base / small / medium / large-v3)
  - LLM model path (.gguf file) and unload timeout
  - LLM server URL (llama-server endpoint)
  - Obsidian vault path (optional, enables vault search tool)
  - Language (en / it / fr)
"""

import sys
import tkinter as tk
import tkinter.filedialog as fd
import customtkinter as ctk

from logger import log
import config
import database as db
import locales
import theme as T
import tts
_WIN_W, _WIN_H = 480, 680


# ── CustomTkinter 5.2.2 Linux scroll fix ─────────────────────────────────────
# Upstream bug: on Linux, _set_scroll_increments leaves yscrollincrement=0
# (canvas default ≈ viewport/10 per "unit"), and _mouse_wheel_all scrolls by
# raw event.delta (±120 from XI2). Result: one wheel tick ≈ 12 viewports →
# jumps straight to the end. Mirror the Windows behavior on Linux.
if sys.platform.startswith("linux"):
    def _vigil_set_scroll_increments(self):
        self._parent_canvas.configure(xscrollincrement=1, yscrollincrement=1)
        # Tag so _vigil_mouse_wheel_all can pick THIS canvas out of the master
        # chain without colliding with CTkButton/CTkFrame's internal CTkCanvas.
        self._parent_canvas._vigil_sf_canvas = True

    def _vigil_mouse_wheel_all(self, event):
        # bind_all fires every CTkScrollableFrame's handler on every scroll;
        # check_if_master_is_canvas passes for ANY scrollable canvas on the
        # master chain — in nested layouts (popup dropdown inside settings pad)
        # that's both. Route the event to the innermost scrollable frame only.
        w = event.widget
        while w is not None:
            if getattr(w, "_vigil_sf_canvas", False):
                break
            w = getattr(w, "master", None)
        if w is not self._parent_canvas:
            return
        step = -int(event.delta / 6) if abs(event.delta) >= 6 else -int(event.delta)
        if self._shift_pressed:
            if self._parent_canvas.xview() != (0.0, 1.0):
                self._parent_canvas.xview("scroll", step, "units")
        else:
            if self._parent_canvas.yview() != (0.0, 1.0):
                self._parent_canvas.yview("scroll", step, "units")

    ctk.CTkScrollableFrame._set_scroll_increments = _vigil_set_scroll_increments
    ctk.CTkScrollableFrame._mouse_wheel_all = _vigil_mouse_wheel_all


class _ScrollableDropdown(ctk.CTkFrame):
    """CTkOptionMenu replacement with a fixed-height scrollable popup."""

    _POPUP_H = 180

    def __init__(self, master, *, values: list[str], variable: tk.StringVar,
                 command=None):
        super().__init__(master, fg_color="transparent")
        self._values = list(values)
        self._var = variable
        self._command = command
        self._popup: ctk.CTkToplevel | None = None

        self._btn = ctk.CTkButton(
            self, textvariable=variable, anchor="w",
            fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
            text_color=T.FG, font=T.FONT_SMALL, corner_radius=6,
            command=self._toggle,
        )
        self._btn.pack(fill="both", expand=True)

    def configure(self, **kw):
        if "values" in kw:
            self._values = list(kw.pop("values"))
        if kw:
            super().configure(**kw)

    def _toggle(self):
        if self._popup and self._popup.winfo_exists():
            self._close_popup()
        else:
            self._open()

    def _open(self):
        self.update_idletasks()
        bx = self._btn.winfo_rootx()
        by = self._btn.winfo_rooty() + self._btn.winfo_height()
        bw = self._btn.winfo_width()

        p = ctk.CTkToplevel(self)
        p.overrideredirect(True)
        p.attributes("-topmost", True)
        p.geometry(f"{bw}x{self._POPUP_H}+{bx}+{by}")
        p.configure(fg_color=T.BG_CARD)
        self._popup = p

        sf = ctk.CTkScrollableFrame(p, fg_color=T.BG_CARD,
                                     scrollbar_button_color=T.BORDER,
                                     scrollbar_button_hover_color=T.BORDER_GLOW)
        sf.pack(fill="both", expand=True, padx=1, pady=1)

        cur = self._var.get()
        for v in self._values:
            ctk.CTkButton(
                sf, text=v, height=26, anchor="w",
                fg_color=T.BG_HOVER if v == cur else "transparent",
                hover_color=T.BORDER_GLOW, text_color=T.FG,
                font=T.FONT_SMALL, corner_radius=0,
                command=lambda val=v: self._pick(val),
            ).pack(fill="x")

        if cur in self._values:
            idx = self._values.index(cur)
            p.after(80, lambda: self._scroll_to(sf, idx))

        p.after(50, p.focus_set)
        p.bind("<FocusOut>", lambda e: p.after(100, self._maybe_close))

    def _scroll_to(self, sf: ctk.CTkScrollableFrame, idx: int) -> None:
        try:
            sf._parent_canvas.yview_moveto(idx / max(len(self._values), 1))
        except Exception:
            pass

    def _maybe_close(self) -> None:
        if not self._popup or not self._popup.winfo_exists():
            return
        try:
            fw = self._popup.focus_get()
            if fw is not None and fw.winfo_toplevel() is self._popup:
                return
        except Exception:
            pass
        self._close_popup()

    def _pick(self, val: str) -> None:
        self._var.set(val)
        self._close_popup()
        if self._command:
            self._command(val)

    def _close_popup(self) -> None:
        if self._popup and self._popup.winfo_exists():
            self._popup.destroy()
        self._popup = None


class SettingsWindow:
    def __init__(self, root: tk.Tk, on_whisper_change=None, on_hotkey_change=None):
        self._root = root
        self._win = None
        self._on_whisper_change_cb = on_whisper_change
        self._on_hotkey_change_cb = on_hotkey_change
        self._whisper_var = None
        self._llm_model_var = None
        self._llm_timeout_var = None
        self._llm_url_var = None
        self._vault_path_var = None
        self._lang_var = None
        self._overlay_pos_var = None
        self._overlay_screen_var = None
        self._tts_mode_var = None
        self._tts_voice_fr_var = None
        self._tts_voice_en_var = None
        self._tts_volume_var = None
        self._tts_voice_fr_menu = None
        self._tts_voice_en_menu = None
        self._tts_speaker_fr_var = None
        self._tts_speaker_en_var = None
        self._tts_speaker_fr_row = None
        self._tts_speaker_en_row = None
        self._tts_speaker_fr_menu = None
        self._tts_speaker_en_menu = None
        self._tts_fr_row = None
        self._tts_en_row = None
        self._hotkey_dict_var = None
        self._hotkey_asst_var = None
        self._answer_timeout_var = None
        self._assistant_name_var = None
        self._llm_gpu_layers_var = None
        self._llm_ctx_size_var = None
        self._provider_var = None
        self._ollama_model_var = None
        self._ollama_url_var = None
        self._ollama_api_key_var = None
        self._ollama_model_dropdown = None    # _ScrollableDropdown instance
        self._ollama_model_entry = None       # fallback CTkEntry if fetch fails
        self._ollama_fetch_btn = None         # refresh button
        self._ollama_fetch_label = None       # status label ("3 models loaded")
        self._llama_pack_row = None       # frame contenant les widgets llama.cpp
        self._ollama_pack_row = None      # frame contenant les widgets Ollama

    def show(self):
        if self._win is not None:
            try:
                if self._win.winfo_exists():
                    self._win.lift()
                    self._win.focus_force()
                    self._sync_ui()
                    return
            except Exception:
                pass
        self._build()
        self._sync_ui()

    # ── Build ─────────────────────────────────────────────────────────────

    def _build(self):
        win = ctk.CTkToplevel(self._root)
        win.configure(fg_color=T.BG_DEEP)
        win.title("Vigil")
        win.protocol("WM_DELETE_WINDOW", self._close)

        sx = win.winfo_screenwidth()
        sy = win.winfo_screenheight()
        x = (sx - _WIN_W) // 2
        y = (sy - _WIN_H) // 2
        win.geometry(f"{_WIN_W}x{_WIN_H}+{x}+{y}")
        self._win = win

        outer = ctk.CTkFrame(win, fg_color=T.BG_DEEP, border_color=T.BORDER,
                             border_width=1, corner_radius=0)
        outer.pack(fill="both", expand=True)

        win.bind("<Escape>", lambda e: self._close())

        # ── Content ──────────────────────────────────────────────────
        content = ctk.CTkFrame(outer, fg_color=T.BG, corner_radius=0)
        content.pack(fill="both", expand=True, padx=1, pady=(0, 1))

        pad = ctk.CTkScrollableFrame(content, fg_color="transparent", corner_radius=0,
                                     scrollbar_button_color=T.BG_HOVER,
                                     scrollbar_button_hover_color=T.ACCENT)
        pad.pack(fill="both", expand=True, padx=T.PAD_XL, pady=T.PAD_L)
        # Prevent the scrollable canvas from stealing keyboard focus from Entry widgets
        pad._parent_canvas.configure(takefocus=False)

        # ── Whisper Model ──────────────────────────────────────────────────────
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1, corner_radius=0).pack(
            fill="x", pady=(0, T.PAD_M))
        ctk.CTkLabel(pad, text=locales.get("setting_whisper_model"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))
        self._whisper_var = tk.StringVar(master=self._win, value=getattr(config, "MODEL_SIZE", "base"))
        ctk.CTkOptionMenu(
            pad,
            values=["tiny", "base", "small", "medium", "large-v3"],
            variable=self._whisper_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
            command=self._on_whisper_change,
        ).pack(fill="x", pady=(0, T.PAD_L))

        # ── LLM Provider ──────────────────────────────────────────────────
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1, corner_radius=0).pack(
            fill="x", pady=(0, T.PAD_M))
        ctk.CTkLabel(pad, text="LLM Provider",
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))
        self._provider_var = tk.StringVar(
            master=self._win,
            value=db.get_setting("llm_provider", "llama_cpp"))
        ctk.CTkOptionMenu(
            pad,
            values=["llama_cpp", "ollama_local", "ollama_cloud"],
            variable=self._provider_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
            command=self._on_provider_change,
        ).pack(fill="x", pady=(0, T.PAD_L))

        # ── llama.cpp settings pack ────────────────────────────────────────
        self._llama_pack_row = ctk.CTkFrame(pad, fg_color="transparent")

        # ── LLM Model ──────────────────────────────────────────────────────
        ctk.CTkFrame(self._llama_pack_row, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))
        ctk.CTkLabel(self._llama_pack_row, text=locales.get("setting_llm_model"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))
        llm_row = ctk.CTkFrame(self._llama_pack_row, fg_color="transparent")
        llm_row.pack(fill="x", pady=(0, T.PAD_L))
        self._llm_model_var = tk.StringVar(master=self._win, value=db.get_setting("llama_model", ""))
        ctk.CTkEntry(llm_row, textvariable=self._llm_model_var,
                     fg_color=T.BG_INPUT, border_color=T.BORDER,
                     text_color=T.FG, font=T.FONT_SMALL,
                     height=32, corner_radius=6).pack(
            side="left", fill="x", expand=True, padx=(0, T.PAD_M))
        ctk.CTkButton(llm_row, text=locales.get("setting_browse"), width=80, height=32,
                      fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
                      border_color=T.BORDER, border_width=1,
                      text_color=T.FG, font=T.FONT_SMALL, corner_radius=6,
                      command=self._browse_model).pack(side="right")

        # ── LLM Unload Timeout ────────────────────────────────────────────
        ctk.CTkFrame(self._llama_pack_row, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))
        ctk.CTkLabel(self._llama_pack_row, text=locales.get("setting_llm_unload"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))
        self._llm_timeout_var = tk.StringVar(
            master=self._win, value=db.get_setting("llama_unload_timeout", "120"))
        ctk.CTkOptionMenu(
            self._llama_pack_row,
            values=["60", "120", "300", "0"],
            variable=self._llm_timeout_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
            command=self._on_llm_timeout_change,
        ).pack(fill="x", pady=(0, T.PAD_L))

        # ── GPU Layers (ngl) ──────────────────────────────────────────
        ctk.CTkLabel(self._llama_pack_row, text=locales.get("setting_llm_gpu_layers"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))
        self._llm_gpu_layers_var = tk.StringVar(
            master=self._win, value=db.get_setting("llm_gpu_layers", "99"))
        ctk.CTkOptionMenu(
            self._llama_pack_row,
            values=["off", "10", "20", "33", "99"],
            variable=self._llm_gpu_layers_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
        ).pack(fill="x", pady=(0, T.PAD_L))

        # ── Context size (-c) ─────────────────────────────────────────
        ctk.CTkLabel(self._llama_pack_row, text=locales.get("setting_llm_ctx_size"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))
        self._llm_ctx_size_var = tk.StringVar(
            master=self._win, value=db.get_setting("llama_ctx_size", "8192"))
        ctk.CTkOptionMenu(
            self._llama_pack_row,
            values=["2048", "4096", "8192", "16384", "32768"],
            variable=self._llm_ctx_size_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
        ).pack(fill="x", pady=(0, T.PAD_L))

        # ── LLM Server URL ────────────────────────────────────────────
        ctk.CTkLabel(self._llama_pack_row, text=locales.get("setting_llm_url"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))

        self._llm_url_var = tk.StringVar(master=self._win, value=getattr(config, "LLAMA_SERVER_URL", ""))
        ctk.CTkEntry(self._llama_pack_row, textvariable=self._llm_url_var,
                     fg_color=T.BG_INPUT, border_color=T.BORDER,
                     text_color=T.FG, font=T.FONT_SMALL,
                     height=32, corner_radius=6).pack(fill="x", pady=(0, T.PAD_L))

        # Initial visibility: llama_cpp visible by default
        self._llama_pack_row.pack(fill="x")

        # ── Ollama settings pack (hidden by default) ───────────────────────
        self._ollama_pack_row = ctk.CTkFrame(pad, fg_color="transparent")

        # Ollama URL (auto-switched by _on_provider_change)
        ctk.CTkLabel(self._ollama_pack_row, text="Ollama URL",
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))
        self._ollama_url_var = tk.StringVar(
            master=self._win, value=db.get_setting("ollama_local_url", "http://localhost:11434"))
        ctk.CTkEntry(self._ollama_pack_row, textvariable=self._ollama_url_var,
                     fg_color=T.BG_INPUT, border_color=T.BORDER,
                     text_color=T.FG, font=T.FONT_SMALL,
                     height=32, corner_radius=6).pack(fill="x", pady=(0, T.PAD_L))

        # Ollama Model (fetch button + dropdown)
        ctk.CTkFrame(self._ollama_pack_row, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))
        ctk.CTkLabel(self._ollama_pack_row, text="Ollama Model",
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))

        model_row = ctk.CTkFrame(self._ollama_pack_row, fg_color="transparent")
        model_row.pack(fill="x", pady=(0, T.PAD_L))

        saved_model = db.get_setting("ollama_model", "")
        if saved_model:
            self._ollama_model_var = tk.StringVar(master=self._win, value=saved_model)
            # Dropdown (populated on fetch)
            self._ollama_model_dropdown = _ScrollableDropdown(
                model_row,
                values=[saved_model],
                variable=self._ollama_model_var,
            )
        else:
            placeholder = "No models — press Refresh"
            self._ollama_model_var = tk.StringVar(master=self._win, value=placeholder)
            self._ollama_model_dropdown = _ScrollableDropdown(
                model_row,
                values=[placeholder],
                variable=self._ollama_model_var,
            )
        self._ollama_model_dropdown.pack(side="left", fill="x", expand=True, padx=(0, T.PAD_M))

        # Refresh button
        self._ollama_fetch_btn = ctk.CTkButton(
            model_row, text="Refresh", width=80, height=32,
            fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
            border_color=T.BORDER, border_width=1,
            text_color=T.FG, font=T.FONT_SMALL, corner_radius=6,
            command=self._fetch_ollama_models,
        )
        self._ollama_fetch_btn.pack(side="right")

        # Entry fallback (hidden by default — shown if fetch returns nothing)
        self._ollama_model_entry = ctk.CTkEntry(
            self._ollama_pack_row, textvariable=self._ollama_model_var,
            fg_color=T.BG_INPUT, border_color=T.BORDER,
            text_color=T.FG, font=T.FONT_SMALL,
            height=32, corner_radius=6,
        )

        # Status label
        self._ollama_fetch_label = ctk.CTkLabel(
            self._ollama_pack_row, text="",
            font=T.FONT_SMALL, text_color=T.FG_DIM, anchor="w",
        )

        # Ollama API Key (cloud only — toggled visibility by _on_provider_change)
        self._ollama_api_key_frame = ctk.CTkFrame(self._ollama_pack_row, fg_color=T.BORDER, height=1,
                                                   corner_radius=0)
        self._ollama_api_key_label = ctk.CTkLabel(self._ollama_pack_row, text="API Key",
                                                   font=T.FONT_TITLE, text_color=T.FG, anchor="w")
        self._ollama_api_key_var = tk.StringVar(
            master=self._win, value=db.get_setting("ollama_api_key", ""))
        self._ollama_api_key_entry = ctk.CTkEntry(self._ollama_pack_row, textvariable=self._ollama_api_key_var,
                                                   fg_color=T.BG_INPUT, border_color=T.BORDER,
                                                   text_color=T.FG, font=T.FONT_SMALL,
                                                   height=32, corner_radius=6, show="*")

        # Separator
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))

        # ── Obsidian Vault Path ───────────────────────────────────────
        ctk.CTkLabel(pad, text=locales.get("setting_obsidian_vault"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))

        self._vault_path_var = tk.StringVar(
            master=self._win, value=getattr(config, "OBSIDIAN_VAULT_PATH", ""))
        vault_row = ctk.CTkFrame(pad, fg_color="transparent")
        vault_row.pack(fill="x", pady=(0, T.PAD_L))

        ctk.CTkEntry(vault_row, textvariable=self._vault_path_var,
                     fg_color=T.BG_INPUT, border_color=T.BORDER,
                     text_color=T.FG, font=T.FONT_SMALL,
                     height=32, corner_radius=6).pack(side="left", fill="x",
                                                       expand=True, padx=(0, T.PAD_M))

        ctk.CTkButton(vault_row, text=locales.get("setting_browse"), width=80, height=32,
                      fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
                      border_color=T.BORDER, border_width=1,
                      text_color=T.FG, font=T.FONT_SMALL,
                      corner_radius=6,
                      command=self._browse_vault).pack(side="right")

        # Separator
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))

        # ── Assistant name ────────────────────────────────────────────
        ctk.CTkLabel(pad, text=locales.get("setting_assistant_name"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))

        self._assistant_name_var = tk.StringVar(
            master=self._win, value=getattr(config, "ASSISTANT_NAME", "Vigil"))
        ctk.CTkEntry(pad, textvariable=self._assistant_name_var,
                     fg_color=T.BG_INPUT, border_color=T.BORDER,
                     text_color=T.FG, font=T.FONT_SMALL,
                     height=32, corner_radius=6).pack(fill="x", pady=(0, T.PAD_L))

        # Separator
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))

        # ── Language ──────────────────────────────────────────────────
        ctk.CTkLabel(pad, text=locales.get("setting_language"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))

        self._lang_var = tk.StringVar(master=self._win, value=getattr(config, "LANGUAGE", "en"))
        lang_menu = ctk.CTkOptionMenu(
            pad,
            values=["en", "it", "fr"],
            variable=self._lang_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
            command=self._on_lang_change,
        )
        lang_menu.pack(fill="x", pady=(0, T.PAD_L))

        # Separator
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))

        # ── Overlay Position ──────────────────────────────────────────
        ctk.CTkLabel(pad, text=locales.get("setting_overlay_position"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))

        self._overlay_pos_var = tk.StringVar(
            master=self._win, value=getattr(config, "OVERLAY_POSITION", "bottom-center"))
        ctk.CTkOptionMenu(
            pad,
            values=["bottom-center", "bottom-left", "bottom-right",
                    "middle-center", "middle-left", "middle-right",
                    "top-center", "top-left", "top-right"],
            variable=self._overlay_pos_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
            command=self._on_overlay_pos_change,
        ).pack(fill="x", pady=(0, T.PAD_L))

        # ── Lock to screen ────────────────────────────────────────────
        ctk.CTkLabel(pad, text=locales.get("setting_overlay_screen"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))

        try:
            from platform_linux import get_xrandr_screens
            _screen_choices = ["auto"] + get_xrandr_screens()
        except Exception:
            _screen_choices = ["auto"]

        self._overlay_screen_var = tk.StringVar(
            master=self._win, value=getattr(config, "OVERLAY_SCREEN", "auto"))
        ctk.CTkOptionMenu(
            pad,
            values=_screen_choices,
            variable=self._overlay_screen_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
            command=self._on_overlay_screen_change,
        ).pack(fill="x", pady=(0, T.PAD_L))

        # ── Answer card timeout ───────────────────────────────────────────
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))
        ctk.CTkLabel(pad, text=locales.get("setting_answer_timeout"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))
        self._answer_timeout_var = tk.StringVar(
            master=self._win,
            value=str(db.get_setting("overlay_answer_timeout", "8")))
        ctk.CTkOptionMenu(
            pad,
            values=["5", "8", "10", "15", "20", "30"],
            variable=self._answer_timeout_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
        ).pack(fill="x", pady=(0, T.PAD_L))

        # Separator
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))

        # ── Hotkeys ──────────────────────────────────────────────────────────
        ctk.CTkLabel(pad, text=locales.get("setting_hotkeys"),
                     font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))

        ctk.CTkLabel(pad, text=locales.get("setting_hotkey_dict_hint"),
                     font=T.FONT_SMALL, text_color=T.FG_DIM,
                     anchor="w").pack(fill="x")
        self._hotkey_dict_var = tk.StringVar(
            master=self._win, value=getattr(config, "HOTKEY", "Ctrl+Alt+W"))
        ctk.CTkEntry(pad, textvariable=self._hotkey_dict_var,
                     fg_color=T.BG_INPUT, border_color=T.BORDER,
                     text_color=T.FG, font=T.FONT_SMALL,
                     height=32, corner_radius=6).pack(fill="x", pady=(0, T.PAD_M))

        ctk.CTkLabel(pad, text=locales.get("setting_hotkey_asst_hint"),
                     font=T.FONT_SMALL, text_color=T.FG_DIM,
                     anchor="w").pack(fill="x")
        self._hotkey_asst_var = tk.StringVar(
            master=self._win, value=getattr(config, "ASSISTANT_HOTKEY", "Ctrl+Alt+R"))
        ctk.CTkEntry(pad, textvariable=self._hotkey_asst_var,
                     fg_color=T.BG_INPUT, border_color=T.BORDER,
                     text_color=T.FG, font=T.FONT_SMALL,
                     height=32, corner_radius=6).pack(fill="x", pady=(0, T.PAD_M))

        ctk.CTkLabel(pad, text=locales.get("setting_hotkey_hint"),
                     font=T.FONT_SMALL, text_color=T.FG_DIM,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_L))

        # ── TTS ───────────────────────────────────────────────────────
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(0, T.PAD_M))
        ctk.CTkLabel(pad, text="TTS", font=T.FONT_TITLE, text_color=T.FG,
                     anchor="w").pack(fill="x", pady=(0, T.PAD_M))

        ctk.CTkLabel(pad, text="Mode", font=T.FONT_SMALL,
                     text_color=T.FG_DIM, anchor="w").pack(fill="x")
        self._tts_mode_var = tk.StringVar(
            master=self._win, value=db.get_setting("tts_mode", "overlay"))
        ctk.CTkOptionMenu(
            pad,
            values=["off", "overlay", "tts", "both"],
            variable=self._tts_mode_var,
            fg_color=T.BG_CARD, button_color=T.BG_HOVER,
            button_hover_color=T.BG_HOVER, text_color=T.FG,
            dropdown_fg_color=T.BG_CARD, dropdown_text_color=T.FG,
            dropdown_hover_color=T.BG_HOVER,
            font=T.FONT_SMALL, corner_radius=6,
            command=self._on_tts_mode_change,
        ).pack(fill="x", pady=(0, T.PAD_M))

        engine_row = ctk.CTkFrame(pad, fg_color="transparent")
        engine_row.pack(fill="x", pady=(0, T.PAD_M))
        ctk.CTkLabel(engine_row, text=locales.get("setting_tts_engine"), font=T.FONT_SMALL,
                     text_color=T.FG_DIM, anchor="w").pack(side="left")
        ctk.CTkLabel(engine_row,
                     text=db.get_setting("tts_engine", "off"),
                     font=T.FONT_SMALL, text_color=T.FG,
                     anchor="e").pack(side="right")

        ctk.CTkLabel(pad, text=locales.get("setting_tts_voice_fr"), font=T.FONT_SMALL,
                     text_color=T.FG_DIM, anchor="w").pack(fill="x")
        fr_voices = [v["name"] for v in tts.list_voices("fr")] or ["(none)"]
        self._tts_voice_fr_var = tk.StringVar(
            master=self._win,
            value=db.get_setting("tts_voice_fr", fr_voices[0]))
        self._tts_fr_row = ctk.CTkFrame(pad, fg_color="transparent")
        self._tts_fr_row.pack(fill="x", pady=(0, T.PAD_M))
        self._tts_voice_fr_menu = _ScrollableDropdown(
            self._tts_fr_row,
            values=fr_voices,
            variable=self._tts_voice_fr_var,
            command=lambda v: (db.save_setting("tts_voice_fr", v), tts.init(),
                               self._update_speaker_row("fr")),
        )
        self._tts_voice_fr_menu.pack(side="left", fill="x", expand=True, padx=(0, T.PAD_M))
        ctk.CTkButton(
            self._tts_fr_row, text=locales.get("setting_more_voices"), width=110, height=32,
            fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
            border_color=T.BORDER, border_width=1,
            text_color=T.FG, font=T.FONT_SMALL, corner_radius=6,
            command=lambda: self._show_more_voices("fr"),
        ).pack(side="right")

        # FR speaker row (hidden when voice is single-speaker)
        self._tts_speaker_fr_row = ctk.CTkFrame(pad, fg_color="transparent")
        ctk.CTkLabel(self._tts_speaker_fr_row, text="Speaker",
                     font=T.FONT_SMALL, text_color=T.FG_DIM, anchor="w").pack(side="left",
                     padx=(0, T.PAD_M))
        _fr_n = tts.get_num_speakers(self._tts_voice_fr_var.get())
        _fr_values = [str(i) for i in range(max(1, _fr_n))]
        _fr_saved = db.get_setting("tts_speaker_fr", "0")
        if _fr_saved not in _fr_values:
            _fr_saved = "0"
        self._tts_speaker_fr_var = tk.StringVar(master=self._win, value=_fr_saved)
        self._tts_speaker_fr_menu = _ScrollableDropdown(
            self._tts_speaker_fr_row,
            values=_fr_values,
            variable=self._tts_speaker_fr_var,
        )
        self._tts_speaker_fr_menu.pack(side="left", fill="x", expand=True, padx=(0, T.PAD_M))
        ctk.CTkButton(
            self._tts_speaker_fr_row, text="\u25b6 Sample", width=90, height=32,
            fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
            border_color=T.BORDER, border_width=1,
            text_color=T.FG, font=T.FONT_SMALL, corner_radius=6,
            command=lambda: self._preview_voice("fr"),
        ).pack(side="left")
        if _fr_n > 1:
            self._tts_speaker_fr_row.pack(fill="x", pady=(0, T.PAD_M))

        ctk.CTkLabel(pad, text=locales.get("setting_tts_voice_en"), font=T.FONT_SMALL,
                     text_color=T.FG_DIM, anchor="w").pack(fill="x")
        en_voices = [v["name"] for v in tts.list_voices("en")] or ["(none)"]
        self._tts_voice_en_var = tk.StringVar(
            master=self._win,
            value=db.get_setting("tts_voice_en", en_voices[0]))
        self._tts_en_row = ctk.CTkFrame(pad, fg_color="transparent")
        self._tts_en_row.pack(fill="x", pady=(0, T.PAD_L))
        self._tts_voice_en_menu = _ScrollableDropdown(
            self._tts_en_row,
            values=en_voices,
            variable=self._tts_voice_en_var,
            command=lambda v: (db.save_setting("tts_voice_en", v), tts.init(),
                               self._update_speaker_row("en")),
        )
        self._tts_voice_en_menu.pack(side="left", fill="x", expand=True, padx=(0, T.PAD_M))
        ctk.CTkButton(
            self._tts_en_row, text=locales.get("setting_more_voices"), width=110, height=32,
            fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
            border_color=T.BORDER, border_width=1,
            text_color=T.FG, font=T.FONT_SMALL, corner_radius=6,
            command=lambda: self._show_more_voices("en"),
        ).pack(side="right")

        # EN speaker row (hidden when voice is single-speaker)
        self._tts_speaker_en_row = ctk.CTkFrame(pad, fg_color="transparent")
        ctk.CTkLabel(self._tts_speaker_en_row, text="Speaker",
                     font=T.FONT_SMALL, text_color=T.FG_DIM, anchor="w").pack(side="left",
                     padx=(0, T.PAD_M))
        _en_n = tts.get_num_speakers(self._tts_voice_en_var.get())
        _en_values = [str(i) for i in range(max(1, _en_n))]
        _en_saved = db.get_setting("tts_speaker_en", "0")
        if _en_saved not in _en_values:
            _en_saved = "0"
        self._tts_speaker_en_var = tk.StringVar(master=self._win, value=_en_saved)
        self._tts_speaker_en_menu = _ScrollableDropdown(
            self._tts_speaker_en_row,
            values=_en_values,
            variable=self._tts_speaker_en_var,
        )
        self._tts_speaker_en_menu.pack(side="left", fill="x", expand=True, padx=(0, T.PAD_M))
        ctk.CTkButton(
            self._tts_speaker_en_row, text="\u25b6 Sample", width=90, height=32,
            fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
            border_color=T.BORDER, border_width=1,
            text_color=T.FG, font=T.FONT_SMALL, corner_radius=6,
            command=lambda: self._preview_voice("en"),
        ).pack(side="left")
        if _en_n > 1:
            self._tts_speaker_en_row.pack(fill="x", pady=(0, T.PAD_M))

        # ── Volume ───────────────────────────────────────────────────
        vol_row = ctk.CTkFrame(pad, fg_color="transparent")
        vol_row.pack(fill="x", pady=(0, T.PAD_M))
        ctk.CTkLabel(vol_row, text=locales.get("setting_tts_volume"), font=T.FONT_SMALL,
                     text_color=T.FG_DIM, anchor="w").pack(side="left")
        self._tts_volume_label = ctk.CTkLabel(vol_row, text="100%",
                                               font=T.FONT_SMALL, text_color=T.FG)
        self._tts_volume_label.pack(side="right")
        try:
            _vol_init = float(db.get_setting("tts_volume", "1.0"))
        except (ValueError, TypeError):
            _vol_init = 1.0
        self._tts_volume_var = tk.DoubleVar(master=self._win, value=_vol_init)

        def _on_volume_slide(v):
            val = round(float(v), 2)
            self._tts_volume_label.configure(text=f"{int(val * 100)}%")
            db.save_setting("tts_volume", str(val))
            tts.init()

        ctk.CTkSlider(
            pad,
            from_=0.0, to=1.0,
            variable=self._tts_volume_var,
            fg_color=T.BG_HOVER, progress_color=T.ACCENT,
            button_color=T.ACCENT, button_hover_color=T.ACCENT_HOVER,
            command=_on_volume_slide,
        ).pack(fill="x", pady=(0, T.PAD_L))

        # ── Save button ───────────────────────────────────────────────
        ctk.CTkButton(
            pad, text=locales.get("setting_saved"), height=36,
            fg_color=T.ACCENT, hover_color=T.ACCENT_HOVER,
            text_color=T.BG_DEEP, font=T.FONT_TITLE,
            corner_radius=6, command=self._save_linux_settings,
        ).pack(fill="x")

        # ── Maintenance ───────────────────────────────────────────────
        ctk.CTkFrame(pad, fg_color=T.BORDER, height=1,
                     corner_radius=0).pack(fill="x", pady=(T.PAD_L, T.PAD_M))
        ctk.CTkButton(
            pad, text=locales.get("setting_rerun_setup"), height=32,
            fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
            border_color=T.BORDER, border_width=1,
            text_color=T.FG_DIM, font=T.FONT_SMALL,
            corner_radius=6, command=self._rerun_setup,
        ).pack(fill="x", pady=(0, T.PAD_M))
        ctk.CTkButton(
            pad, text=locales.get("setting_uninstall"), height=32,
            fg_color=T.BG_CARD, hover_color="#5a0000",
            border_color="#8B0000", border_width=1,
            text_color="#FF6B6B", font=T.FONT_SMALL,
            corner_radius=6, command=self._uninstall,
        ).pack(fill="x", pady=(0, T.PAD_L))

    # ── Drag ──────────────────────────────────────────────────────────────

    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y

    def _on_drag(self, event):
        if self._win:
            x = self._win.winfo_x() + (event.x - self._drag_x)
            y = self._win.winfo_y() + (event.y - self._drag_y)
            self._win.geometry(f"+{x}+{y}")

    def hide(self):
        self._close()

    def _close(self):
        win = self._win
        self._win = None
        if win:
            try:
                win.withdraw()
                win.after(1, win.destroy)
            except Exception:
                pass

    # ── UI sync ───────────────────────────────────────────────────────────

    def _sync_ui(self):
        if self._whisper_var:
            self._whisper_var.set(
                db.get_setting("whisper_model", getattr(config, "MODEL_SIZE", "base")))
        if self._llm_model_var:
            self._llm_model_var.set(db.get_setting("llama_model", ""))
        if self._llm_timeout_var:
            self._llm_timeout_var.set(db.get_setting("llama_unload_timeout", "120"))
        if self._llm_url_var:
            self._llm_url_var.set(getattr(config, "LLAMA_SERVER_URL", ""))
        if self._vault_path_var:
            self._vault_path_var.set(getattr(config, "OBSIDIAN_VAULT_PATH", ""))
        if self._assistant_name_var:
            self._assistant_name_var.set(getattr(config, "ASSISTANT_NAME", "Vigil"))
        if self._lang_var:
            self._lang_var.set(getattr(config, "LANGUAGE", "en"))
        if self._overlay_pos_var:
            self._overlay_pos_var.set(getattr(config, "OVERLAY_POSITION", "bottom-center"))
        if self._overlay_screen_var:
            self._overlay_screen_var.set(getattr(config, "OVERLAY_SCREEN", "auto"))
        if self._tts_mode_var:
            self._tts_mode_var.set(db.get_setting("tts_mode", "overlay"))
        if self._tts_voice_fr_var:
            self._tts_voice_fr_var.set(db.get_setting("tts_voice_fr", ""))
        if self._tts_voice_en_var:
            self._tts_voice_en_var.set(db.get_setting("tts_voice_en", ""))
        if self._tts_volume_var:
            try:
                v = float(db.get_setting("tts_volume", "1.0"))
            except (ValueError, TypeError):
                v = 1.0
            self._tts_volume_var.set(v)
            self._tts_volume_label.configure(text=f"{int(v * 100)}%")
        if self._tts_speaker_fr_var:
            self._tts_speaker_fr_var.set(db.get_setting("tts_speaker_fr", "0"))
        if self._tts_speaker_en_var:
            self._tts_speaker_en_var.set(db.get_setting("tts_speaker_en", "0"))
        if self._hotkey_dict_var:
            self._hotkey_dict_var.set(getattr(config, "HOTKEY", "Ctrl+Alt+W"))
        if self._hotkey_asst_var:
            self._hotkey_asst_var.set(getattr(config, "ASSISTANT_HOTKEY", "Ctrl+Alt+R"))
        if self._answer_timeout_var:
            self._answer_timeout_var.set(
                str(db.get_setting("overlay_answer_timeout", "8")))
        if self._llm_gpu_layers_var:
            self._llm_gpu_layers_var.set(db.get_setting("llm_gpu_layers", "99"))
        if self._llm_ctx_size_var:
            self._llm_ctx_size_var.set(db.get_setting("llama_ctx_size", "8192"))
        if self._provider_var:
            provider = db.get_setting("llm_provider", "llama_cpp")
            self._provider_var.set(provider)
        if self._ollama_model_var:
            self._ollama_model_var.set(db.get_setting("ollama_model", ""))
        if self._ollama_url_var:
            provider = db.get_setting("llm_provider", "llama_cpp")
            if provider == "ollama_local":
                self._ollama_url_var.set(db.get_setting("ollama_local_url", "http://localhost:11434"))
            elif provider == "ollama_cloud":
                self._ollama_url_var.set(db.get_setting("ollama_cloud_url", "https://ollama.com"))
            else:
                self._ollama_url_var.set("http://localhost:11434")
        if self._ollama_api_key_var:
            self._ollama_api_key_var.set(db.get_setting("ollama_api_key", ""))
        # Sync provider panel visibility
        self._on_provider_change(self._provider_var.get() if self._provider_var else "llama_cpp")

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _on_whisper_change(self, value: str):
        if self._on_whisper_change_cb:
            self._on_whisper_change_cb(value)

    def _fetch_ollama_models(self):
        """Fetch available models from Ollama API and populate the dropdown."""
        import threading

        url = (self._ollama_url_var.get() if self._ollama_url_var
               else "http://localhost:11434")
        tags_url = url.rstrip("/") + "/api/tags"

        headers = {}
        provider = self._provider_var.get() if self._provider_var else ""
        if provider == "ollama_cloud" and self._ollama_api_key_var:
            key = self._ollama_api_key_var.get()
            if key:
                headers["Authorization"] = f"Bearer {key}"

        def _do():
            models = []
            err_msg = ""
            try:
                import httpx
                with httpx.Client(timeout=10) as client:
                    resp = client.get(tags_url, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
            except Exception as e:
                err_msg = str(e)

            self._win.after(0, lambda: self._update_ollama_models(models, err_msg))

        # Show loading state
        if self._ollama_fetch_label:
            self._ollama_fetch_label.pack(fill="x", pady=(0, T.PAD_L))
            self._ollama_fetch_label.configure(text="Fetching models...")
        if self._ollama_fetch_btn:
            self._ollama_fetch_btn.configure(state="disabled", text="...")

        threading.Thread(target=_do, daemon=True).start()

    def _update_ollama_models(self, models, error):
        """Update dropdown with fetched models, fall back to entry on failure."""
        if self._ollama_fetch_btn:
            self._ollama_fetch_btn.configure(state="normal", text="Refresh")

        if models:
            saved = db.get_setting("ollama_model", "")
            if saved not in models:
                models.insert(0, saved)  # keep saved model as first option

            if self._ollama_model_dropdown:
                self._ollama_model_dropdown.configure(values=models)
                self._ollama_model_dropdown.pack(side="left", fill="x", expand=True,
                                                  padx=(0, T.PAD_M))
            if self._ollama_model_entry:
                self._ollama_model_entry.pack_forget()

            if self._ollama_fetch_label:
                self._ollama_fetch_label.configure(
                    text=f"{len(models)} model(s) available")
        else:
            # Fetch failed — show text entry as fallback
            if self._ollama_model_dropdown:
                self._ollama_model_dropdown.pack_forget()
            if self._ollama_model_entry:
                self._ollama_model_entry.pack(fill="x", pady=(0, T.PAD_L))

            msg = error if error else "Cannot reach Ollama API"
            if self._ollama_fetch_label:
                self._ollama_fetch_label.configure(text=f"⚠ {msg} — type model manually")

    def _on_provider_change(self, value: str):
        """Show/hide provider-specific settings panels."""
        if value.startswith("ollama"):
            # Switch URL based on sub-provider
            if value == "ollama_local":
                default_url = db.get_setting("ollama_local_url", "") or "http://localhost:11434"
            else:  # ollama_cloud
                default_url = db.get_setting("ollama_cloud_url", "") or "https://ollama.com"
            if self._ollama_url_var:
                self._ollama_url_var.set(default_url)

            # Show API key only for cloud
            if value == "ollama_cloud":
                self._ollama_api_key_frame.pack(fill="x", pady=(0, T.PAD_M))
                self._ollama_api_key_label.pack(fill="x", pady=(0, T.PAD_M))
                self._ollama_api_key_entry.pack(fill="x", pady=(0, T.PAD_L))
            else:
                self._ollama_api_key_frame.pack_forget()
                self._ollama_api_key_label.pack_forget()
                self._ollama_api_key_entry.pack_forget()

            # Toggle pack visibility
            if self._llama_pack_row:
                self._llama_pack_row.pack_forget()
            if self._ollama_pack_row:
                self._ollama_pack_row.pack(fill="x")

            # Auto-fetch available models (deferred to let UI settle)
            if self._win:
                self._win.after(100, self._fetch_ollama_models)
        else:
            if self._ollama_pack_row:
                self._ollama_pack_row.pack_forget()
            if self._llama_pack_row:
                self._llama_pack_row.pack(fill="x")

    def _browse_model(self):
        self._win.withdraw()
        try:
            path = fd.askopenfilename(
                parent=self._root,
                title="Select GGUF model",
                filetypes=[("GGUF files", "*.gguf"), ("All files", "*")],
            )
        finally:
            self._win.deiconify()
            self._win.attributes("-topmost", True)
            self._win.lift()
        if path and self._llm_model_var:
            self._llm_model_var.set(path)

    def _on_llm_timeout_change(self, value: str):
        db.save_setting("llama_unload_timeout", value)
        log.info("LLM unload timeout set to %ss", value)

    def _browse_vault(self):
        self._win.withdraw()
        try:
            path = fd.askdirectory(parent=self._root, title="Select Obsidian Vault")
        finally:
            self._win.deiconify()
            self._win.attributes("-topmost", True)
            self._win.lift()
        if path and self._vault_path_var:
            self._vault_path_var.set(path)

    def _on_lang_change(self, value: str):
        config.LANGUAGE = value
        db.save_setting("language", value)
        log.info("Language changed to %s", value)

    def _on_overlay_pos_change(self, value: str):
        config.OVERLAY_POSITION = value
        db.save_setting("overlay_position", value)
        log.info("Overlay position set to %s", value)

    def _on_overlay_screen_change(self, value: str):
        config.OVERLAY_SCREEN = value
        db.save_setting("overlay_screen", value)
        log.info("Overlay screen set to %s", value)

    def _on_tts_mode_change(self, value: str):
        db.save_setting("tts_mode", value)
        config.TTS_MODE = value
        tts.init()
        log.info("TTS mode set to %s", value)

    def _update_speaker_row(self, lang: str) -> None:
        voice_var = self._tts_voice_fr_var    if lang == "fr" else self._tts_voice_en_var
        row       = self._tts_speaker_fr_row   if lang == "fr" else self._tts_speaker_en_row
        menu      = self._tts_speaker_fr_menu  if lang == "fr" else self._tts_speaker_en_menu
        spkr_var  = self._tts_speaker_fr_var   if lang == "fr" else self._tts_speaker_en_var
        ref_row   = self._tts_fr_row           if lang == "fr" else self._tts_en_row
        if not voice_var or not row or not menu:
            return
        n = tts.get_num_speakers(voice_var.get())
        if n > 1:
            values = [str(i) for i in range(n)]
            menu.configure(values=values)
            cur = spkr_var.get() if spkr_var.get() in values else "0"
            spkr_var.set(cur)
            row.pack(fill="x", pady=(0, T.PAD_M), after=ref_row)
        else:
            row.pack_forget()

    def _preview_voice(self, lang: str) -> None:
        voice_var   = self._tts_voice_fr_var   if lang == "fr" else self._tts_voice_en_var
        speaker_var = self._tts_speaker_fr_var if lang == "fr" else self._tts_speaker_en_var
        if not voice_var:
            return
        voice = voice_var.get()
        if not voice or voice == "(none)":
            return
        speaker_id = None
        if speaker_var and tts.get_num_speakers(voice) > 1:
            speaker_id = int(speaker_var.get())
        tts.preview(voice, speaker_id)

    def _show_more_voices(self, lang: str):
        import threading
        dialog = ctk.CTkToplevel(self._win)
        dialog.title(f"Voices ({lang.upper()})")
        dialog.geometry("500x420")
        dialog.attributes("-topmost", True)

        list_frame = ctk.CTkScrollableFrame(dialog, fg_color=T.BG)
        list_frame.pack(fill="both", expand=True, padx=T.PAD_L, pady=T.PAD_L)

        status_lbl = ctk.CTkLabel(dialog, text=locales.get("setting_loading"),
                                   font=T.FONT_SMALL, text_color=T.FG_DIM)
        status_lbl.pack(pady=T.PAD_M)

        def _populate(voices):
            status_lbl.configure(text=f"{len(voices)} voices available")
            for w in list_frame.winfo_children():
                w.destroy()
            for v in voices:
                row = ctk.CTkFrame(list_frame, fg_color="transparent")
                row.pack(fill="x", pady=2)
                ctk.CTkLabel(row, text=v["name"], font=T.FONT_SMALL,
                             text_color=T.FG, anchor="w").pack(
                    side="left", fill="x", expand=True)
                ctk.CTkButton(
                    row, text=locales.get("setting_download"), width=90, height=28,
                    fg_color=T.BG_CARD, hover_color=T.BG_HOVER,
                    border_color=T.BORDER, border_width=1,
                    text_color=T.FG, font=T.FONT_SMALL, corner_radius=6,
                    command=lambda voice=v: self._download_piper_voice(voice, lang, dialog),
                ).pack(side="right")

        def _fetch():
            voices = tts.fetch_voices(lang)
            dialog.after(0, lambda: _populate(voices))

        threading.Thread(target=_fetch, daemon=True).start()

    def _download_piper_voice(self, voice: dict, lang: str, dialog):
        import threading
        import urllib.request
        from pathlib import Path

        def _do():
            dest_dir = Path.home() / ".local" / "share" / "vigil" / "tts" / "piper"
            dest_dir.mkdir(parents=True, exist_ok=True)
            lang_full, rest = voice["name"].split("-", 1)
            lang_short = lang_full.split("_")[0].lower()
            speaker, quality = rest.rsplit("-", 1)
            base = (
                f"https://huggingface.co/rhasspy/piper-voices/resolve/main"
                f"/{lang_short}/{lang_full}/{speaker}/{quality}/{voice['name']}"
            )
            for ext in (".onnx", ".onnx.json"):
                dest = dest_dir / f"{voice['name']}{ext}"
                if not dest.exists():
                    urllib.request.urlretrieve(base + ext, str(dest))
            db.save_setting(f"tts_voice_{lang}", voice["name"])
            tts.init()
            dialog.after(0, lambda: (
                self._refresh_voice_dropdown(lang),
                dialog.destroy(),
            ))

        threading.Thread(target=_do, daemon=True).start()

    def _refresh_voice_dropdown(self, lang: str):
        voices = [v["name"] for v in tts.list_voices(lang)] or ["(none)"]
        current = db.get_setting(f"tts_voice_{lang}", voices[0])
        if lang == "fr":
            if self._tts_voice_fr_menu:
                self._tts_voice_fr_menu.configure(values=voices)
            if self._tts_voice_fr_var:
                self._tts_voice_fr_var.set(current)
        elif lang == "en":
            if self._tts_voice_en_menu:
                self._tts_voice_en_menu.configure(values=voices)
            if self._tts_voice_en_var:
                self._tts_voice_en_var.set(current)
        self._update_speaker_row(lang)

    def _save_linux_settings(self):
        if self._llm_url_var:
            url = self._llm_url_var.get().strip()
            if url:
                config.LLAMA_SERVER_URL = url
                db.save_setting("llama_server_url", url)
                import assistant as _assistant
                _assistant.reload_backend()
        if self._llm_model_var:
            model = self._llm_model_var.get().strip()
            if model:
                old = db.get_setting("llama_model", "")
                db.save_setting("llama_model", model)
                if model != old:
                    from llm_manager import manager as _mgr
                    _mgr.shutdown()
                    import assistant as _assistant
                    _assistant.reload_backend()
        if self._vault_path_var:
            path = self._vault_path_var.get().strip()
            config.OBSIDIAN_VAULT_PATH = path
            db.save_setting("obsidian_vault_path", path)
        if self._lang_var:
            lang = self._lang_var.get()
            config.LANGUAGE = lang
            db.save_setting("language", lang)
        if self._overlay_pos_var:
            pos = self._overlay_pos_var.get()
            config.OVERLAY_POSITION = pos
            db.save_setting("overlay_position", pos)
        if self._overlay_screen_var:
            screen = self._overlay_screen_var.get()
            config.OVERLAY_SCREEN = screen
            db.save_setting("overlay_screen", screen)
        hk_dict = self._hotkey_dict_var.get().strip() if self._hotkey_dict_var else ""
        hk_asst = self._hotkey_asst_var.get().strip() if self._hotkey_asst_var else ""
        if hk_dict and hk_asst and hk_dict == hk_asst:
            log.warning("Dictation and assistant hotkeys must differ — not saved.")
        else:
            if hk_dict:
                config.HOTKEY = hk_dict
                db.save_setting("hotkey_dict", hk_dict)
            if hk_asst:
                config.ASSISTANT_HOTKEY = hk_asst
                db.save_setting("hotkey_assist", hk_asst)
        if self._answer_timeout_var:
            t = self._answer_timeout_var.get().strip()
            try:
                config.OVERLAY_ANSWER_TIMEOUT = int(t)
                db.save_setting("overlay_answer_timeout", t)
            except ValueError:
                pass
        if self._llm_gpu_layers_var:
            ngl = self._llm_gpu_layers_var.get()
            old_ngl = db.get_setting("llm_gpu_layers", "99")
            db.save_setting("llm_gpu_layers", ngl)
            if ngl != old_ngl:
                from llm_manager import manager as _mgr
                _mgr.shutdown()
        if self._llm_ctx_size_var:
            ctx = self._llm_ctx_size_var.get()
            old_ctx = db.get_setting("llama_ctx_size", "8192")
            db.save_setting("llama_ctx_size", ctx)
            if ctx != old_ctx:
                from llm_manager import manager as _mgr
                _mgr.shutdown()
        if self._assistant_name_var:
            name = self._assistant_name_var.get().strip()
            if name:
                config.ASSISTANT_NAME = name
                db.save_setting("assistant_name", name)
        if self._provider_var:
            provider = self._provider_var.get()
            old_provider = db.get_setting("llm_provider", "llama_cpp")
            config.LLM_PROVIDER = provider
            db.save_setting("llm_provider", provider)
            if provider != old_provider:
                from llm_manager import manager as _mgr
                _mgr.shutdown()
                import assistant as _assistant
                _assistant.reload_backend()
        if self._ollama_model_var:
            model = self._ollama_model_var.get().strip()
            if model:
                db.save_setting("ollama_model", model)
                config.OLLAMA_MODEL = model
        if self._ollama_url_var:
            url = self._ollama_url_var.get().strip()
            if url and self._provider_var:
                provider = self._provider_var.get()
                if provider == "ollama_local":
                    db.save_setting("ollama_local_url", url)
                    config.OLLAMA_LOCAL_URL = url
                elif provider == "ollama_cloud":
                    db.save_setting("ollama_cloud_url", url)
                    config.OLLAMA_CLOUD_URL = url
        if self._ollama_api_key_var:
            key = self._ollama_api_key_var.get().strip()
            db.save_setting("ollama_api_key", key)
            config.OLLAMA_API_KEY = key
        if self._tts_speaker_fr_var:
            try:
                db.save_setting("tts_speaker_fr", str(int(self._tts_speaker_fr_var.get())))
            except ValueError:
                pass
        if self._tts_speaker_en_var:
            try:
                db.save_setting("tts_speaker_en", str(int(self._tts_speaker_en_var.get())))
            except ValueError:
                pass
        tts.init()
        if self._on_hotkey_change_cb:
            self._on_hotkey_change_cb()
        log.info("Settings saved.")

    def _rerun_setup(self):
        import setup_utils
        launched = setup_utils.launch_in_terminal(
            f'uv run python "{setup_utils.REPO_DIR / "first_run.py"}"'
        )
        if not launched:
            log.error("No terminal emulator found. Run: uv run python first_run.py")

    def _uninstall(self):
        import setup_utils
        setup_utils.launch_in_terminal(
            f'bash "{setup_utils.REPO_DIR / "uninstall.sh"}"'
        )
