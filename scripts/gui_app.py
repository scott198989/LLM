"""
GPT Model Inference — Desktop GUI
==================================
A customtkinter desktop application for interacting with your trained model.

Features
--------
  • Browse and load any .pt checkpoint from models/checkpoints/
  • Real-time token-by-token streaming display with colour-coded sections
      – Prompt echo     : steel blue
      – CoT reasoning   : grey italic  (<|think|> … <|/think|>)
      – Model answer    : white
      – Special tokens  : amber
  • All sampling parameters adjustable via sliders:
      – Temperature · Top-K · Top-P · Max Tokens · Repetition Penalty
  • Four sampling modes: Combined (top-k + top-p) · Top-K · Top-P · Greedy
  • Chain-of-Thought toggle
  • Stop generation mid-stream
  • Live status bar: token count · tokens/sec · GPU memory
  • Copy output to clipboard
  • Persistent settings (saved to gui_settings.json between sessions)

Usage
-----
    python scripts/gui_app.py
    python scripts/gui_app.py --ckpt models/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog, font as tkfont

import customtkinter as ctk

# ── Inference engine ──────────────────────────────────────────────────────────
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS)
from inference import InferenceEngine, GenStats  # noqa: E402

# ── Optional torch for GPU memory display ────────────────────────────────────
try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

# ── App-level theme ───────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ── Paths ──────────────────────────────────────────────────────────────────────
_REPO_ROOT       = os.path.dirname(_SCRIPTS)
_CKPT_DIR        = os.path.join(_REPO_ROOT, "models", "checkpoints")
_SETTINGS_FILE   = os.path.join(_REPO_ROOT, "logs", "gui_settings.json")
_SYS_PROMPT_FILE = os.path.join(_REPO_ROOT, "system_prompt.txt")

# ── Output text-tag styles ────────────────────────────────────────────────────
_MONO = ("Courier New", 11) if sys.platform == "win32" else ("Courier", 11)

_TAGS: dict[str, dict] = {
    "prompt":   {"foreground": "#7aacf5", "font": _MONO},
    "think":    {"foreground": "#888888", "font": (*_MONO, "italic")},
    "answer":   {"foreground": "#e8e8e8", "font": _MONO},
    "special":  {"foreground": "#ffa040", "font": _MONO},
    "system":   {"foreground": "#5a5a6a", "font": (*_MONO, "italic")},
    "error":    {"foreground": "#ff5555", "font": _MONO},
    "sep":      {"foreground": "#444466"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_settings() -> dict:
    try:
        with open(_SETTINGS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    os.makedirs(os.path.dirname(_SETTINGS_FILE), exist_ok=True)
    try:
        with open(_SETTINGS_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def _gpu_mem_str() -> str:
    if not _HAS_TORCH or not torch.cuda.is_available():
        return ""
    res = torch.cuda.memory_reserved()  / 1024 ** 3
    tot = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    return f"  GPU {res:.1f}/{tot:.0f} GB"


# ---------------------------------------------------------------------------
# Slider row  (label + slider + live value label)
# ---------------------------------------------------------------------------

class SliderRow(ctk.CTkFrame):
    """
    A labelled slider that shows its current value next to the title.

    Usage:
        row = SliderRow(parent, label="Temperature", from_=0.01, to=2.0,
                        default=0.8, fmt="{:.2f}")
        value = row.get()
    """

    def __init__(self, parent, label: str, from_: float, to: float,
                 default: float, fmt: str = "{:.2f}",
                 steps: int = 200, **kwargs):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self.fmt = fmt

        self.columnconfigure(0, weight=1)

        # ── Title row ──────────────────────────────────────────────────────────
        title_row = ctk.CTkFrame(self, fg_color="transparent")
        title_row.grid(row=0, column=0, sticky="ew")
        title_row.columnconfigure(0, weight=1)

        ctk.CTkLabel(title_row, text=label,
                     font=ctk.CTkFont(size=12),
                     anchor="w").grid(row=0, column=0, sticky="w")

        self._val_label = ctk.CTkLabel(title_row, text=fmt.format(default),
                                       font=ctk.CTkFont(size=12, weight="bold"),
                                       width=52, anchor="e")
        self._val_label.grid(row=0, column=1, sticky="e")

        # ── Slider ──────────────────────────────────────────────────────────────
        self._var = tk.DoubleVar(value=default)
        self._slider = ctk.CTkSlider(
            self,
            from_=from_, to=to,
            number_of_steps=steps,
            variable=self._var,
            command=self._on_change,
        )
        self._slider.grid(row=1, column=0, sticky="ew", pady=(2, 6))

    def _on_change(self, _val) -> None:
        self._val_label.configure(text=self.fmt.format(self._var.get()))

    def get(self) -> float:
        return self._var.get()

    def set(self, value: float) -> None:
        self._var.set(value)
        self._val_label.configure(text=self.fmt.format(value))


# ---------------------------------------------------------------------------
# Left sidebar
# ---------------------------------------------------------------------------

class SidebarPanel(ctk.CTkScrollableFrame):
    """
    Left panel containing:
      ① Model loading controls
      ② Sampling mode radio buttons
      ③ Parameter sliders
      ④ Option checkboxes
    """

    def __init__(self, parent, **kwargs):
        super().__init__(parent, width=290, label_text="", **kwargs)
        self.columnconfigure(0, weight=1)
        self._build()

    def _build(self) -> None:
        row = 0

        # ── App title ──────────────────────────────────────────────────────────
        ctk.CTkLabel(self,
                     text="GPT Inference",
                     font=ctk.CTkFont(size=18, weight="bold"),
                     anchor="w").grid(row=row, column=0, sticky="ew",
                                      padx=12, pady=(14, 2))
        row += 1
        ctk.CTkLabel(self, text="Load a checkpoint and start generating",
                     font=ctk.CTkFont(size=11), text_color="gray60",
                     anchor="w").grid(row=row, column=0, sticky="ew",
                                      padx=12, pady=(0, 12))
        row += 1

        # ── Model section ──────────────────────────────────────────────────────
        self._section_label(row, "MODEL"); row += 1

        self._ckpt_entry = ctk.CTkEntry(self, placeholder_text="No checkpoint loaded",
                                         state="readonly", height=30)
        self._ckpt_entry.grid(row=row, column=0, sticky="ew", padx=12, pady=(0, 4))
        row += 1

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.grid(row=row, column=0, sticky="ew", padx=12)
        btn_row.columnconfigure((0, 1), weight=1)
        self.browse_btn = ctk.CTkButton(btn_row, text="Browse…", height=28,
                                         fg_color="#2b4a7a", hover_color="#3a6aaa")
        self.browse_btn.grid(row=0, column=0, sticky="ew", padx=(0, 3))
        self.load_btn   = ctk.CTkButton(btn_row, text="Load Model", height=28,
                                         fg_color="#1a6b3a", hover_color="#2a9b52",
                                         state="disabled")
        self.load_btn.grid(row=0, column=1, sticky="ew", padx=(3, 0))
        row += 1

        self._status_dot  = ctk.CTkLabel(self, text="● Not loaded",
                                          text_color="#ff5555",
                                          font=ctk.CTkFont(size=11),
                                          anchor="w")
        self._status_dot.grid(row=row, column=0, sticky="ew", padx=14, pady=(6, 2))
        row += 1

        self._model_info  = ctk.CTkLabel(self, text="",
                                          text_color="gray60",
                                          font=ctk.CTkFont(size=10),
                                          anchor="w", justify="left",
                                          wraplength=240)
        self._model_info.grid(row=row, column=0, sticky="ew", padx=14, pady=(0, 10))
        row += 1

        # ── Sampling mode ──────────────────────────────────────────────────────
        self._section_label(row, "SAMPLING MODE"); row += 1

        self._mode_var = tk.StringVar(value="top_kp")
        modes = [
            ("Top-K + Top-P  (recommended)", "top_kp"),
            ("Top-K only",                   "top_k"),
            ("Top-P only (nucleus)",          "top_p"),
            ("Greedy  (no sampling)",         "greedy"),
        ]
        for label, val in modes:
            ctk.CTkRadioButton(self, text=label, variable=self._mode_var,
                               value=val, font=ctk.CTkFont(size=11)
                               ).grid(row=row, column=0, sticky="w",
                                      padx=16, pady=1)
            row += 1

        ctk.CTkFrame(self, height=1, fg_color="#333344"
                     ).grid(row=row, column=0, sticky="ew", padx=8, pady=8)
        row += 1

        # ── Parameter sliders ──────────────────────────────────────────────────
        self._section_label(row, "PARAMETERS"); row += 1

        self.s_temp = SliderRow(self, "Temperature",
                                from_=0.01, to=2.0, default=0.80,
                                fmt="{:.2f}", steps=199)
        self.s_temp.grid(row=row, column=0, sticky="ew", padx=12); row += 1

        self.s_topk = SliderRow(self, "Top-K  (0 = off)",
                                from_=0, to=200, default=40,
                                fmt="{:.0f}", steps=200)
        self.s_topk.grid(row=row, column=0, sticky="ew", padx=12); row += 1

        self.s_topp = SliderRow(self, "Top-P  (nucleus)",
                                from_=0.01, to=1.0, default=0.90,
                                fmt="{:.2f}", steps=99)
        self.s_topp.grid(row=row, column=0, sticky="ew", padx=12); row += 1

        self.s_maxtok = SliderRow(self, "Max New Tokens",
                                  from_=10, to=2000, default=300,
                                  fmt="{:.0f}", steps=199)
        self.s_maxtok.grid(row=row, column=0, sticky="ew", padx=12); row += 1

        self.s_rp = SliderRow(self, "Repetition Penalty",
                              from_=1.0, to=2.0, default=1.1,
                              fmt="{:.2f}", steps=100)
        self.s_rp.grid(row=row, column=0, sticky="ew", padx=12); row += 1

        ctk.CTkFrame(self, height=1, fg_color="#333344"
                     ).grid(row=row, column=0, sticky="ew", padx=8, pady=8)
        row += 1

        # ── Options ────────────────────────────────────────────────────────────
        self._section_label(row, "OPTIONS"); row += 1

        self._cot_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(self, text="Chain of Thought  (<|think|>…<|/think|>)",
                        variable=self._cot_var,
                        font=ctk.CTkFont(size=11)
                        ).grid(row=row, column=0, sticky="w", padx=14, pady=2)
        row += 1

        self._show_special_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(self, text="Show special tokens",
                        variable=self._show_special_var,
                        font=ctk.CTkFont(size=11)
                        ).grid(row=row, column=0, sticky="w", padx=14, pady=2)
        row += 1

        self._echo_prompt_var = tk.BooleanVar(value=True)
        ctk.CTkCheckBox(self, text="Echo prompt in output",
                        variable=self._echo_prompt_var,
                        font=ctk.CTkFont(size=11)
                        ).grid(row=row, column=0, sticky="w", padx=14, pady=(2, 16))

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _section_label(self, row: int, text: str) -> None:
        ctk.CTkLabel(self, text=text,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="gray55", anchor="w"
                     ).grid(row=row, column=0, sticky="ew", padx=14, pady=(8, 2))

    def set_ckpt_path(self, path: str) -> None:
        self._ckpt_entry.configure(state="normal")
        self._ckpt_entry.delete(0, "end")
        self._ckpt_entry.insert(0, os.path.basename(path))
        self._ckpt_entry.configure(state="readonly")

    def set_status(self, text: str, color: str = "#ff5555") -> None:
        self._status_dot.configure(text=text, text_color=color)

    def set_model_info(self, info: dict) -> None:
        n = info.get("n_params", 0)
        text = (
            f"  {info.get('n_layer')}L · {info.get('n_embd')}D · {info.get('n_head')}H\n"
            f"  {n/1e6:.1f}M params · block {info.get('block_size')}\n"
            f"  epoch {info.get('epoch')} · step {info.get('step')}\n"
            f"  val loss {info.get('val_loss', 0):.4f} · {info.get('device', '').upper()}"
        )
        self._model_info.configure(text=text)

    # ── Property accessors ─────────────────────────────────────────────────────

    @property
    def temperature(self) -> float:    return self.s_temp.get()
    @property
    def top_k(self) -> int:            return int(self.s_topk.get())
    @property
    def top_p(self) -> float:          return self.s_topp.get()
    @property
    def max_new_tokens(self) -> int:   return int(self.s_maxtok.get())
    @property
    def rep_penalty(self) -> float:    return self.s_rp.get()
    @property
    def sampling_mode(self) -> str:    return self._mode_var.get()
    @property
    def use_cot(self) -> bool:         return self._cot_var.get()
    @property
    def show_special(self) -> bool:    return self._show_special_var.get()
    @property
    def echo_prompt(self) -> bool:     return self._echo_prompt_var.get()

    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_k":       self.top_k,
            "top_p":       self.top_p,
            "max_tokens":  self.max_new_tokens,
            "rep_penalty": self.rep_penalty,
            "mode":        self.sampling_mode,
            "cot":         self.use_cot,
            "echo":        self.echo_prompt,
        }

    def from_dict(self, d: dict) -> None:
        if "temperature" in d: self.s_temp.set(d["temperature"])
        if "top_k"       in d: self.s_topk.set(d["top_k"])
        if "top_p"       in d: self.s_topp.set(d["top_p"])
        if "max_tokens"  in d: self.s_maxtok.set(d["max_tokens"])
        if "rep_penalty" in d: self.s_rp.set(d["rep_penalty"])
        if "mode"        in d: self._mode_var.set(d["mode"])
        if "cot"         in d: self._cot_var.set(d["cot"])
        if "echo"        in d: self._echo_prompt_var.set(d["echo"])


# ---------------------------------------------------------------------------
# Main (right) panel
# ---------------------------------------------------------------------------

class MainPanel(ctk.CTkFrame):
    """
    Right panel containing the output display, prompt input, action buttons,
    and the status bar.
    """

    # Token tag → (text-tag-name, is_special)
    _THINK_OPEN   = "<|think|>"
    _THINK_CLOSE  = "<|/think|>"
    _SPECIAL_TOKS = {
        "<|endoftext|>", "<|sep|>",
        "<|think|>", "<|/think|>",
        "<|system|>", "<|/system|>",
        "<|user|>", "<|/user|>", "<|assistant|>",
    }

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color="#0f0f1a", **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._in_think_block = False
        self._build()

    def _build(self) -> None:
        # ── Output area ────────────────────────────────────────────────────────
        out_frame = ctk.CTkFrame(self, fg_color="#141428", corner_radius=8)
        out_frame.grid(row=0, column=0, sticky="nsew", padx=12, pady=(12, 4))
        out_frame.columnconfigure(0, weight=1)
        out_frame.rowconfigure(1, weight=1)

        ctk.CTkLabel(out_frame, text="OUTPUT",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="gray55", anchor="w"
                     ).grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))

        # Use underlying tk.Text for full tag control
        self._out_text = tk.Text(
            out_frame,
            wrap="word",
            state="disabled",
            font=_MONO,
            bg="#0d0d1f",
            fg="#e8e8e8",
            insertbackground="#e8e8e8",
            selectbackground="#2a4a8a",
            selectforeground="#ffffff",
            relief="flat",
            padx=10, pady=8,
            cursor="arrow",
        )
        self._out_text.grid(row=1, column=0, sticky="nsew", padx=4, pady=4)

        out_scroll = ctk.CTkScrollbar(out_frame, command=self._out_text.yview)
        out_scroll.grid(row=1, column=1, sticky="ns", padx=(0, 4), pady=4)
        self._out_text.configure(yscrollcommand=out_scroll.set)

        # Configure text tags
        for tag, cfg in _TAGS.items():
            self._out_text.tag_configure(tag, **cfg)

        # ── Prompt input ───────────────────────────────────────────────────────
        prompt_frame = ctk.CTkFrame(self, fg_color="#141428", corner_radius=8)
        prompt_frame.grid(row=1, column=0, sticky="ew", padx=12, pady=4)
        prompt_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(prompt_frame, text="PROMPT",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="gray55", anchor="w"
                     ).grid(row=0, column=0, sticky="w", padx=10, pady=(6, 0))

        self._prompt_text = tk.Text(
            prompt_frame,
            height=4,
            wrap="word",
            font=_MONO,
            bg="#0d0d1f",
            fg="#c8d8f8",
            insertbackground="#c8d8f8",
            selectbackground="#2a4a8a",
            selectforeground="#ffffff",
            relief="flat",
            padx=10, pady=8,
        )
        self._prompt_text.grid(row=1, column=0, columnspan=2,
                               sticky="ew", padx=4, pady=(0, 4))
        # Ctrl+Enter submits
        self._prompt_text.bind("<Control-Return>", lambda _e: self._on_generate())
        self._prompt_text.bind("<Shift-Return>",   lambda _e: self._on_generate())

        hint = ctk.CTkLabel(prompt_frame,
                            text="Ctrl+Enter or Shift+Enter to generate",
                            text_color="gray45", font=ctk.CTkFont(size=10))
        hint.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 6))

        # ── Action buttons ─────────────────────────────────────────────────────
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=4)

        self.generate_btn = ctk.CTkButton(
            btn_frame, text="▶  Generate",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38, width=150,
            fg_color="#1a6b3a", hover_color="#2a9b52",
            state="disabled",
            command=self._on_generate,
        )
        self.generate_btn.pack(side="left", padx=(0, 6))

        self.stop_btn = ctk.CTkButton(
            btn_frame, text="■  Stop",
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38, width=110,
            fg_color="#6b1a1a", hover_color="#9b2a2a",
            state="disabled",
            command=self._on_stop,
        )
        self.stop_btn.pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_frame, text="Clear",
            height=38, width=80,
            fg_color="#2a2a3e", hover_color="#3a3a5e",
            command=self._on_clear,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_frame, text="Copy",
            height=38, width=80,
            fg_color="#2a2a3e", hover_color="#3a3a5e",
            command=self._on_copy,
        ).pack(side="left")

        # ── Status bar ─────────────────────────────────────────────────────────
        status_bar = ctk.CTkFrame(self, height=28, fg_color="#0a0a14",
                                   corner_radius=0)
        status_bar.grid(row=3, column=0, sticky="ew", padx=0, pady=(4, 0))
        status_bar.columnconfigure(0, weight=1)

        self._status_var = tk.StringVar(value="  Ready — load a model to begin")
        self._status_lbl = ctk.CTkLabel(
            status_bar,
            textvariable=self._status_var,
            font=ctk.CTkFont(size=11),
            text_color="gray60",
            anchor="w",
        )
        self._status_lbl.grid(row=0, column=0, sticky="ew", padx=10)

    # ── Callbacks (wired up by GPTInferenceApp) ────────────────────────────────

    def _on_generate(self) -> None:
        """Placeholder — overridden by GPTInferenceApp."""

    def _on_stop(self) -> None:
        """Placeholder — overridden by GPTInferenceApp."""

    def _on_clear(self) -> None:
        self._out_text.configure(state="normal")
        self._out_text.delete("1.0", "end")
        self._out_text.configure(state="disabled")
        self._in_think_block = False
        self._status_var.set("  Output cleared")

    def _on_copy(self) -> None:
        text = self._out_text.get("1.0", "end").strip()
        self.clipboard_clear()
        self.clipboard_append(text)
        self._status_var.set("  Output copied to clipboard")

    # ── Output writing ─────────────────────────────────────────────────────────

    def append_output(self, text: str, tag: str = "answer",
                      show_special: bool = True) -> None:
        """Thread-safe output append with tag-based colouring."""
        if not text:
            return
        # Classify special tokens
        if text.strip() in self._SPECIAL_TOKS:
            if not show_special:
                return
            tag = "special"
            # Track think block state
            if text.strip() == self._THINK_OPEN:
                self._in_think_block = True
            elif text.strip() == self._THINK_CLOSE:
                self._in_think_block = False

        # Inside think block → use think style
        if self._in_think_block and tag not in ("special",):
            tag = "think"

        self._out_text.configure(state="normal")
        self._out_text.insert("end", text, tag)
        self._out_text.see("end")
        self._out_text.configure(state="disabled")

    def append_system(self, text: str) -> None:
        self._out_text.configure(state="normal")
        self._out_text.insert("end", text, "system")
        self._out_text.see("end")
        self._out_text.configure(state="disabled")

    def set_status(self, text: str, color: str = "gray60") -> None:
        self._status_var.set(f"  {text}")
        self._status_lbl.configure(text_color=color)

    def get_prompt(self) -> str:
        return self._prompt_text.get("1.0", "end").strip()

    def set_generate_state(self, generating: bool) -> None:
        if generating:
            self.generate_btn.configure(state="disabled")
            self.stop_btn.configure(state="normal")
        else:
            self.generate_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")

    def enable_generate(self) -> None:
        self.generate_btn.configure(state="normal")


# ---------------------------------------------------------------------------
# System Prompt Panel  (collapsible footer)
# ---------------------------------------------------------------------------

class SystemPromptPanel(ctk.CTkFrame):
    """
    Collapsible panel for editing the system prompt.

    When collapsed shows a single header row.
    When expanded shows a text editor + Save / Reset buttons.

    Usage
    -----
        panel = SystemPromptPanel(parent, on_save=callback)
        panel.set_text("You are a helpful assistant.")
        text = panel.get_text()
    """

    _DEFAULT_HEIGHT = 110   # editor height when expanded

    def __init__(self, parent, on_save=None, **kwargs):
        super().__init__(parent, fg_color="#0c0c1e", corner_radius=6, **kwargs)
        self.columnconfigure(0, weight=1)
        self._on_save_cb = on_save
        self._expanded   = tk.BooleanVar(value=False)
        self._build()

    def _build(self) -> None:
        # ── Header row (always visible) ───────────────────────────────────────
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 4))
        hdr.columnconfigure(1, weight=1)

        self._toggle_btn = ctk.CTkButton(
            hdr,
            text="▶  SYSTEM PROMPT",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray55",
            fg_color="transparent",
            hover_color="#1a1a2e",
            anchor="w",
            width=170,
            height=24,
            command=self._toggle,
        )
        self._toggle_btn.grid(row=0, column=0, sticky="w")

        self._hint_lbl = ctk.CTkLabel(
            hdr,
            text="(click to expand — rules the model follows on every generation)",
            font=ctk.CTkFont(size=10),
            text_color="gray40",
            anchor="w",
        )
        self._hint_lbl.grid(row=0, column=1, sticky="w", padx=(6, 0))

        # ── Collapsible body ──────────────────────────────────────────────────
        self._body = ctk.CTkFrame(self, fg_color="transparent")
        # Not gridded until expanded

        self._editor = ctk.CTkTextbox(
            self._body,
            height=self._DEFAULT_HEIGHT,
            font=ctk.CTkFont(family="Courier New" if sys.platform == "win32" else "Courier",
                             size=11),
            fg_color="#0d0d1f",
            text_color="#c8c8e8",
            wrap="word",
            border_width=1,
            border_color="#333355",
        )
        self._editor.pack(fill="both", expand=True, padx=4, pady=(0, 4))

        btn_row = ctk.CTkFrame(self._body, fg_color="transparent")
        btn_row.pack(fill="x", padx=4, pady=(0, 6))

        ctk.CTkButton(
            btn_row,
            text="Save",
            width=80, height=28,
            fg_color="#1a6b3a", hover_color="#2a9b52",
            command=self._on_save,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_row,
            text="Reset to Default",
            width=130, height=28,
            fg_color="#2a2a3e", hover_color="#3a3a5e",
            command=self._on_reset,
        ).pack(side="left", padx=(0, 6))

        self._saved_lbl = ctk.CTkLabel(
            btn_row,
            text="",
            font=ctk.CTkFont(size=10),
            text_color="#44ff88",
            anchor="w",
        )
        self._saved_lbl.pack(side="left", padx=4)

    # ── Toggle ──────────────────────────────────────────────────────────────────

    def _toggle(self) -> None:
        if self._expanded.get():
            self._body.grid_forget()
            self._toggle_btn.configure(text="▶  SYSTEM PROMPT")
            self._hint_lbl.configure(
                text="(click to expand — rules the model follows on every generation)"
            )
            self._expanded.set(False)
        else:
            self._body.grid(row=1, column=0, sticky="ew", padx=4, pady=(0, 4))
            self._toggle_btn.configure(text="▼  SYSTEM PROMPT")
            self._hint_lbl.configure(text="")
            self._expanded.set(True)

    # ── Callbacks ───────────────────────────────────────────────────────────────

    def _on_save(self) -> None:
        if self._on_save_cb:
            self._on_save_cb(self.get_text())
        self._saved_lbl.configure(text="✓ Saved")
        self.after(2000, lambda: self._saved_lbl.configure(text=""))

    def _on_reset(self) -> None:
        _default = _load_default_system_prompt()
        self.set_text(_default)
        self._on_save()

    # ── Public API ──────────────────────────────────────────────────────────────

    def get_text(self) -> str:
        return self._editor.get("1.0", "end").strip()

    def set_text(self, text: str) -> None:
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", text)


def _load_default_system_prompt() -> str:
    """Return contents of system_prompt.txt, or a built-in default."""
    try:
        with open(_SYS_PROMPT_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return (
            "You are a helpful, harmless, and honest AI assistant. "
            "You provide clear, accurate, and thoughtful responses. "
            "You do not generate harmful or deceptive content."
        )


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class GPTInferenceApp(ctk.CTk):
    """
    Top-level application window.  Wires together the sidebar, main panel,
    inference engine, and background threads.

    Architecture
    ------------
    Generation runs in a daemon thread.  Each yielded (token, done, stats)
    tuple is posted to self._queue.  The main thread polls the queue every
    25 ms via Tk's after() mechanism and appends tokens to the output widget.
    This keeps the UI responsive throughout generation.
    """

    _POLL_MS = 25    # queue poll interval in milliseconds

    def __init__(self, auto_ckpt: str | None = None) -> None:
        super().__init__()

        self.title("GPT Model Inference")
        self.geometry("1280x820")
        self.minsize(900, 600)

        # State
        self._engine      = InferenceEngine()
        self._queue:  queue.Queue = queue.Queue()
        self._stop_ev: threading.Event = threading.Event()
        self._generating: bool = False
        self._pending_ckpt: str | None = auto_ckpt

        self._build_ui()
        self._load_settings()
        self._start_polling()

        # Auto-load checkpoint passed on the command line
        if auto_ckpt:
            self._sidebar.set_ckpt_path(auto_ckpt)
            self._sidebar.load_btn.configure(state="normal")
            self.after(300, self._load_model)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)  # system prompt panel row

        # Sidebar
        self._sidebar = SidebarPanel(self)
        self._sidebar.grid(row=0, column=0, sticky="nsew", padx=(8, 4), pady=(8, 4))
        self._sidebar.browse_btn.configure(command=self._browse_ckpt)
        self._sidebar.load_btn.configure(command=self._load_model)

        # Main panel
        self._main = MainPanel(self)
        self._main.grid(row=0, column=1, sticky="nsew", padx=(4, 8), pady=(8, 4))

        # System prompt panel (spans both columns)
        self._sysprompt = SystemPromptPanel(self, on_save=self._on_system_prompt_save)
        self._sysprompt.grid(row=1, column=0, columnspan=2, sticky="ew",
                             padx=8, pady=(0, 8))
        # Load from file on startup
        self._sysprompt.set_text(_load_default_system_prompt())

        # Wire generate / stop callbacks
        self._main._on_generate = self._start_generation
        self._main._on_stop     = self._stop_generation

    # ── Settings persistence ────────────────────────────────────────────────────

    def _load_settings(self) -> None:
        s = _load_settings()
        if "params" in s:
            self._sidebar.from_dict(s["params"])
        if "ckpt_path" in s and os.path.isfile(s["ckpt_path"]):
            self._pending_ckpt = self._pending_ckpt or s["ckpt_path"]
            self._sidebar.set_ckpt_path(s["ckpt_path"])
            self._sidebar.load_btn.configure(state="normal")
        # Restore saved system prompt (falls back to file or built-in default)
        sys_prompt = s.get("system_prompt") or _load_default_system_prompt()
        self._sysprompt.set_text(sys_prompt)
        self._engine.set_system_prompt(sys_prompt)

    # ── System prompt ────────────────────────────────────────────────────────────

    def _on_system_prompt_save(self, text: str) -> None:
        """Called when the user hits Save in the system prompt panel."""
        self._engine.set_system_prompt(text)
        # Persist to disk
        try:
            with open(_SYS_PROMPT_FILE, "w", encoding="utf-8") as f:
                f.write(text + "\n")
        except OSError:
            pass

    def _save_settings(self) -> None:
        _save_settings({
            "params":        self._sidebar.to_dict(),
            "ckpt_path":     getattr(self._engine, "ckpt_meta", {}).get("path", ""),
            "system_prompt": self._sysprompt.get_text(),
        })

    def on_closing(self) -> None:
        self._save_settings()
        self.destroy()

    # ── Checkpoint browsing ─────────────────────────────────────────────────────

    def _browse_ckpt(self) -> None:
        initial = _CKPT_DIR if os.path.isdir(_CKPT_DIR) else os.path.expanduser("~")
        path = filedialog.askopenfilename(
            title="Select model checkpoint",
            initialdir=initial,
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
        )
        if path:
            self._pending_ckpt = path
            self._sidebar.set_ckpt_path(path)
            self._sidebar.load_btn.configure(state="normal")

    # ── Model loading ───────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        path = self._pending_ckpt
        if not path or not os.path.isfile(path):
            self._sidebar.set_status("● File not found", "#ff5555")
            return

        self._sidebar.load_btn.configure(state="disabled")
        self._sidebar.set_status("● Loading…", "#ffaa44")
        self._main.append_system(f"\nLoading checkpoint: {os.path.basename(path)}\n")

        def _worker():
            try:
                info = self._engine.load_model(path)
                self._queue.put(("model_loaded", info))
            except Exception as exc:
                self._queue.put(("model_error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_model_loaded(self, info: dict) -> None:
        self._sidebar.set_status("● Model loaded", "#44ff88")
        self._sidebar.set_model_info(info)
        self._main.enable_generate()
        self._main.append_system(
            f"Loaded {info['n_params']/1e6:.1f}M-param model on {info['device'].upper()}\n"
            f"Layers {info['n_layer']} · Hidden {info['n_embd']} · Heads {info['n_head']}\n"
            f"Context {info['block_size']} tokens · Val loss {info.get('val_loss',0):.4f}\n"
            "─" * 50 + "\n"
        )
        self._save_settings()

    # ── Generation ─────────────────────────────────────────────────────────────

    def _start_generation(self) -> None:
        if self._generating:
            return
        prompt = self._main.get_prompt()
        if not prompt:
            self._main.set_status("Enter a prompt first", "#ffaa44")
            return
        if not self._engine.loaded:
            self._main.set_status("Load a model first", "#ff5555")
            return

        self._generating = True
        self._stop_ev.clear()
        self._main.set_generate_state(True)

        # Sync system prompt to engine (picks up any unsaved edits)
        self._engine.set_system_prompt(self._sysprompt.get_text())

        # Echo prompt in output
        if self._sidebar.echo_prompt:
            self._main.append_output("\n")
            self._main.append_output(prompt, tag="prompt")
            self._main.append_output("\n", tag="prompt")
            self._main._in_think_block = False

        # Spin up generation thread
        gen_kwargs = {
            "prompt":             prompt,
            "max_new_tokens":     self._sidebar.max_new_tokens,
            "temperature":        self._sidebar.temperature,
            "top_k":              self._sidebar.top_k,
            "top_p":              self._sidebar.top_p,
            "repetition_penalty": self._sidebar.rep_penalty,
            "sampling_mode":      self._sidebar.sampling_mode,
            "use_cot":            self._sidebar.use_cot,
            "stop_event":         self._stop_ev,
        }

        def _worker():
            try:
                for tok, done, stats in self._engine.generate_stream(**gen_kwargs):
                    self._queue.put(("token", tok, done, stats))
                    if done:
                        break
            except Exception as exc:
                self._queue.put(("gen_error", str(exc)))

        threading.Thread(target=_worker, daemon=True).start()

    def _stop_generation(self) -> None:
        self._stop_ev.set()

    # ── Queue polling ───────────────────────────────────────────────────────────

    def _start_polling(self) -> None:
        self.after(self._POLL_MS, self._poll_queue)

    def _poll_queue(self) -> None:
        try:
            while True:
                item = self._queue.get_nowait()
                self._handle_queue_item(item)
        except queue.Empty:
            pass
        self.after(self._POLL_MS, self._poll_queue)

    def _handle_queue_item(self, item: tuple) -> None:
        kind = item[0]

        if kind == "model_loaded":
            self._on_model_loaded(item[1])
            self._sidebar.load_btn.configure(state="normal")

        elif kind == "model_error":
            self._sidebar.set_status(f"● Error: {item[1][:40]}", "#ff5555")
            self._main.append_system(f"\nLoad error: {item[1]}\n")
            self._sidebar.load_btn.configure(state="normal")

        elif kind == "token":
            _, tok_text, done, stats = item
            if tok_text:
                self._main.append_output(
                    tok_text,
                    tag="answer",
                    show_special=self._sidebar.show_special,
                )
            self._update_status(stats)
            if done:
                self._on_generation_done(stats)

        elif kind == "gen_error":
            self._main.append_system(f"\nGeneration error: {item[1]}\n")
            self._generating = False
            self._main.set_generate_state(False)

    def _update_status(self, stats: GenStats) -> None:
        mem  = _gpu_mem_str()
        text = (
            f"Tokens: {stats.n_tokens}  ·  "
            f"{stats.tok_per_sec:.1f} tok/s  ·  "
            f"{stats.elapsed_s:.1f}s  ·  "
            f"{stats.status}"
            f"{mem}"
        )
        self._main.set_status(text, "#44aa88")

    def _on_generation_done(self, stats: GenStats) -> None:
        self._main.append_output("\n")
        self._generating = False
        self._main.set_generate_state(False)
        mem = _gpu_mem_str()
        self._main.set_status(
            f"Done — {stats.n_tokens} tokens  ·  "
            f"{stats.tok_per_sec:.1f} tok/s  ·  "
            f"{stats.elapsed_s:.1f}s  ·  {stats.status}{mem}",
            color="#44ff88",
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPT Inference GUI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ckpt", default=None,
        help="Path to checkpoint to load automatically on startup",
    )
    # Auto-discover best.pt if --ckpt not given
    auto = None
    args, _ = parser.parse_known_args()
    if args.ckpt and os.path.isfile(args.ckpt):
        auto = args.ckpt
    elif os.path.isfile(os.path.join(_CKPT_DIR, "best.pt")):
        auto = os.path.join(_CKPT_DIR, "best.pt")

    app = GPTInferenceApp(auto_ckpt=auto)
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
