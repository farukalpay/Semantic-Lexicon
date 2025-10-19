# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Clipboard access utilities."""

from __future__ import annotations

import platform
import shutil
import subprocess
from importlib import import_module
from typing import Any, Optional


class ClipboardError(RuntimeError):
    """Raised when clipboard text cannot be retrieved."""


def _load_pyperclip() -> Any:
    """Attempt to import pyperclip dynamically."""

    try:
        return import_module("pyperclip")
    except ImportError:
        return None


def _read_with_pure_python() -> Optional[str]:
    """Attempt to read the clipboard using pure Python backends."""

    # Prefer third-party helpers if available; fall back to tkinter.
    pyperclip = _load_pyperclip()
    if pyperclip is not None:
        try:
            text = pyperclip.paste()
        except Exception:  # pragma: no cover - best effort fallback
            text = ""
        if text is not None:
            return str(text)

    try:
        import tkinter
    except ImportError:
        return None

    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        return None

    root.withdraw()
    try:
        return root.clipboard_get()
    except tkinter.TclError:
        return ""
    finally:
        root.destroy()


def _read_with_command(command: list[str]) -> Optional[str]:
    """Execute a platform command and return stdout as text."""

    if shutil.which(command[0]) is None:
        return None

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf8",
        )
    except (subprocess.SubprocessError, OSError):
        return None

    return completed.stdout


def _read_with_platform_command() -> Optional[str]:
    """Dispatch to an OS-appropriate clipboard command."""

    system = platform.system().lower()

    if system == "darwin":
        return _read_with_command(["pbpaste"])

    if system == "windows":
        return _read_with_command(["powershell", "-NoProfile", "-Command", "Get-Clipboard"])

    # Assume POSIX; try known clipboard tools.
    for candidate in (
        ["wl-paste", "--no-newline"],
        ["xclip", "-selection", "clipboard", "-o"],
        ["xsel", "--clipboard", "--output"],
    ):
        text = _read_with_command(candidate)
        if text is not None:
            return text

    return None


def get_clipboard_text() -> str:
    """Return the current clipboard text or raise ``ClipboardError``."""

    text = _read_with_pure_python()
    if text is None:
        text = _read_with_platform_command()
    if text is None:
        raise ClipboardError("No clipboard reader is available on this platform.")

    text = str(text)
    if text == "":
        raise ClipboardError("Clipboard is empty.")

    return text
