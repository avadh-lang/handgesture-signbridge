"""Get recognised text off the screen and into the world.

Cross-platform (macOS / Windows / Linux):

* ``speak`` - say it aloud via ``pyttsx3`` (NSSpeechSynthesizer on macOS, SAPI5
  on Windows, ``espeak``/``espeak-ng`` on Linux - install one of those with
  your package manager if voice output stays silent on Linux).
* ``type_text`` - send the characters as keystrokes to the frontmost app via
  ``pyautogui``, so you can sign into Slack, Notes, a search bar, anything.
  Needs Accessibility permission on macOS (System Settings -> Privacy &
  Security -> Accessibility). On Linux this only works under X11 - Wayland
  blocks synthetic keystrokes/clicks from other apps by design.
* ``press_key`` - press a single named key (``"space"``, ``"enter"``, ``"tab"``,
  ``"esc"``, ``"backspace"``, ``"left"``/``"right"``/``"up"``/``"down"``, ``"f5"``, ...)
  via ``pyautogui``, which maps the name to the right OS key code itself.
* ``open_app`` - best-effort "open this app by name": ``open -a`` on macOS,
  ``os.startfile`` on Windows, a direct command launch on Linux. App naming
  conventions differ a lot across OSes, so this is best-effort, not guaranteed.

Everything is best-effort and swallows errors - a translator that crashes
because the TTS engine is busy is worse than one that drops a phrase.
"""

from __future__ import annotations

import subprocess
import sys
import threading

_PLATFORM = sys.platform  # "darwin" | "win32" | "linux" | ...

_tts_lock = threading.Lock()
_tts_engine = None
_tts_failed = False


def _get_tts_engine():
    global _tts_engine, _tts_failed
    if _tts_engine is not None or _tts_failed:
        return _tts_engine
    try:
        import pyttsx3
        _tts_engine = pyttsx3.init()
    except Exception:
        _tts_failed = True
        _tts_engine = None
    return _tts_engine


def _speak_now(text: str, voice: str | None) -> None:
    with _tts_lock:
        engine = _get_tts_engine()
        if engine is None:
            return
        try:
            if voice:
                for v in engine.getProperty("voices"):
                    if voice.lower() in (v.name or "").lower() or voice == v.id:
                        engine.setProperty("voice", v.id)
                        break
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass


def speak(text: str, voice: str | None = None, blocking: bool = False) -> None:
    text = (text or "").strip()
    if not text:
        return
    if blocking:
        _speak_now(text, voice)
    else:
        threading.Thread(target=_speak_now, args=(text, voice), daemon=True).start()


def _pyautogui():
    try:
        import pyautogui
        return pyautogui
    except Exception:
        return None


def type_text(text: str, newline: bool = False) -> bool:
    """Type ``text`` into the frontmost application. Returns False if unavailable."""
    if not text:
        return False
    pag = _pyautogui()
    if pag is None:
        return False
    try:
        pag.write(text, _pause=False)
        if newline:
            pag.press("enter", _pause=False)
    except Exception:
        return False
    return True


def press_key(key: str, modifiers: list[str] | None = None) -> None:
    """Press a named key (pyautogui names: 'space', 'enter', 'tab', 'esc',
    'backspace', 'left'/'right'/'up'/'down', 'f5', ...), optionally with
    held modifiers (e.g. ['ctrl'] / ['command'] / ['shift'])."""
    pag = _pyautogui()
    if pag is None:
        return
    try:
        if modifiers:
            pag.hotkey(*modifiers, key, _pause=False)
        else:
            pag.press(key, _pause=False)
    except Exception:
        pass


def open_app(name: str) -> None:
    try:
        if _PLATFORM == "darwin":
            subprocess.run(["open", "-a", name], check=False, capture_output=True, timeout=5)
        elif _PLATFORM == "win32":
            import os
            os.startfile(name)  # type: ignore[attr-defined]
        else:  # linux and other unix-likes: best-effort direct launch
            subprocess.Popen([name.lower()], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _tts_available() -> bool:
    return _get_tts_engine() is not None


AVAILABLE = {"speak": _tts_available(), "type": _pyautogui() is not None}
