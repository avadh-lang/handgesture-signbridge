"""Get recognised text off the screen and into the world (macOS).

* ``speak`` - say it aloud with the built-in ``say`` voice.
* ``type_text`` - send the characters as keystrokes to the frontmost app, so you
  can sign into Slack, Notes, a search bar, anything.  Needs Accessibility
  permission for the terminal (System Settings -> Privacy & Security ->
  Accessibility).  Off by default; the caller opts in.

Everything is best-effort and swallows errors - a translator that crashes
because ``say`` is busy is worse than one that drops a phrase.
"""

from __future__ import annotations

import shutil
import subprocess
import threading

_HAS_OSASCRIPT = shutil.which("osascript") is not None
_HAS_SAY = shutil.which("say") is not None


def _run(args: list[str]) -> None:
    try:
        subprocess.run(args, check=False, capture_output=True, timeout=5)
    except Exception:
        pass


def speak(text: str, voice: str | None = None, blocking: bool = False) -> None:
    text = (text or "").strip()
    if not text or not _HAS_SAY:
        return
    args = ["say"] + (["-v", voice] if voice else []) + [text]
    if blocking:
        _run(args)
    else:
        threading.Thread(target=_run, args=(args,), daemon=True).start()


def _applescript_escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"')


def type_text(text: str, newline: bool = False) -> bool:
    """Type ``text`` into the frontmost application. Returns False if unavailable."""
    if not text or not _HAS_OSASCRIPT:
        return False
    script = f'tell application "System Events" to keystroke "{_applescript_escape(text)}"'
    _run(["osascript", "-e", script])
    if newline:
        _run(["osascript", "-e",
              'tell application "System Events" to key code 36'])   # return
    return True


def press_key(key_code: int, modifiers: list[str] | None = None) -> None:
    mods = ""
    if modifiers:
        mods = " using {" + ", ".join(f"{m} down" for m in modifiers) + "}"
    _run(["osascript", "-e",
          f'tell application "System Events" to key code {key_code}{mods}'])


def open_app(name: str) -> None:
    _run(["open", "-a", name])


AVAILABLE = {"speak": _HAS_SAY, "type": _HAS_OSASCRIPT}
