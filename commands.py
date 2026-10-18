"""Sign -> action bindings for Control mode.

``data/commands.json`` maps a sign name (as stored in the sign bank) to an
action:

    {
      "wave hello":  {"type": "text",  "value": "hey!"},
      "flat spin":   {"type": "app",   "value": "Safari"},
      "double tap":  {"type": "key",   "value": "space"},
      "twist":       {"type": "shell", "value": "echo done"}
    }

``type:shell`` runs a local shell command - only ever your own, typed by you.
It runs through your OS's default shell, so the command itself needs to be
one that shell understands (e.g. ``say`` only exists on macOS, ``notify-send``
only on Linux, etc.) - that's on you to pick per-platform, same as typing it
yourself.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import output_sink

COMMANDS_PATH = Path(__file__).parent / "data" / "commands.json"

# aliases -> the key name pyautogui expects (pyautogui maps these to the
# right OS key code itself, so this table only needs to cover spelling
# differences, not per-platform codes).
_KEY_ALIASES = {
    "delete": "backspace", "playpause": "space",
}
_VALID_KEYS = {
    "space", "return", "enter", "tab", "esc", "backspace",
    "left", "right", "down", "up", "f5",
}


def load_commands(path: str | Path = COMMANDS_PATH) -> dict[str, dict]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_commands(commands: dict[str, dict], path: str | Path = COMMANDS_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(commands, indent=2) + "\n")


def parse_action(spec: str) -> dict | None:
    """'app:Safari' / 'key:space' / 'text:hi there' / 'shell:say hi' -> dict."""
    if ":" not in spec:
        return None
    kind, _, value = spec.partition(":")
    kind, value = kind.strip().lower(), value.strip()
    if kind not in ("app", "key", "text", "shell") or not value:
        return None
    return {"type": kind, "value": value}


def run_action(action: dict) -> str:
    kind, value = action.get("type"), action.get("value", "")
    if kind == "app":
        output_sink.open_app(value)
    elif kind == "text":
        output_sink.type_text(value + " ")
    elif kind == "key":
        name = value.lower()
        name = _KEY_ALIASES.get(name, name)
        if name in _VALID_KEYS:
            output_sink.press_key(name)
    elif kind == "shell":
        try:
            subprocess.Popen(value, shell=True)
        except Exception:
            return f"shell failed: {value}"
    return f"{kind}: {value}"
