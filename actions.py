"""Map stable gestures to actions.

This is deliberately side-effect-light: by default every action just prints a
line. Uncomment / edit ``_SYSTEM_ACTIONS`` to drive real macOS media keys via
AppleScript, or wire the callbacks to anything you like.

An action fires once on the *transition* into a gesture (edge triggered), never
every frame, and only after the gesture has been held long enough to count.
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable

HOLD_SECONDS = 0.4          # gesture must persist this long before it counts
COOLDOWN_SECONDS = 1.0      # min gap between two fires of the same gesture


def _osascript(cmd: str) -> None:
    try:
        subprocess.run(["osascript", "-e", cmd], check=False,
                       capture_output=True, timeout=2)
    except Exception:
        pass


# gesture label -> (description, callback)
_ACTIONS: dict[str, tuple[str, Callable[[], None]]] = {
    "Thumbs up":        ("volume up",   lambda: _osascript(
        "set volume output volume ((output volume of (get volume settings)) + 8)")),
    "Thumbs down":      ("volume down", lambda: _osascript(
        "set volume output volume ((output volume of (get volume settings)) - 8)")),
    "Open palm":        ("play / pause", lambda: _osascript(
        'tell application "System Events" to key code 49')),  # space
    "Fist":             ("mute toggle", lambda: _osascript(
        "set volume output muted (not (output muted of (get volume settings)))")),
    "Peace / Victory":  ("screenshot", lambda: _osascript(
        'tell application "System Events" to keystroke "4" using {command down, shift down}')),
}


class ActionMapper:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.last_action: str | None = None
        self._current: dict[str, str] = {}      # hand -> gesture being held
        self._since: dict[str, float] = {}      # hand -> when the hold started
        self._fired_at: dict[str, float] = {}   # gesture -> last fire time

    def feed(self, hand: str, gesture: str) -> None:
        now = time.time()
        if self._current.get(hand) != gesture:
            self._current[hand] = gesture
            self._since[hand] = now
            return
        if not self.enabled:
            return
        if now - self._since[hand] < HOLD_SECONDS:
            return
        entry = _ACTIONS.get(gesture)
        if not entry:
            return
        if now - self._fired_at.get(gesture, 0.0) < COOLDOWN_SECONDS:
            return
        desc, fn = entry
        self._fired_at[gesture] = now
        self._since[hand] = now + 10 * COOLDOWN_SECONDS  # require re-entry
        self.last_action = desc
        print(f"[action] {gesture} -> {desc}")
        fn()

    def feed_all_idle(self) -> None:
        self._current.clear()
        self._since.clear()
