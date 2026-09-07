"""A one-line text field driven by OpenCV ``waitKey`` codes.

Used for naming a sign or typing a phrase for the reverse channel without
leaving the video window.
"""

from __future__ import annotations

from dataclasses import dataclass

from hud import FONT, WHITE, shade
import cv2

_PRINTABLE = set(range(32, 127))
ENTER = {13, 10}
BACKSPACE = {8, 127}
ESC = 27


@dataclass
class TextPrompt:
    label: str = ""
    value: str = ""
    active: bool = False

    def start(self, label: str, initial: str = "") -> None:
        self.label, self.value, self.active = label, initial, True

    def key(self, code: int) -> tuple[str, str | None]:
        """Return ('typing'|'submit'|'cancel', value_or_None)."""
        if not self.active:
            return ("typing", None)
        if code in ENTER:
            self.active = False
            return ("submit", self.value.strip())
        if code == ESC:
            self.active = False
            return ("cancel", None)
        if code in BACKSPACE:
            self.value = self.value[:-1]
        elif code in _PRINTABLE:
            self.value += chr(code)
        return ("typing", None)

    def draw(self, frame) -> None:
        if not self.active:
            return
        h, w = frame.shape[:2]
        shade(frame, 0, h // 2 - 44, w, h // 2 + 44, 0.75)
        cv2.putText(frame, f"{self.label}: {self.value}_", (24, h // 2 + 8),
                    FONT, 1.0, WHITE, 2, cv2.LINE_AA)
        cv2.putText(frame, "enter = ok   esc = cancel", (24, h // 2 + 34),
                    FONT, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
