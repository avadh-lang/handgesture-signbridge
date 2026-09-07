"""Gesture recognition from MediaPipe hand landmarks.

Landmark index reference (MediaPipe Hands, 21 points):
    0  wrist
    1-4   thumb  : CMC, MCP, IP, TIP
    5-8   index  : MCP, PIP, DIP, TIP
    9-12  middle : MCP, PIP, DIP, TIP
    13-16 ring   : MCP, PIP, DIP, TIP
    17-20 pinky  : MCP, PIP, DIP, TIP

All landmark coordinates are normalised to [0, 1] with the origin at the
top-left of the image, so the y axis points *down*.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")


@dataclass
class HandState:
    """Everything the classifier derives about one hand."""

    handedness: str          # "Left" or "Right" (as seen in a mirrored selfie view)
    fingers: tuple[int, int, int, int, int]   # 1 = extended, 0 = curled
    finger_count: int
    gesture: str             # human readable label
    emoji: str


def _dist(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _finger_extended(lm, tip: int, pip: int) -> bool:
    """A non-thumb finger is extended when its tip sits further from the
    wrist than its PIP joint does (works regardless of hand rotation)."""
    return _dist(lm[tip], lm[WRIST]) > _dist(lm[pip], lm[WRIST]) * 1.05


def _thumb_extended(lm, handedness: str) -> bool:
    """The thumb is extended when the tip is pulled away from the hand's
    centre line, i.e. further from the index-finger MCP than the IP joint."""
    hand_span = _dist(lm[INDEX_MCP], lm[PINKY_MCP]) + 1e-6
    tip_gap = _dist(lm[THUMB_TIP], lm[INDEX_MCP])
    ip_gap = _dist(lm[THUMB_IP], lm[INDEX_MCP])
    return tip_gap > ip_gap and tip_gap > hand_span * 0.7


def _finger_vector_up(lm, tip: int, mcp: int) -> bool:
    """True when the finger points roughly upward in the image."""
    return (lm[mcp].y - lm[tip].y) > 0.10


def _finger_vector_down(lm, tip: int, mcp: int) -> bool:
    return (lm[tip].y - lm[mcp].y) > 0.10


def _classify(lm, handedness: str) -> tuple[str, str]:
    thumb = _thumb_extended(lm, handedness)
    index = _finger_extended(lm, INDEX_TIP, INDEX_PIP)
    middle = _finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring = _finger_extended(lm, RING_TIP, RING_PIP)
    pinky = _finger_extended(lm, PINKY_TIP, PINKY_PIP)
    pattern = (int(thumb), int(index), int(middle), int(ring), int(pinky))

    # --- special cases that depend on distances, not just the pattern ----
    pinch = _dist(lm[THUMB_TIP], lm[INDEX_TIP])
    hand_scale = _dist(lm[WRIST], lm[MIDDLE_MCP]) + 1e-6
    if pinch < hand_scale * 0.35 and middle and ring and pinky:
        return "OK", "\U0001F44C"

    # --- pattern table -------------------------------------------------
    table = {
        (0, 0, 0, 0, 0): ("Fist", "✊"),
        (1, 1, 1, 1, 1): ("Open palm", "\U0001F590️"),
        (0, 1, 1, 1, 1): ("Four", "✈️"),
        (0, 1, 1, 1, 0): ("Three", "\U0001F918"),
        (0, 1, 1, 0, 0): ("Peace / Victory", "✌️"),
        (0, 1, 0, 0, 0): ("Pointing", "☝️"),
        (0, 0, 0, 0, 1): ("Pinky", "\U0001F90F"),
        (0, 1, 0, 0, 1): ("Rock on", "\U0001F918"),
        (1, 1, 0, 0, 1): ("Spider-Man", "\U0001F578️"),
        (1, 0, 0, 0, 1): ("Call me", "\U0001F919"),
        (1, 1, 0, 0, 0): ("Gun", "\U0001F52B"),
        (1, 1, 1, 0, 0): ("Three (thumb)", "\U0001F91F"),
    }
    if pattern in table:
        return table[pattern]

    # --- thumb-only: up or down --------------------------------------
    if pattern == (1, 0, 0, 0, 0):
        if _finger_vector_up(lm, THUMB_TIP, THUMB_MCP):
            return "Thumbs up", "\U0001F44D"
        if _finger_vector_down(lm, THUMB_TIP, THUMB_MCP):
            return "Thumbs down", "\U0001F44E"
        return "Thumb", "\U0001F44D"

    count = sum(pattern)
    return f"{count} finger{'s' if count != 1 else ''}", "✋"


def recognize(landmarks, handedness: str = "Right") -> HandState:
    """Classify a single hand.

    ``landmarks`` is a sequence of 21 objects each exposing ``.x`` and ``.y``
    (and optionally ``.z``) attributes, exactly as MediaPipe returns them.
    """
    lm = list(landmarks)
    if len(lm) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(lm)}")

    fingers = (
        int(_thumb_extended(lm, handedness)),
        int(_finger_extended(lm, INDEX_TIP, INDEX_PIP)),
        int(_finger_extended(lm, MIDDLE_TIP, MIDDLE_PIP)),
        int(_finger_extended(lm, RING_TIP, RING_PIP)),
        int(_finger_extended(lm, PINKY_TIP, PINKY_PIP)),
    )
    gesture, emoji = _classify(lm, handedness)
    return HandState(
        handedness=handedness,
        fingers=fingers,
        finger_count=sum(fingers),
        gesture=gesture,
        emoji=emoji,
    )
