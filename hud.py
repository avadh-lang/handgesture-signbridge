"""Small OpenCV drawing helpers shared by every mode.

Kept dumb on purpose: each function takes a BGR frame and draws on it in place.
"""

from __future__ import annotations

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (0, 235, 140)
AMBER = (0, 200, 255)
RED = (60, 60, 235)
GREY = (170, 170, 170)
WHITE = (245, 245, 245)


def shade(frame, x0, y0, x1, y1, alpha: float = 0.55, colour=(0, 0, 0)):
    h, w = frame.shape[:2]
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    box = roi.copy()
    box[:] = colour
    cv2.addWeighted(box, alpha, roi, 1 - alpha, 0, roi)


def text(frame, s, org, scale=0.7, colour=WHITE, thick=2, bg=True):
    if bg:
        (tw, th), base = cv2.getTextSize(s, FONT, scale, thick)
        x, y = org
        shade(frame, x - 6, y - th - 6, x + tw + 6, y + base + 4, 0.5)
    cv2.putText(frame, s, org, FONT, scale, colour, thick, cv2.LINE_AA)


def top_bar(frame, left_text: str, right_text: str = ""):
    w = frame.shape[1]
    shade(frame, 0, 0, w, 44, 0.5)
    cv2.putText(frame, left_text, (16, 29), FONT, 0.7, WHITE, 2, cv2.LINE_AA)
    if right_text:
        (tw, _), _ = cv2.getTextSize(right_text, FONT, 0.6, 2)
        cv2.putText(frame, right_text, (w - tw - 16, 28), FONT, 0.6, GREEN, 2, cv2.LINE_AA)


def panel(frame, lines, org=(16, 70), scale=0.72, gap=40):
    x, y = org
    for s, colour in lines:
        text(frame, s, (x, y), scale, colour)
        y += gap


def caption_bar(frame, s: str, suggestions=None):
    """Bottom strip: the sentence so far + up to 3 numbered suggestions."""
    h, w = frame.shape[:2]
    shade(frame, 0, h - 96, w, h, 0.6)
    cv2.putText(frame, s or " ", (16, h - 58), FONT, 0.9, WHITE, 2, cv2.LINE_AA)
    if suggestions:
        chip = "   ".join(f"[{i+1}] {g}" for i, g in enumerate(suggestions))
        cv2.putText(frame, chip, (16, h - 22), FONT, 0.7, AMBER, 2, cv2.LINE_AA)


def banner(frame, s: str, colour=GREEN, scale=2.4):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, 4)
    x, y = (w - tw) // 2, (h + th) // 2
    shade(frame, x - 24, y - th - 24, x + tw + 24, y + 24, 0.5)
    cv2.putText(frame, s, (x, y), FONT, scale, colour, 4, cv2.LINE_AA)


def progress(frame, frac: float, colour=GREEN):
    h, w = frame.shape[:2]
    frac = max(0.0, min(1.0, frac))
    cv2.rectangle(frame, (0, h - 6), (int(w * frac), h), colour, -1)


def cards(frame, card_texts, active: int = -1):
    """Reverse channel: big gloss cards laid across the screen."""
    h, w = frame.shape[:2]
    shade(frame, 0, 0, w, h, 0.72)
    if not card_texts:
        banner(frame, "...", GREY, 1.6)
        return
    n = len(card_texts)
    cw = min(int(w / max(n, 1)) - 20, 260)
    total = n * (cw + 20) - 20
    x = (w - total) // 2
    for i, t in enumerate(card_texts):
        cx0, cx1 = x, x + cw
        cy0, cy1 = h // 2 - 90, h // 2 + 90
        colour = GREEN if i == active else WHITE
        cv2.rectangle(frame, (cx0, cy0), (cx1, cy1), colour, 2 if i != active else 4)
        scale = 1.4 if len(t) <= 6 else 0.9
        (tw, th), _ = cv2.getTextSize(t, FONT, scale, 3)
        cv2.putText(frame, t, (cx0 + (cw - tw) // 2, h // 2 + th // 2),
                    FONT, scale, colour, 3, cv2.LINE_AA)
        x += cw + 20
