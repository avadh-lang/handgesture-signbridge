"""Hands-free mouse: point to move, pinch to click / drag, two fingers to scroll.

Adapted from Prathamesh-2803/Real-time-hand-gesture-control-system - the cursor
smoothing (One-Euro), the active-zone -> screen mapping and the pinch
click/drag state machine - rebuilt on top of ``skeleton.Pose`` so it shares
SignBridge's tracker and runs as a 4th mode.

Why it belongs here: SignBridge already aims to be a full hands-free interface
for someone who signs.  With this you can also move the pointer, click, drag and
scroll - navigate *and* communicate without a keyboard, mouse or voice.

The OS calls go through a ``sink`` so the state machine is testable with a fake.
macOS needs Accessibility permission for the terminal.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from skeleton import (INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_PIP,
                      MIDDLE_TIP, PINKY_PIP, PINKY_TIP, RING_PIP, RING_TIP,
                      THUMB_TIP, WRIST)

# --- tuning (normalised by hand size unless noted) -------------------------
ZONE_MARGIN = 0.14          # inset of the active rectangle within the frame
PINCH_ON = 0.45             # thumb-index gap / hand size: below -> pinched
PINCH_OFF = 0.75            #                              above -> released
DEEP_PINCH = 0.32           # a firmer pinch -> right click
CLICK_MAX_FRAMES = 9        # pinch shorter than this -> click, not drag
DRAG_MIN_FRAMES = 12        # pinch held at least this -> drag
DOUBLE_CLICK_FRAMES = 22    # two clicks within this many frames -> double
CLICK_COOLDOWN = 10
SCROLL_GAIN = 0.6           # screen-px of hand travel -> scroll steps
WAKE_MARGIN = 0.03          # fingertip must be this far above the wrist


def _hand_size(hand: np.ndarray) -> float:
    return max(float(np.hypot(*(hand[INDEX_MCP] - hand[WRIST]))), 1e-3)


def _finger_up(hand: np.ndarray, tip: int, pip: int) -> bool:
    # tip clearly above its own PIP joint (image y points down)
    return (hand[pip][1] - hand[tip][1]) > 0.18 * _hand_size(hand)


def is_pointing(hand: np.ndarray) -> bool:
    """Index extended, middle/ring/pinky folded."""
    return (_finger_up(hand, INDEX_TIP, INDEX_PIP)
            and not _finger_up(hand, MIDDLE_TIP, MIDDLE_PIP)
            and not _finger_up(hand, RING_TIP, RING_PIP)
            and not _finger_up(hand, PINKY_TIP, PINKY_PIP))


def is_scroll_shape(hand: np.ndarray) -> bool:
    """Index + middle both up and close together, ring + pinky folded."""
    if not (_finger_up(hand, INDEX_TIP, INDEX_PIP) and _finger_up(hand, MIDDLE_TIP, MIDDLE_PIP)):
        return False
    if _finger_up(hand, RING_TIP, RING_PIP) or _finger_up(hand, PINKY_TIP, PINKY_PIP):
        return False
    gap = np.hypot(*(hand[INDEX_TIP] - hand[MIDDLE_TIP])) / _hand_size(hand)
    return gap < 0.9


def pinch_ratio(hand: np.ndarray) -> float:
    return float(np.hypot(*(hand[THUMB_TIP] - hand[INDEX_TIP])) / _hand_size(hand))


def hand_awake(hand: np.ndarray) -> bool:
    return (hand[WRIST][1] - hand[INDEX_TIP][1]) > WAKE_MARGIN


def zone_map(pt_xy, screen_w: int, screen_h: int, margin: float = ZONE_MARGIN):
    """Normalised frame point in the active rectangle -> screen pixel."""
    lo, hi = margin, 1.0 - margin
    x = (float(pt_xy[0]) - lo) / (hi - lo)
    y = (float(pt_xy[1]) - lo) / (hi - lo)
    return (int(np.clip(x, 0, 1) * (screen_w - 1)),
            int(np.clip(y, 0, 1) * (screen_h - 1)))


# ---------------------------------------------------------------- OS sink
class PyAutoGUISink:
    def __init__(self):
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
        self._p = pyautogui
        self.screen_w, self.screen_h = pyautogui.size()

    def move(self, x, y):        self._p.moveTo(x, y, _pause=False)
    def drag_to(self, x, y):     self._p.dragTo(x, y, button="left", _pause=False)
    def mouse_down(self):        self._p.mouseDown(button="left", _pause=False)
    def mouse_up(self):          self._p.mouseUp(button="left", _pause=False)
    def click(self):             self._p.click(_pause=False)
    def double_click(self):      self._p.doubleClick(_pause=False)
    def right_click(self):       self._p.rightClick(_pause=False)
    def scroll(self, steps):     self._p.scroll(int(steps), _pause=False)


@dataclass
class RecordingSink:
    """Test double - records calls instead of driving the OS."""
    calls: list = None
    screen_w: int = 1920
    screen_h: int = 1080

    def __post_init__(self):
        self.calls = []

    def __getattr__(self, name):
        def rec(*a):
            self.calls.append((name, *a))
        return rec


def make_sink():
    try:
        return PyAutoGUISink()
    except Exception:
        return None


# ---------------------------------------------------------------- controller
_IDLE, _PINCHED, _DRAG = 0, 1, 2


class PointerControl:
    def __init__(self, sink=None, smoother=None):
        from filters import Hysteresis, OneEuroPoint
        self.sink = sink if sink is not None else make_sink()
        self.smooth = smoother or OneEuroPoint(freq=60.0, min_cutoff=1.2, beta=0.035)
        self._pinch_hyst = Hysteresis(enter=2, exit_=3)
        self.screen = (getattr(self.sink, "screen_w", 1920),
                       getattr(self.sink, "screen_h", 1080))
        self._state = _IDLE
        self._pinch_frames = 0
        self._cooldown = 0
        self._since_click = 999
        self._deep = False
        self._scroll_anchor: float | None = None
        self._start_xy = (0, 0)
        self.available = self.sink is not None

    @property
    def enabled(self) -> bool:
        return self.available

    def reset(self) -> None:
        if self._state == _DRAG and self.sink:
            try:
                self.sink.mouse_up()
            except Exception:
                pass
        self._state = _IDLE
        self._pinch_frames = 0
        self._scroll_anchor = None
        self.smooth.reset()
        self._pinch_hyst.reset()

    def update(self, hand: np.ndarray | None, now: float, dt: float | None = None):
        """Drive the cursor from one hand. Returns overlay lines."""
        if self._cooldown > 0:
            self._cooldown -= 1
        self._since_click += 1

        if hand is None or self.sink is None or not hand_awake(hand):
            if self._state != _IDLE:
                self.reset()
            self._scroll_anchor = None
            return [("point at the screen to wake the pointer", (150, 150, 150))]

        sx, sy = zone_map(hand[INDEX_TIP], *self.screen)
        fx, fy = self.smooth(sx, sy, dt)

        # ---- scroll shape takes over ----
        if is_scroll_shape(hand) and self._state == _IDLE:
            if self._scroll_anchor is None:
                self._scroll_anchor = fy
                return [("scroll ready", (255, 220, 120))]
            delta = self._scroll_anchor - fy
            if abs(delta) > 6:
                self.sink.scroll(delta * SCROLL_GAIN / 10.0)
                if abs(delta) > 90:
                    self._scroll_anchor = fy
            return [(f"scroll  {'UP' if delta > 0 else 'DOWN'}", (255, 220, 0))]
        self._scroll_anchor = None

        ratio = pinch_ratio(hand)
        pinched = self._pinch_hyst.update(ratio < PINCH_ON)
        if ratio > PINCH_OFF:
            pinched = self._pinch_hyst.update(False)

        overlay = [(f"pinch {ratio:.2f}  {'[####]' if pinched else '[    ]'}", (150, 150, 150))]

        if self._state == _IDLE:
            self.sink.move(int(fx), int(fy))
            if pinched:
                self._state = _PINCHED
                self._pinch_frames = 1
                self._deep = ratio < DEEP_PINCH
                self._start_xy = (fx, fy)

        elif self._state == _PINCHED:
            self._pinch_frames += 1
            self._deep = self._deep or ratio < DEEP_PINCH
            moved = math.hypot(fx - self._start_xy[0], fy - self._start_xy[1])
            if not pinched:
                self._end_pinch(moved, overlay)
            elif self._pinch_frames >= DRAG_MIN_FRAMES and moved > 14:
                self._state = _DRAG
                self.sink.mouse_down()
                overlay.append(("drag start", (0, 180, 255)))
            else:
                self.sink.move(int(fx), int(fy))

        elif self._state == _DRAG:
            self.sink.drag_to(int(fx), int(fy))
            if not pinched:
                self.sink.mouse_up()
                self._state = _IDLE
                self._pinch_frames = 0
                overlay.append(("drag end", (0, 180, 255)))

        return overlay

    def _end_pinch(self, moved: float, overlay: list) -> None:
        held = self._pinch_frames
        deep = self._deep
        self._state = _IDLE
        self._pinch_frames = 0
        if moved > 26 or self._cooldown > 0 or held < 2:
            return
        if held <= CLICK_MAX_FRAMES and deep:
            self.sink.right_click()
            overlay.append(("right click", (0, 165, 255)))
        elif self._since_click <= DOUBLE_CLICK_FRAMES:
            self.sink.double_click()
            overlay.append(("double click", (0, 255, 200)))
            self._since_click = 999
        else:
            self.sink.click()
            overlay.append(("click", (0, 255, 130)))
            self._since_click = 0
        self._cooldown = CLICK_COOLDOWN
