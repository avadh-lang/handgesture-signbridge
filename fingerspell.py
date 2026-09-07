"""Rule-based ASL fingerspelling alphabet from one hand's landmarks.

Pure geometry, camera-free, so it's covered by the self-test.

Reality check: the ASL alphabet has hard look-alikes (A/S/T/M/N are all
near-fists; U/V/H differ only by spread and orientation).  This classifier
nails the well-separated letters and returns a lower ``confidence`` for the
ambiguous clusters.  **J** and **Z** are drawn in the air - they're reported
with ``motion=True`` on their static base handshape (I and D) and resolved by
the dynamic layer / sign bank.

For rock-solid fingerspelling, record your own A-Z into the sign bank
(``learn_sign.py``); the bank is checked before this classifier.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from skeleton import (
    INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_TIP, RING_MCP, RING_PIP, RING_TIP,
    THUMB_IP, THUMB_MCP, THUMB_TIP, WRIST,
    as_xy, hand_angle, normalize_hand, palm_scale,
)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class Letter:
    letter: str
    confidence: float          # margin between best and runner-up, 0..1
    motion: bool = False       # True for J / Z - needs the air-drawn stroke


@dataclass
class _Feat:
    ext: np.ndarray            # (4,) index..pinky fully extended
    half: np.ndarray           # (4,) partly raised (hooked / curved)
    curled: np.ndarray         # (4,) folded to the palm
    tips: np.ndarray           # (4,2) fingertip xy in the up-pointing norm frame
    thumb: np.ndarray          # (2,) thumb tip xy, same frame
    thumb_ip: np.ndarray
    thumb_touch: np.ndarray    # (4,) thumb-tip -> each fingertip distance
    spread: float              # index<->pinky tip gap
    im_gap: float              # index<->middle tip gap
    crossed: bool              # index / middle tips swapped (R)
    point: str                 # "up" | "down" | "side" | "camera"


def _features(hand, handedness: str = "Right") -> _Feat:
    mirror = handedness.lower().startswith("l")
    norm = normalize_hand(hand, mirror=mirror)
    raw = as_xy(hand)
    if mirror:
        raw = raw * np.array([-1.0, 1.0])

    tip_idx = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    pip_idx = [INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP]
    mcp_idx = [INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    tips = norm[tip_idx]
    pips = norm[pip_idx]
    mcps = norm[mcp_idx]
    tip_d = np.linalg.norm(tips, axis=1)
    pip_d = np.linalg.norm(pips, axis=1)

    # a finger is judged by its tip relative to its own knuckle, not the wrist
    rise = mcps[:, 1] - tips[:, 1]                       # how far the tip is above the MCP
    ext = (rise > 0.55) & (tip_d > pip_d * 1.03)
    curled = rise < 0.2
    half = ~ext & ~curled
    thumb = norm[THUMB_TIP]
    thumb_touch = np.linalg.norm(tips - thumb, axis=1)

    # pointing direction from the raw image-frame knuckle axis
    ang = hand_angle(raw)                       # 0 = +x (image right), CW positive (y down)
    deg = np.degrees(ang) % 360
    if 200 <= deg <= 340:
        point = "up"
    elif 20 <= deg <= 160:
        point = "down"
    else:
        point = "side"
    if palm_scale(raw) < 0.06:                  # tiny apparent hand -> pointing at lens
        point = "camera"

    im_gap = float(np.hypot(*(tips[0] - tips[1])))
    spread = float(np.hypot(*(tips[0] - tips[3])))
    # index normally sits left of middle in this frame; R crosses it over
    crossed = bool(ext[0] and ext[1] and tips[0, 0] > tips[1, 0] + 0.08)
    return _Feat(ext, half, curled, tips, thumb, norm[THUMB_IP],
                 thumb_touch, spread, im_gap, crossed, point)


def _b(x) -> float:
    return 1.0 if x else 0.0


def _scores(f: _Feat) -> dict[str, float]:
    i, m, r, p = f.ext
    ic, mc, rc, pc = f.curled
    fist = bool(np.all(f.curled))
    all_ext = bool(np.all(f.ext))
    thumb_up = f.thumb[1] < -1.1
    thumb_out = abs(f.thumb[0]) > 0.7 and abs(f.thumb[1]) < 0.95
    thumb_across = abs(f.thumb[0]) < 0.55 and f.thumb[1] > -0.95
    thumb_among_fingers = f.thumb[1] < -0.95
    touch_index = f.thumb_touch[0] < 0.6
    touch_all = float(np.mean(f.thumb_touch < 0.7))
    curvy = bool(np.all(f.half | f.ext)) and not all_ext and not fist

    s: dict[str, float] = {}
    s["A"] = _b(fist) * (0.55 + 0.4 * _b(not thumb_across))
    s["S"] = _b(fist) * (0.5 + 0.45 * _b(thumb_across))
    s["E"] = _b(fist) * (0.4 + 0.35 * _b(f.thumb[1] > -0.6) + 0.2 * touch_all)
    s["T"] = _b(fist) * (0.35 + 0.45 * _b(-0.2 < f.thumb[0] < 0.4 and -0.9 < f.thumb[1] < -0.2))
    s["M"] = _b(fist) * (0.3 + 0.35 * _b(f.thumb[0] > 0.35))
    s["N"] = _b(fist) * (0.28 + 0.3 * _b(f.thumb[0] > 0.2))

    s["B"] = _b(all_ext) * (0.55 + 0.25 * _b(f.spread < 1.1) + 0.2 * _b(thumb_across))
    s["C"] = _b(curvy) * (0.5 + 0.5 * _b(f.thumb_touch[:2].min() > 0.55))
    s["O"] = _b(not f.ext.any() and thumb_among_fingers) * (0.3 + 0.7 * _b(touch_all >= 0.5))

    s["I"] = _b(p and not i and not m and not r) * 0.95
    s["Y"] = _b(p and not i and not m and not r and thumb_out) * 1.0
    s["L"] = _b(i and not m and not r and not p and thumb_out) * 1.0
    s["D"] = _b(i and not m and not r and not p and f.thumb_touch[1] < 0.7) * (
        0.7 + 0.3 * _b(not thumb_out))
    s["G"] = _b(i and not m and not r and not p and f.point in ("side", "camera")) * 0.6
    s["X"] = _b(f.half[0] and not i and f.curled[1] and f.curled[2] and f.curled[3]) * 0.8

    s["U"] = _b(i and m and not r and not p and f.im_gap < 0.5) * 0.9
    s["V"] = _b(i and m and not r and not p and f.im_gap >= 0.5) * 0.95
    s["H"] = _b(i and m and not r and not p and f.point == "side") * 0.7
    s["R"] = _b(f.crossed and not r and not p) * 1.0
    s["K"] = _b(i and m and not r and not p and thumb_up and f.im_gap >= 0.4) * 0.7
    s["P"] = _b(i and m and not r and not p and f.point == "down") * 0.6

    s["W"] = _b(i and m and r and not p) * 0.95
    s["F"] = _b(m and r and p and touch_index) * 1.0

    s["Q"] = _b(i and not m and not r and not p and f.point == "down") * 0.6
    s["J"] = s["I"] * 0.4       # base handshape only; motion resolves it
    s["Z"] = s.get("D", 0) * 0.35
    s.pop("4", None)
    return s


def classify_letter(hand, handedness: str = "Right") -> Letter:
    """Best ASL letter for a single hand, with a confidence margin."""
    landmarks = list(hand)
    if len(landmarks) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(landmarks)}")
    scores = _scores(_features(landmarks, handedness))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, bs), (_, ss) = ranked[0], ranked[1]
    conf = 0.0 if bs <= 0 else max(0.0, min(1.0, (bs - ss) / max(bs, 1e-6)))
    return Letter(best, round(conf, 3), motion=best in ("J", "Z"))
