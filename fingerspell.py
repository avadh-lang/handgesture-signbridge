"""ASL fingerspelling alphabet from one hand's landmarks - two classifiers,
one entry point.

``classify_letter`` is the original rule-based geometry classifier: pure
math, zero dependencies, works out of the box. Its hard look-alikes
(A/S/T/M/N are all near-fists; U/V/H differ only by spread; D/G/P/Q differ
only by orientation) are thresholds tuned to one hand's proportions - see
docs/REWORK_ROADMAP.md Phase 4 for the full writeup.

``classify_letter_ml`` is a trained classifier (tools/train_fingerspell.py)
over data/models/fingerspell_rf.joblib, when one has been trained on data
collected with tools/collect_fingerspell.py. Returns ``None`` if no model
file exists.

``classify_letter_best`` is what the app actually calls: the trained model
if there is one, else the rule-based fallback - no call site needs to know
which is active.

**J** and **Z** are drawn in the air - both classifiers report them with
``motion=True`` on their static base handshape (I and D) and rely on the
dynamic layer / sign bank to resolve the stroke.

For guaranteed-correct results on any single letter right now, record it
into the sign bank instead (``learn_sign.py``); the bank is checked before
either classifier runs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from skeleton import (
    INDEX_MCP, INDEX_PIP, INDEX_TIP, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP,
    PINKY_MCP, PINKY_PIP, PINKY_TIP, RING_MCP, RING_PIP, RING_TIP,
    THUMB_IP, THUMB_MCP, THUMB_TIP, WRIST,
    as_xy, hand_angle, hand_feature, normalize_hand, palm_scale,
)

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_ML_MODEL_PATH = Path(__file__).parent / "data" / "models" / "fingerspell_rf.joblib"
_ML_META_PATH = _ML_MODEL_PATH.with_suffix("").with_suffix(".meta.json")


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

    # a finger is judged by its tip relative to its own knuckle, not the wrist.
    # The ext/curled cutoffs used to sit at 0.55/0.2 with a strict 1.03x
    # tip-vs-pip margin - fine for a hand held ruler-straight, but real
    # fingers (especially on hands with different proportions than the
    # original tuner's) often land in the 0.2-0.55 gap between them, where
    # almost no letter scores at all since every rule below tests `ext`/
    # `curled` as strict booleans. Narrowing the gap (and dropping the extra
    # safety margin on the tip/pip ratio - `rise` alone already rules out a
    # folded-but-coincidentally-high tip) means fewer real hands fall into
    # that no-man's-land. See docs/REWORK_ROADMAP.md Phase 4 for the real
    # fix (a trained classifier); this is a bounded improvement to the
    # existing rules, not a replacement for it.
    rise = mcps[:, 1] - tips[:, 1]                       # how far the tip is above the MCP
    ext = (rise > 0.42) & (tip_d > pip_d)
    curled = rise < 0.25
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
    # D is signed with the index pointing up - excluding "down"/"side" stops
    # it swallowing P/Q/G (same single-finger shape, different orientation),
    # which otherwise always lost to D since D never checked orientation.
    s["D"] = _b(i and not m and not r and not p and f.thumb_touch[1] < 0.7
                and f.point in ("up", "camera")) * (
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
    """Best ASL letter for a single hand, with a confidence margin.

    Rule-based: fast, zero dependencies, no data needed - but the hand-tuned
    thresholds above are tuned to one hand's proportions and genuinely
    struggle on the near-fist (A/S/T/M/N) and orientation (D/G/P/Q, U/V/H)
    clusters on a different hand. ``classify_letter_best`` is almost always
    what callers want instead - it uses a trained model when one exists
    (see tools/train_fingerspell.py) and only falls back to this.
    """
    landmarks = list(hand)
    if len(landmarks) != 21:
        raise ValueError(f"expected 21 landmarks, got {len(landmarks)}")
    scores = _scores(_features(landmarks, handedness))
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, bs), (_, ss) = ranked[0], ranked[1]
    conf = 0.0 if bs <= 0 else max(0.0, min(1.0, (bs - ss) / max(bs, 1e-6)))
    return Letter(best, round(conf, 3), motion=best in ("J", "Z"))


# --------------------------------------------------------------- Phase 4 ML
_ml_model = None    # lazy-loaded; None until first use, stays None if unavailable
_ml_failed = False


def _load_ml_model():
    global _ml_model, _ml_failed
    if _ml_model is not None or _ml_failed:
        return _ml_model
    if not _ML_MODEL_PATH.exists():
        _ml_failed = True
        return None
    try:
        import joblib
        _ml_model = joblib.load(_ML_MODEL_PATH)
    except Exception:
        _ml_failed = True
        return None
    return _ml_model


def classify_letter_ml(hand, handedness: str = "Right") -> Letter | None:
    """Trained-classifier fingerspelling (see tools/train_fingerspell.py).
    Returns ``None`` if no model has been trained yet - callers should fall
    back to ``classify_letter``, or just call ``classify_letter_best``."""
    model = _load_ml_model()
    if model is None:
        return None
    hand_xy = np.asarray(hand, dtype=np.float64)[:, :2]
    mirror = handedness.lower().startswith("l")
    feat = hand_feature(hand_xy, mirror=mirror).reshape(1, -1)
    proba = model.predict_proba(feat)[0]
    classes = model.classes_
    order = np.argsort(proba)[::-1]
    best_letter = str(classes[order[0]])
    best_p = float(proba[order[0]])
    second_p = float(proba[order[1]]) if len(order) > 1 else 0.0
    conf = 0.0 if best_p <= 0 else max(0.0, min(1.0, (best_p - second_p) / max(best_p, 1e-6)))
    return Letter(best_letter, round(conf, 3), motion=best_letter in ("J", "Z"))


def classify_letter_best(hand, handedness: str = "Right") -> Letter:
    """What every caller in the app actually uses: the trained classifier
    if one has been trained (see tools/train_fingerspell.py), else the
    rule-based ``classify_letter`` - zero-dependency, works with no setup."""
    return classify_letter_ml(hand, handedness) or classify_letter(hand, handedness)


def ml_model_info() -> dict | None:
    """Metadata about the currently-loaded trained model, or None if there
    isn't one - e.g. to show whether a result came from a model trained on
    a real generalization test or just one person's data."""
    if _load_ml_model() is None:
        return None
    if not _ML_META_PATH.exists():
        return {}
    try:
        return json.loads(_ML_META_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
