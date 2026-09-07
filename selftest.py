"""Synthetic-landmark checks for gestures.recognize() - runs without a camera.

Builds an upright, palm-to-camera right hand with chosen fingers extended or
curled, then asserts the classifier agrees.

    python selftest.py
"""

from __future__ import annotations

from dataclasses import dataclass

from gestures import recognize

# column x for each finger's MCP, upright hand
_MCP_X = {"index": 0.44, "middle": 0.50, "ring": 0.56, "pinky": 0.62}


@dataclass
class LM:
    x: float
    y: float
    z: float = 0.0


def _finger(x: float, extended: bool) -> list[LM]:
    """Return [MCP, PIP, DIP, TIP] for one non-thumb finger."""
    mcp = LM(x, 0.60)
    if extended:
        return [mcp, LM(x, 0.48), LM(x, 0.40), LM(x, 0.32)]
    # curled: joints fold back down toward the palm / wrist
    return [mcp, LM(x, 0.55), LM(x, 0.60), LM(x, 0.57)]


def build_hand(thumb: bool, index: bool, middle: bool, ring: bool, pinky: bool,
               thumb_dir: str = "side") -> list[LM]:
    pts: list[LM] = [LM(0.50, 0.92)]  # 0 wrist

    # thumb (1..4): CMC, MCP, IP, TIP
    if thumb:
        if thumb_dir == "up":
            tip, ip = LM(0.40, 0.45), LM(0.41, 0.55)
        elif thumb_dir == "down":
            tip, ip = LM(0.40, 0.95), LM(0.41, 0.86)
        else:  # sideways, away from palm
            tip, ip = LM(0.24, 0.60), LM(0.32, 0.66)
    else:
        tip, ip = LM(0.47, 0.60), LM(0.45, 0.68)  # tucked across the palm
    pts += [LM(0.44, 0.82), LM(0.42, 0.74), ip, tip]

    for name, on in (("index", index), ("middle", middle),
                     ("ring", ring), ("pinky", pinky)):
        pts += _finger(_MCP_X[name], on)
    assert len(pts) == 21
    return pts


CASES = [
    ("Fist",            dict(thumb=False, index=False, middle=False, ring=False, pinky=False)),
    ("Open palm",       dict(thumb=True,  index=True,  middle=True,  ring=True,  pinky=True)),
    ("Peace / Victory", dict(thumb=False, index=True,  middle=True,  ring=False, pinky=False)),
    ("Pointing",        dict(thumb=False, index=True,  middle=False, ring=False, pinky=False)),
    ("Pinky",           dict(thumb=False, index=False, middle=False, ring=False, pinky=True)),
    ("Rock on",         dict(thumb=False, index=True,  middle=False, ring=False, pinky=True)),
    ("Call me",         dict(thumb=True,  index=False, middle=False, ring=False, pinky=True)),
    ("Thumbs up",       dict(thumb=True,  index=False, middle=False, ring=False, pinky=False, thumb_dir="up")),
    ("Thumbs down",     dict(thumb=True,  index=False, middle=False, ring=False, pinky=False, thumb_dir="down")),
]


def run() -> int:
    failures = 0
    for expected, kw in CASES:
        hand = build_hand(**kw)
        state = recognize(hand, "Right")
        ok = state.gesture == expected
        flag = "ok " if ok else "FAIL"
        print(f"  [{flag}] expected {expected!r:20s} got {state.gesture!r:20s} "
              f"fingers={state.fingers}")
        failures += not ok
    print(f"\n{len(CASES) - failures}/{len(CASES)} passed")
    return failures


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
