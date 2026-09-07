"""Turn raw MediaPipe landmarks into pose features a matcher can use.

No MediaPipe / OpenCV import here on purpose - this is pure geometry so it can
be unit-tested without a camera.

A *hand* is 21 landmarks, each with ``.x .y .z`` in [0, 1] (origin top-left,
y points down), exactly as MediaPipe returns them - or an ``(N, 2|3)`` array.

Three things get derived:

* **handshape** - wrist-centred, palm-scaled, de-rotated landmarks.  Invariant
  to where the hand is, how big it looks, and how it's rolled, so the same
  handshape lands in the same place whoever signs it and wherever in frame.
* **location** - where the hand sits *relative to the body* (origin between the
  shoulders, unit = shoulder width, corrected for body roll).  This is the ASL
  "location" parameter - "at the chin" vs "at the chest" is a different sign.
* **movement** - the frame-to-frame change of the above over a short window.

``frame_feature`` packs one moment into a vector; ``sequence_feature`` packs a
whole clip (resampled to a fixed length) so two clips of different duration can
be compared frame-for-frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---- landmark indices (MediaPipe Hands / Holistic hand model) --------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

MCPS = (INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)
PIPS = (INDEX_PIP, MIDDLE_PIP, RING_PIP, PINKY_PIP)
TIPS = (INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)

HAND_FEATURE_DIM = 46          # see hand_feature()
FRAME_FEATURE_DIM = 2 * HAND_FEATURE_DIM + 10   # see frame_feature()
SEQ_LEN = 32                   # frames every clip is resampled to


def as_xy(landmarks) -> np.ndarray:
    """Accept MediaPipe landmark objects or an array; return an (N, 2) float array."""
    arr = np.asarray(
        [(p.x, p.y) for p in landmarks] if hasattr(landmarks[0], "x") else landmarks,
        dtype=np.float64,
    )
    return arr[:, :2]


@dataclass
class BodyRef:
    """A frame of reference glued to the signer's torso."""

    origin: np.ndarray        # (2,) midpoint of the shoulders, image coords
    scale: float              # shoulder width in image units
    angle: float              # roll of the shoulder line, radians

    @classmethod
    def from_pose_landmarks(cls, pose) -> "BodyRef | None":
        if pose is None:
            return None
        pts = as_xy(pose)
        if pts.shape[0] < 25:
            return None
        l_sh, r_sh = pts[11], pts[12]
        origin = (l_sh + r_sh) / 2.0
        scale = float(np.hypot(*(l_sh - r_sh))) or 1e-6
        angle = float(np.arctan2(l_sh[1] - r_sh[1], l_sh[0] - r_sh[0]))
        return cls(origin, scale, angle)

    def project(self, point_xy: np.ndarray) -> np.ndarray:
        """Image point -> body frame: (0,0) between shoulders, +x toward the
        signer's left, +y downward, 1.0 == one shoulder width."""
        rel = np.asarray(point_xy, dtype=np.float64) - self.origin
        c, s = np.cos(-self.angle), np.sin(-self.angle)
        rot = np.array([[c, -s], [s, c]]) @ rel
        return rot / self.scale


@dataclass
class Pose:
    """Everything we know about one video frame."""

    left: np.ndarray | None = None     # (21, 2) image coords, or None
    right: np.ndarray | None = None
    body: BodyRef | None = None

    def has_hands(self) -> bool:
        return self.left is not None or self.right is not None


def _rotate(pts: np.ndarray, angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return pts @ np.array([[c, s], [-s, c]])


def palm_scale(hand: np.ndarray) -> float:
    """Mean wrist->MCP distance - a stable proxy for apparent hand size."""
    d = np.linalg.norm(hand[list(MCPS)] - hand[WRIST], axis=1)
    return float(d.mean()) or 1e-6


def hand_angle(hand: np.ndarray) -> float:
    """Image-frame angle of the wrist->middle-knuckle axis (0 = pointing right)."""
    v = hand[MIDDLE_MCP] - hand[WRIST]
    return float(np.arctan2(v[1], v[0]))


def _oriented(hand: np.ndarray, mirror: bool) -> np.ndarray:
    pts = np.asarray(hand, dtype=np.float64)[:, :2].copy()
    if mirror:
        pts[:, 0] = -pts[:, 0]
    return pts


def normalize_hand(hand: np.ndarray, mirror: bool = False) -> np.ndarray:
    """Wrist-centred, palm-scaled, de-rotated so the hand 'points up'.

    ``mirror`` flips left-hand geometry onto the right so a sign matches
    whichever hand performs it.
    """
    pts = _oriented(hand, mirror)
    pts = pts - pts[WRIST]
    pts = pts / palm_scale(pts)
    # rotate wrist->middle-MCP onto -y ("up" in image terms)
    ang = np.arctan2(pts[MIDDLE_MCP][1], pts[MIDDLE_MCP][0])
    return _rotate(pts, -(ang + np.pi / 2))


def _extended(hand: np.ndarray) -> np.ndarray:
    """Boolean (4,) for index..pinky: tip further from wrist than PIP."""
    tip_d = np.linalg.norm(hand[list(TIPS)] - hand[WRIST], axis=1)
    pip_d = np.linalg.norm(hand[list(PIPS)] - hand[WRIST], axis=1)
    return tip_d > pip_d * 1.05


def hand_feature(hand: np.ndarray, mirror: bool = False) -> np.ndarray:
    """(46,) description of one handshape + its image-frame orientation.

    layout: 42 de-rotated (x, y) landmark coords
          +  2 orientation unit vector (cos, sin of hand angle)
          +  1 finger spread   (mean adjacent fingertip gap / palm scale)
          +  1 finger curl     (mean tip-to-wrist / MCP-to-wrist ratio)
    """
    oriented = _oriented(hand, mirror)
    norm = normalize_hand(hand, mirror=mirror)
    ang = hand_angle(oriented)
    tips = norm[list(TIPS)]
    spread = float(np.linalg.norm(np.diff(tips, axis=0), axis=1).mean())
    tip_d = np.linalg.norm(norm[list(TIPS)], axis=1)
    mcp_d = np.linalg.norm(norm[list(MCPS)], axis=1) + 1e-6
    curl = float((tip_d / mcp_d).mean())
    return np.concatenate([
        norm.reshape(-1),
        [np.cos(ang), np.sin(ang), spread, curl],
    ])


_ZERO_HAND = np.zeros(HAND_FEATURE_DIM)


def frame_feature(pose: Pose) -> np.ndarray:
    """(102,) vector for one frame.

    layout: left hand_feature (46)  [zeros if absent]
          + right hand_feature (46)
          + left present, right present            (2)
          + left location (x, y), right location   (4)
          + inter-hand vector (x, y)               (2)
          + inter-hand distance                    (1)
          + both-hands flag                        (1)
    """
    body = pose.body
    left_present = pose.left is not None
    right_present = pose.right is not None

    lf = hand_feature(pose.left, mirror=True) if left_present else _ZERO_HAND
    rf = hand_feature(pose.right) if right_present else _ZERO_HAND

    def loc(hand):
        if hand is None:
            return np.zeros(2)
        c = np.asarray(hand, dtype=np.float64).mean(axis=0)
        return body.project(c) if body is not None else (c - 0.5) / 0.5

    l_loc, r_loc = loc(pose.left), loc(pose.right)
    if left_present and right_present:
        between = r_loc - l_loc
        dist = float(np.hypot(*between))
        both = 1.0
    else:
        between, dist, both = np.zeros(2), 0.0, 0.0

    return np.concatenate([
        lf, rf,
        [float(left_present), float(right_present)],
        l_loc, r_loc,
        between, [dist, both],
    ])


def _resample(seq: np.ndarray, n: int) -> np.ndarray:
    """Linear-resample an (T, D) array onto ``n`` rows."""
    seq = np.asarray(seq, dtype=np.float64)
    if seq.shape[0] == n:
        return seq
    if seq.shape[0] == 1:
        return np.repeat(seq, n, axis=0)
    src = np.linspace(0.0, 1.0, seq.shape[0])
    dst = np.linspace(0.0, 1.0, n)
    return np.stack([np.interp(dst, src, seq[:, i]) for i in range(seq.shape[1])], axis=1)


def sequence_feature(poses, length: int = SEQ_LEN, with_velocity: bool = True) -> np.ndarray:
    """A clip of poses -> ``(length, D)`` feature matrix ready for DTW.

    With velocity the second half of D is the frame-to-frame delta, which is
    what makes "move up" and "move down" from the same handshape distinct.
    """
    frames = np.stack([frame_feature(p) for p in poses]) if len(poses) else \
        np.zeros((1, FRAME_FEATURE_DIM))
    frames = _resample(frames, length)
    if not with_velocity:
        return frames
    vel = np.gradient(frames, axis=0)
    return np.concatenate([frames, vel * 3.0], axis=1)


def motion_energy(poses) -> float:
    """How much the hands moved across the clip, in shoulder-widths.  Lets the
    recorder tell a static handshape from a travelling sign."""
    if len(poses) < 2:
        return 0.0
    cs = []
    for p in poses:
        hs = [h for h in (p.left, p.right) if h is not None]
        if not hs:
            cs.append(None)
            continue
        c = np.mean([np.asarray(h).mean(axis=0) for h in hs], axis=0)
        cs.append(p.body.project(c) if p.body is not None else (c - 0.5) / 0.5)
    pts = [c for c in cs if c is not None]
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
