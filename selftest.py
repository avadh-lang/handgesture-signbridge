"""Camera-free checks for the whole SignBridge pipeline.

    python selftest.py

Covers: the original gesture classifier, landmark normalisation invariance,
DTW, the sign bank (record -> match -> reject), fingerspelling on synthetic
handshapes, word prediction, the speller state machine, the glosser, the
Leitner scheduler, the One-Euro filter, and the hands-free pointer.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ------------------------------------------------------------------ harness
_PASS = 0
_FAIL = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    flag = "ok  " if ok else "FAIL"
    _PASS += ok
    _FAIL += not ok
    print(f"  [{flag}] {name}" + (f"   {detail}" if detail and not ok else ""))


# ============================================================ 1. gestures
from gestures import recognize  # noqa: E402

_MCP_X = {"index": 0.44, "middle": 0.50, "ring": 0.56, "pinky": 0.62}


@dataclass
class LM:
    x: float
    y: float
    z: float = 0.0


def _finger(x, extended):
    mcp = LM(x, 0.60)
    if extended:
        return [mcp, LM(x, 0.48), LM(x, 0.40), LM(x, 0.32)]
    return [mcp, LM(x, 0.55), LM(x, 0.60), LM(x, 0.57)]


def build_hand(thumb, index, middle, ring, pinky, thumb_dir="side"):
    pts = [LM(0.50, 0.92)]
    if thumb:
        if thumb_dir == "up":
            tip, ip = LM(0.40, 0.45), LM(0.41, 0.55)
        elif thumb_dir == "down":
            tip, ip = LM(0.40, 0.95), LM(0.41, 0.86)
        else:
            tip, ip = LM(0.24, 0.60), LM(0.32, 0.66)
    else:
        tip, ip = LM(0.47, 0.60), LM(0.45, 0.68)
    pts += [LM(0.44, 0.82), LM(0.42, 0.74), ip, tip]
    for nm, on in (("index", index), ("middle", middle), ("ring", ring), ("pinky", pinky)):
        pts += _finger(_MCP_X[nm], on)
    return pts


def test_gestures():
    cases = [
        ("Fist", dict(thumb=0, index=0, middle=0, ring=0, pinky=0)),
        ("Open palm", dict(thumb=1, index=1, middle=1, ring=1, pinky=1)),
        ("Peace / Victory", dict(thumb=0, index=1, middle=1, ring=0, pinky=0)),
        ("Pointing", dict(thumb=0, index=1, middle=0, ring=0, pinky=0)),
        ("Pinky", dict(thumb=0, index=0, middle=0, ring=0, pinky=1)),
        ("Rock on", dict(thumb=0, index=1, middle=0, ring=0, pinky=1)),
        ("Call me", dict(thumb=1, index=0, middle=0, ring=0, pinky=1)),
        ("Thumbs up", dict(thumb=1, index=0, middle=0, ring=0, pinky=0, thumb_dir="up")),
        ("Thumbs down", dict(thumb=1, index=0, middle=0, ring=0, pinky=0, thumb_dir="down")),
    ]
    for expected, kw in cases:
        got = recognize(build_hand(**kw), "Right").gesture
        check(f"gesture {expected}", got == expected, f"got {got}")


# ============================================================ synthetic hand
# upright right hand, wrist below, fingers up (-y).  centred near origin.
CANON = np.array([
    (0.00, 0.35),                                              # 0 wrist
    (-0.10, 0.22), (-0.16, 0.10), (-0.20, 0.00), (-0.24, -0.08),  # thumb
    (-0.09, 0.00), (-0.10, -0.14), (-0.10, -0.24), (-0.10, -0.33),  # index
    (-0.03, 0.00), (-0.03, -0.16), (-0.03, -0.27), (-0.03, -0.37),  # middle
    (0.03, 0.00), (0.03, -0.14), (0.03, -0.24), (0.03, -0.33),      # ring
    (0.09, 0.02), (0.09, -0.10), (0.09, -0.18), (0.09, -0.25),      # pinky
])
_FINGER_JOINTS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12),
                  "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}


def place(hand, center=(0.5, 0.5), scale=0.8, angle=0.0):
    c, s = np.cos(angle), np.sin(angle)
    out = (hand * scale) @ np.array([[c, s], [-s, c]])
    return out + np.array(center)


def fold(hand, fingers, amount=0.95):
    h = hand.copy()
    for f in fingers:
        mcp, pip, dip, tip = _FINGER_JOINTS[f]
        base = h[mcp].copy()
        for j in (pip, dip, tip):
            h[j] = base + (h[j] - base) * (1 - amount) + np.array([0.0, 0.02 * amount])
    return h


def splay(hand, spread=0.12):
    h = hand.copy()
    h[[6, 7, 8]] += np.array([-spread, 0])
    h[[14, 15, 16]] += np.array([spread, 0])
    return h


def thumb_to(hand, xy):
    h = hand.copy()
    h[4] = np.array(xy)
    h[3] = (h[2] + h[4]) / 2
    return h


# ============================================================ 2. skeleton
from skeleton import (BodyRef, Pose, frame_feature, hand_feature,  # noqa: E402
                      normalize_hand, sequence_feature)


def test_skeleton():
    base = place(CANON, (0.5, 0.5), 0.8, 0.0)
    moved = place(CANON, (0.2, 0.7), 1.3, 0.5)
    f0 = hand_feature(base)
    f1 = hand_feature(moved)
    check("handshape invariant to pos/scale/rotation",
          np.linalg.norm(f0[:42] - f1[:42]) < 0.15,
          f"delta={np.linalg.norm(f0[:42] - f1[:42]):.3f}")

    fist = fold(CANON, ["index", "middle", "ring", "pinky"])
    check("fist vs open differ",
          np.linalg.norm(hand_feature(place(fist)) - hand_feature(base)) > 0.5)

    # body-relative location: same hand, different body position -> same rel loc
    # same hand offset from the torso, but torso in a different place / rolled
    pose_a = Pose(right=place(CANON, (0.5, 0.4)),
                  body=BodyRef(np.array([0.5, 0.3]), 0.4, 0.0))
    pose_b = Pose(right=place(CANON, (0.7, 0.5)),
                  body=BodyRef(np.array([0.7, 0.4]), 0.4, 0.0))
    ra = frame_feature(pose_a)[96:98]
    rb = frame_feature(pose_b)[96:98]
    check("hand location is body-relative", np.linalg.norm(ra - rb) < 0.05,
          f"{ra} vs {rb}")
    # and it actually moves when the hand moves relative to the torso
    pose_c = Pose(right=place(CANON, (0.9, 0.4)),
                  body=BodyRef(np.array([0.5, 0.3]), 0.4, 0.0))
    check("hand location tracks the hand",
          np.linalg.norm(frame_feature(pose_a)[96:98] - frame_feature(pose_c)[96:98]) > 0.3)

    seq = sequence_feature([pose_a] * 10)
    check("sequence_feature shape", seq.shape == (32, 2 * 102), str(seq.shape))


# ============================================================ 3. dtw
from dtw import dtw_distance  # noqa: E402


def test_dtw():
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.standard_normal((32, 8)), axis=0)
    check("dtw identical == 0", dtw_distance(a, a) < 1e-9)
    warped = np.concatenate([a[:1], a, a[-1:]])[:32]
    check("dtw small for time-warp", dtw_distance(a, warped) < 0.5,
          f"{dtw_distance(a, warped):.3f}")
    b = np.cumsum(rng.standard_normal((32, 8)), axis=0)
    check("dtw large for different", dtw_distance(a, b) > dtw_distance(a, warped))


# ============================================================ 4. sign bank
from signbank import SignBank  # noqa: E402


def _sign_clip(kind, seed):
    """A 16-frame Pose sequence: 'wave' = open hand sliding sideways,
    'rise' = fist travelling up."""
    rng = np.random.default_rng(seed)
    body = BodyRef(np.array([0.5, 0.3]), 0.4, 0.0)
    poses = []
    for t in np.linspace(0, 1, 16):
        jitter = rng.standard_normal((21, 2)) * 0.004
        if kind == "wave":
            hand = place(CANON, (0.4 + 0.25 * np.sin(t * np.pi), 0.4), 0.8)
        else:
            hand = place(fold(CANON, ["index", "middle", "ring", "pinky"]),
                         (0.5, 0.6 - 0.3 * t), 0.8)
        poses.append(Pose(right=hand + jitter, body=body))
    return sequence_feature(poses)


def test_signbank():
    with tempfile.TemporaryDirectory() as d:
        bank = SignBank(d)
        bank.add("wave", [_sign_clip("wave", s) for s in (1, 2, 3)], kind="dynamic")
        bank.add("rise", [_sign_clip("rise", s) for s in (4, 5, 6)], kind="dynamic")
        check("bank persists", len(bank) == 2)

        m = bank.match(_sign_clip("wave", 99))
        check("bank matches wave", m is not None and m.name == "wave",
              str(m and m.name))
        m2 = bank.match(_sign_clip("rise", 98))
        check("bank matches rise", m2 is not None and m2.name == "rise",
              str(m2 and m2.name))

        # a totally different clip should be rejected
        rng = np.random.default_rng(7)
        body = BodyRef(np.array([0.5, 0.3]), 0.4, 0.0)
        fist = fold(CANON, ["index", "middle", "ring", "pinky"])
        junk = sequence_feature([
            Pose(right=place(fist, (rng.random(), rng.random()), 0.8), body=body)
            for _ in range(16)])
        check("bank rejects nonsense", bank.match(junk) is None,
              str(bank.match(junk)))

        bank2 = SignBank(d)
        check("bank reloads from disk", set(bank2.names()) == {"wave", "rise"})


# ============================================================ 5. fingerspell
from fingerspell import classify_letter  # noqa: E402


def test_fingerspell():
    B = thumb_to(CANON, (0.0, 0.08))
    A = thumb_to(fold(CANON, ["index", "middle", "ring", "pinky"]), (-0.02, 0.10))
    I = fold(CANON, ["index", "middle", "ring"])
    Y = thumb_to(fold(CANON, ["index", "middle", "ring"]), (-0.30, 0.06))
    L = thumb_to(fold(CANON, ["middle", "ring", "pinky"]), (-0.32, 0.12))
    U = fold(CANON, ["ring", "pinky"])
    V = splay(fold(CANON, ["ring", "pinky"]), 0.13)
    W = fold(CANON, ["pinky"])
    C = thumb_to(fold(CANON, ["index", "middle", "ring", "pinky"], 0.5), (-0.35, 0.05))
    O = thumb_to(fold(CANON, ["index", "middle", "ring", "pinky"], 0.62), (-0.06, -0.22))

    exact = {"B": B, "I": I, "Y": Y, "L": L, "U": U, "V": V, "W": W}
    for want, hand in exact.items():
        got = classify_letter(place(hand), "Right")
        check(f"fingerspell {want}", got.letter == want, f"got {got.letter} ({got.confidence})")

    a_got = classify_letter(place(A), "Right").letter
    check("fingerspell A (fist family)", a_got in {"A", "S", "T", "E", "M", "N"}, a_got)
    for want, hand in (("C", C), ("O", O)):
        got = classify_letter(place(hand), "Right").letter
        check(f"fingerspell {want}", got in {want, "C", "O"}, got)

    check("I hand is static (not motion)", not classify_letter(place(I), "Right").motion)
    check("letter always valid",
          all(classify_letter(place(h), "Right").letter in
              "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for h in exact.values()))

    # handedness: mirror the hand + call it Left -> same letter
    left_v = place(V) * np.array([-1, 1]) + np.array([1, 0])
    check("fingerspell handedness-agnostic",
          classify_letter(left_v, "Left").letter == "V",
          classify_letter(left_v, "Left").letter)


def test_fingerspell_relaxed():
    """Regression tests for a real user-reported bug: fingers held
    *relaxed* (not ruler-straight, as every CANON-based hand above is) used
    to land in the unclassified gap between `ext` and `curled`, where almost
    no letter scores - so letters needing multiple extended fingers (B, U,
    V, H, K, W) and the fist family (A, S, T, E, M, N, which all require
    every finger strictly `curled`) silently failed to read. Fixed by
    narrowing that gap in fingerspell.py's `_features`.
    """
    u_relaxed = fold(fold(CANON, ["ring", "pinky"]), ["index", "middle"], amount=0.18)
    got = classify_letter(place(u_relaxed), "Right").letter
    check("relaxed (not ruler-straight) U still reads as U", got == "U", got)

    v_relaxed = splay(fold(fold(CANON, ["ring", "pinky"]), ["index", "middle"], amount=0.18), 0.13)
    got = classify_letter(place(v_relaxed), "Right").letter
    check("relaxed (not ruler-straight) V still reads as V", got == "V", got)

    b_relaxed = fold(thumb_to(CANON, (0.0, 0.08)), ["index", "middle", "ring", "pinky"], amount=0.15)
    got = classify_letter(place(b_relaxed), "Right").letter
    check("relaxed (not ruler-straight) B still reads as B", got == "B", got)

    a_relaxed = thumb_to(fold(CANON, ["index", "middle", "ring", "pinky"], amount=0.8), (-0.02, 0.10))
    got = classify_letter(place(a_relaxed), "Right").letter
    check("a less-than-fully-clenched fist still reads as the fist family",
          got in {"A", "S", "T", "E", "M", "N"}, got)

    # D is signed with the index pointing up; P/Q/G are the same single-
    # extended-finger shape at a different orientation. D used to win
    # regardless of orientation since it never checked `point` - now the
    # same handshape, just rotated, should read as D / Q / G respectively.
    pointer_hand = thumb_to(fold(CANON, ["middle", "ring", "pinky"]), (0.0, -0.02))
    up = classify_letter(place(pointer_hand, angle=0.0), "Right").letter
    down = classify_letter(place(pointer_hand, angle=np.pi), "Right").letter
    side = classify_letter(place(pointer_hand, angle=np.pi / 2), "Right").letter
    check("index-up reads as D", up == "D", up)
    check("the same handshape pointing down does not get read as D (Q, not D)",
          down != "D", down)
    check("the same handshape pointing sideways does not get read as D (G, not D)",
          side != "D", side)


# ============================================================ 6. predict
from predict import Predictor  # noqa: E402


def test_predict():
    p = Predictor()
    check("predict hello", "hello" in p.complete("hel", 5), str(p.complete("hel", 5)))
    check("predict thank", "thank" in p.complete("than", 5), str(p.complete("than", 5)))
    check("predict ranks common first", p.complete("th", 1)[0] == "the",
          str(p.complete("th", 1)))
    check("predict empty -> []", p.complete("") == [])
    check("is_word", p.is_word("water") and not p.is_word("zzzzq"))


# ============================================================ 7. speller
from speller import Speller  # noqa: E402


def test_speller():
    s = Speller()
    t = 0.0
    for _ in range(20):
        s.feed_letter("h", 0.9, t); t += 0.1
    check("speller commits held letter", s.word == "h", s.word)
    # a repeat needs a release
    for _ in range(20):
        s.feed_letter("h", 0.9, t); t += 0.1
    check("speller blocks repeat w/o release", s.word == "h", s.word)
    for _ in range(4):
        s.feed_letter(None, 0.0, t); t += 0.1
    for _ in range(20):
        s.feed_letter("i", 0.9, t); t += 0.1
    check("speller adds next letter", s.word == "hi", s.word)
    check("speller suggests", "hi" in s.suggestions or "him" in s.suggestions,
          str(s.suggestions))
    s.space()
    check("speller space commits word", s.sentence == ["hi"] and s.word == "")
    s.word = "wor"
    s._refresh()
    s.accept(0)
    check("speller accept suggestion", s.sentence[-1].startswith("wor"),
          str(s.sentence))
    s.word = "abc"
    s.backspace()
    check("speller backspace trims word", s.word == "ab", s.word)
    s.word = ""
    s.backspace()
    check("speller backspace pulls word back from sentence", s.word != "", s.word)
    s.clear()
    check("speller clear", s.text() == "")


# ============================================================ 8. glosser
from glosser import gloss, gloss_line  # noqa: E402


def test_glosser():
    g = gloss_line("The cat is on the mat.")
    check("glosser drops articles/copula", "THE" not in g and "IS" not in g, g)
    check("glosser keeps content", "CAT" in g and "MAT" in g, g)
    cards = gloss("I am going to Zurich")
    check("glosser points pronouns", cards[0].text == "ME", cards[0].text)
    check("glosser fingerspells unknown names",
          any(c.kind == "fingerspell" for c in cards),
          str([(c.text, c.kind) for c in cards]))


# ============================================================ 9. leitner
from leitner import Leitner  # noqa: E402


def test_leitner():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "l.json"
        lt = Leitner(path)
        pool = ["letter:A", "letter:B", "sign:wave"]
        k = lt.next_due(pool, now=100.0)
        check("leitner picks from pool", k in pool, k)
        box = lt.grade("letter:A", True, now=100.0)
        check("leitner promotes on correct", box == 1)
        lt.grade("letter:A", False, now=100.0)
        check("leitner resets on wrong", lt.items["letter:A"]["box"] == 0)
        check("leitner persists", Leitner(path).items["letter:A"]["seen"] == 2)
        check("leitner box_counts sums", sum(lt.box_counts()) == len(lt.items))


# ============================================================ 10. live
from live import LiveSigner  # noqa: E402


def test_live():
    with tempfile.TemporaryDirectory() as d:
        bank = SignBank(d)
        bank.add("wave", [_sign_clip("wave", s) for s in (1, 2, 3)], kind="dynamic")
        ls = LiveSigner()
        body = BodyRef(np.array([0.5, 0.3]), 0.4, 0.0)
        t = 0.0
        fired = None
        for t_i in np.linspace(0, 1, 20):
            hand = place(CANON, (0.4 + 0.25 * np.sin(t_i * np.pi), 0.4), 0.8)
            ls.push(Pose(right=hand, body=body), now=t)
            m = ls.poll(bank, now=t)
            fired = fired or m
            t += 0.08
        check("live matcher fires on a known sign",
              fired is not None and fired.name == "wave", str(fired and fired.name))
        # cooldown: immediately polling again returns nothing
        check("live matcher debounces", ls.poll(bank, now=t) is None)


# ============================================================ 11. filters
from filters import Hysteresis, OneEuroFilter  # noqa: E402


def test_filters():
    rng = np.random.default_rng(3)
    f = OneEuroFilter(freq=60, min_cutoff=1.0, beta=0.02)
    clean = np.linspace(0, 100, 200)
    noisy = clean + rng.standard_normal(200) * 4
    out = np.array([f(v) for v in noisy])
    # smoother than the input ...
    check("one-euro cuts jitter",
          np.std(np.diff(out)) < np.std(np.diff(noisy)) * 0.6,
          f"{np.std(np.diff(out)):.2f} vs {np.std(np.diff(noisy)):.2f}")
    # ... but still tracks the ramp
    check("one-euro still tracks", abs(out[-1] - clean[-1]) < 6, f"{out[-1]:.1f}")

    h = Hysteresis(enter=3, exit_=3)
    seq = [h.update(x) for x in [1, 1, 1, 1, 1, 0, 0, 0, 0]]
    check("hysteresis needs 3 to enter", seq == [False, False, True, True, True,
                                                 True, True, False, False], str(seq))
    h2 = Hysteresis(enter=3, exit_=3)
    seq2 = [h2.update(x) for x in [1, 1, 0, 1, 1, 1]]
    check("hysteresis noise doesn't trip it", seq2 == [False] * 5 + [True], str(seq2))


# ============================================================ 12. pointer
from pointer import (PointerControl, RecordingSink, is_pointing,  # noqa: E402
                     is_scroll_shape, pinch_ratio, zone_map)


def _point_hand(center, pinch=1.0):
    """CANON at center with index extended; thumb pulled toward index if pinch<1."""
    h = fold(CANON, ["middle", "ring", "pinky"])
    if pinch < 1.0:
        h = h.copy()
        h[4] = h[8] + (h[4] - h[8]) * pinch
    return place(h, center, 0.8)


def test_pointer():
    check("zone_map centre -> screen centre",
          zone_map((0.5, 0.5), 1000, 800) == (499, 399))
    tl = zone_map((0.14, 0.14), 1000, 800)
    check("zone_map maps the active-zone corner to (0,0)", tl == (0, 0), str(tl))

    open_h = _point_hand((0.5, 0.4), pinch=1.0)
    check("is_pointing on an index-point", is_pointing(open_h))
    check("pinch open ratio is large", pinch_ratio(open_h) > 0.7,
          f"{pinch_ratio(open_h):.2f}")
    pinched_h = _point_hand((0.5, 0.4), pinch=0.1)
    check("pinch closed ratio is small", pinch_ratio(pinched_h) < 0.45,
          f"{pinch_ratio(pinched_h):.2f}")

    sink = RecordingSink()
    pc = PointerControl(sink=sink)
    t = 0.0
    for _ in range(6):
        pc.update(_point_hand((0.4 + t * 0.05, 0.4)), t, 1 / 30)
        t += 1 / 30
    check("pointer moves the cursor", any(c[0] == "move" for c in sink.calls))
    # a light pinch -> release should left-click
    for _ in range(5):
        pc.update(_point_hand((0.55, 0.4), pinch=0.5), t, 1 / 30); t += 1 / 30
    for _ in range(6):
        pc.update(_point_hand((0.55, 0.4), pinch=1.0), t, 1 / 30); t += 1 / 30
    check("pointer light pinch -> click",
          any(c[0] == "click" for c in sink.calls),
          str([c[0] for c in sink.calls]))
    # a deep short pinch -> right click
    sink2 = RecordingSink()
    pc2 = PointerControl(sink=sink2)
    t = 0.0
    for _ in range(4):
        pc2.update(_point_hand((0.5, 0.4), pinch=1.0), t, 1 / 30); t += 1 / 30
    for _ in range(4):
        pc2.update(_point_hand((0.5, 0.4), pinch=0.05), t, 1 / 30); t += 1 / 30
    for _ in range(6):
        pc2.update(_point_hand((0.5, 0.4), pinch=1.0), t, 1 / 30); t += 1 / 30
    check("pointer deep pinch -> right click",
          any(c[0] == "right_click" for c in sink2.calls),
          str([c[0] for c in sink2.calls]))

    scroll_h = place(fold(CANON, ["ring", "pinky"]), (0.5, 0.4), 0.8)
    check("is_scroll_shape on index+middle", is_scroll_shape(scroll_h))


# ============================================================ 13. sessionlog
from sessionlog import SessionLog  # noqa: E402


def test_sessionlog():
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "log.csv"
        log = SessionLog(path)
        log.log("coach", target="letter:A", score=0.8, result="pass")
        log.log("coach", target="letter:A", score=0.3, result="fail")
        log.log("coach", target="letter:B", score=0.9, result="pass")
        s = log.summary()
        check("sessionlog aggregates attempts", s["letter:A"]["attempts"] == 2)
        check("sessionlog counts passes", s["letter:A"]["passes"] == 1)
        check("sessionlog keeps last score", abs(s["letter:A"]["last_score"] - 0.3) < 1e-6)
        check("sessionlog reload sees rows", SessionLog(path).summary()["letter:B"]["passes"] == 1)


# ============================================================ 14. modes
# Phase 2 of docs/REWORK_ROADMAP.md split each Mode.update() (decide + draw)
# into Mode.compute() (pure-ish state, no hud/cv2 calls) + Mode.update()
# (compute() then draw). These checks exercise compute() directly - the
# path a non-OpenCV frontend (and backend/runtime.py's Engine) uses - plus
# the on_key -> KEYMAP -> on_action shim that keeps the keyboard CLI and any
# future named-action frontend using one shared action vocabulary.
from modes import Communicator, Coach, Control, Ctx, Pointer  # noqa: E402
from recorder import SignRecorder  # noqa: E402


def test_modes():
    with tempfile.TemporaryDirectory() as d:
        bank = SignBank(d)
        live = LiveSigner()
        recorder = SignRecorder()
        body = BodyRef(np.array([0.5, 0.3]), 0.4, 0.0)
        hand = place(thumb_to(CANON, (0.0, 0.08)), (0.5, 0.4), 0.8)  # a static "B"
        pose = Pose(right=hand, body=body)
        ctx = Ctx(None, pose, 0.0, bank, live, recorder)

        comm = Communicator()
        comm.activate(ctx)
        state = comm.compute(ctx)
        check("communicator.compute returns a view-state dict",
              {"word", "sentence_text", "suggestions", "detecting_letter"} <= set(state))
        check("communicator detects a letter from a static handshape",
              state["detecting_letter"] is not None, str(state["detecting_letter"]))

        comm_via_action, comm_via_key = Communicator(), Communicator()
        comm_via_action.on_action("communicator.toggle_reverse", ctx)
        comm_via_key.on_key(ord('v'), ctx)
        check("on_key -> KEYMAP -> on_action agrees with calling on_action directly",
              comm_via_action.reverse and comm_via_key.reverse)

        coach = Coach()
        coach.leitner = Leitner(Path(d) / "leitner.json")      # keep test isolated -
        coach.log = SessionLog(Path(d) / "practice_log.csv")   # don't touch real data/
        coach.activate(ctx)
        cs = coach.compute(ctx)
        check("coach.compute picks a target", cs["target_name"] is not None, str(cs["target_name"]))
        check("coach.compute score is in [0, 1]", 0.0 <= cs["score"] <= 1.0, str(cs["score"]))

        control = Control()
        control.activate(ctx)
        handled = control.on_action("control.learn", ctx)
        check("control.learn (named action) requests a prompt",
              handled and ctx._prompt_request is not None)
        ctx._prompt_request = None
        handled2 = control.on_key(ord('l'), ctx)
        check("control 'l' keypress requests the same prompt via KEYMAP",
              handled2 and ctx._prompt_request is not None)

        pointer = Pointer()
        pointer.activate(ctx)
        ps = pointer.compute(ctx)
        check("pointer.compute reports an enabled flag", "enabled" in ps, str(ps))


# ============================================================ 15. backend
from backend.runtime import _pose_to_landmarks  # noqa: E402


def test_backend_runtime():
    # _pose_to_landmarks is the Pose -> JSON serializer backend/server.py's
    # `landmarks` WebSocket message uses - pure function, no camera needed.
    hand = place(thumb_to(CANON, (0.0, 0.08)), (0.5, 0.4), 0.8)
    body = BodyRef(np.array([0.5, 0.3]), 0.4, 0.0)
    lm = _pose_to_landmarks(Pose(right=hand, body=body))
    check("_pose_to_landmarks serialises the present hand",
          lm["right"] is not None and len(lm["right"]) == 21, str(lm["right"] and len(lm["right"])))
    check("_pose_to_landmarks serialises the body reference",
          lm["body"] is not None and "scale" in lm["body"], str(lm["body"]))
    check("_pose_to_landmarks leaves an absent hand as null", lm["left"] is None)


# ==================================================================== run
def run():
    for fn in (test_gestures, test_skeleton, test_dtw, test_signbank,
               test_fingerspell, test_fingerspell_relaxed, test_predict, test_speller, test_glosser,
               test_leitner, test_live, test_filters, test_pointer,
               test_sessionlog, test_modes, test_backend_runtime):
        print(f"\n{fn.__name__}:")
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            check(fn.__name__, False, f"raised {e!r}")
    print(f"\n{_PASS}/{_PASS + _FAIL} passed")
    return _FAIL


if __name__ == "__main__":
    raise SystemExit(1 if run() else 0)
