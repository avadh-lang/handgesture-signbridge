"""Collect labeled handshape data for training a fingerspelling classifier
(Phase 4 of docs/REWORK_ROADMAP.md) - guides one person through the static
ASL letters, saving raw hand landmarks tagged by person/letter/handedness.

    python tools/collect_fingerspell.py "avadh"
    python tools/collect_fingerspell.py "avadh" --reps 5
    python tools/collect_fingerspell.py "avadh" --letters ABEMNST   # just these
    python tools/collect_fingerspell.py --list                     # who's recorded what, and how much

J and Z are air-drawn (motion, not a static handshape) - skipped here, same
as fingerspell.py's existing design.

This is NOT learn_sign.py - that teaches SignBridge one whole-word sign via
DTW, instantly, no training needed. This collects raw training examples for
tools/train_fingerspell.py, and that model only helps once there's data from
SEVERAL DIFFERENT PEOPLE: a model trained on one person's hand just
reproduces today's hand-tuned-rules problem (fits one hand's proportions,
fails on everyone else's) under a different name. See docs/REWORK_ROADMAP.md.

Saved to data/fingerspell_dataset/<person_id>/<LETTER>.npz (gitignored -
never committed; these are real people's hand-landmark recordings).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # sibling modules (hud, tracking, ...)

import hud
from skeleton import Pose
from tracking import dominant_hand, make_tracker

STATIC_LETTERS = [c for c in "ABCDEFGHIKLMNOPQRSTUVWXY"]  # J, Z are motion
COUNTDOWN = 1.4
RECORD_SECONDS = 1.8
MIN_SAMPLES_PER_REP = 6
OUT_DIR = Path(__file__).parent.parent / "data" / "fingerspell_dataset"


def _camera(index: int, w: int = 1280, h: int = 720):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    return cap


def _safe_person_id(name: str) -> str:
    slug = re.sub(r"[^a-z0-9_-]+", "-", name.strip().lower()).strip("-")
    if not slug:
        raise SystemExit("--person name must contain at least one letter or digit")
    return slug


def _collect_one_letter(cap, tracker, letter: str, reps: int, flip: bool) -> tuple[list, list, str]:
    """Returns (hands, handedness, status) - status is 'done'/'skipped'/'cancelled'."""
    hands: list[np.ndarray] = []
    handedness: list[str] = []
    rep = 0
    while rep < reps:
        # ---- countdown ----
        t0 = time.time()
        while time.time() - t0 < COUNTDOWN:
            ok, frame = cap.read()
            if not ok:
                return hands, handedness, "cancelled"
            if flip:
                frame = cv2.flip(frame, 1)
            pose = tracker.process(frame)
            tracker.draw(frame, pose)
            left = COUNTDOWN - (time.time() - t0)
            hud.banner(frame, f'"{letter}"  rep {rep + 1}/{reps}  get ready  {max(0, int(left) + 1)}',
                       hud.AMBER, 1.6)
            hud.text(frame, "s=skip letter  Esc=stop & save", (16, frame.shape[0] - 16), 0.55, hud.GREY)
            cv2.imshow("collect fingerspell data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return hands, handedness, "cancelled"
            if key == ord("s"):
                return hands, handedness, "skipped"

        # ---- record ----
        t0 = time.time()
        rep_hands: list[np.ndarray] = []
        rep_handedness: list[str] = []
        while time.time() - t0 < RECORD_SECONDS:
            ok, frame = cap.read()
            if not ok:
                return hands, handedness, "cancelled"
            if flip:
                frame = cv2.flip(frame, 1)
            pose: Pose = tracker.process(frame)
            tracker.draw(frame, pose)
            hand, label = dominant_hand(pose)
            if hand is not None:
                rep_hands.append(hand[:, :2].copy())
                rep_handedness.append(label)
            hud.banner(frame, f'"{letter}"  rep {rep + 1}/{reps}  hold it...', hud.GREEN, 1.6)
            cv2.imshow("collect fingerspell data", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return hands, handedness, "cancelled"
            if key == ord("s"):
                return hands, handedness, "skipped"

        if len(rep_hands) < MIN_SAMPLES_PER_REP:
            continue  # no steady hand seen - retry this rep, don't count it
        hands.extend(rep_hands)
        handedness.extend(rep_handedness)
        rep += 1

    return hands, handedness, "done"


def cmd_list(_args) -> int:
    if not OUT_DIR.exists() or not any(OUT_DIR.iterdir()):
        print("(no fingerspelling data collected yet)")
        return 0
    for person_dir in sorted(p for p in OUT_DIR.iterdir() if p.is_dir()):
        letter_files = sorted(person_dir.glob("*.npz"))
        counts = [f"{lf.stem}:{len(np.load(lf)['hands'])}" for lf in letter_files]
        print(f"  {person_dir.name:20s} {len(letter_files):2d} letters   {' '.join(counts)}")
    return 0


def cmd_collect(args) -> int:
    person_id = _safe_person_id(args.person)
    letters = [c.upper() for c in args.letters] if args.letters else STATIC_LETTERS
    bad = [c for c in letters if c not in STATIC_LETTERS]
    if bad:
        raise SystemExit(f"not static letters (or already recorded elsewhere): {bad} "
                          f"- choose from {''.join(STATIC_LETTERS)}")

    person_dir = OUT_DIR / person_id
    person_dir.mkdir(parents=True, exist_ok=True)

    cap = _camera(args.camera)
    print(f'collecting for "{person_id}" - {len(letters)} letters x {args.reps} reps. '
          f"'s' skips a letter, Esc stops and saves what's done so far.")
    saved, skipped = [], []
    try:
        with make_tracker(args.tracker, swap_hands=args.swap_hands) as tracker:
            for letter in letters:
                hands, handed, status = _collect_one_letter(
                    cap, tracker, letter, args.reps, not args.no_flip)
                if hands:
                    out_path = person_dir / f"{letter}.npz"
                    np.savez(out_path,
                             hands=np.array(hands, dtype=np.float32),
                             handedness=np.array(handed),
                             letter=letter, person_id=person_id)
                    saved.append(letter)
                    print(f'  "{letter}": saved {len(hands)} samples -> {out_path}')
                else:
                    skipped.append(letter)
                if status == "cancelled":
                    print("stopped early - saved everything completed so far")
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\ndone: {len(saved)} letters saved for \"{person_id}\"" +
          (f", {len(skipped)} skipped ({''.join(skipped)})" if skipped else ""))
    print(f"Next: python tools/train_fingerspell.py --holdout {person_id}  "
          f"(once at least one OTHER person has also been collected)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("person", nargs="?", help="who's signing - becomes the folder name")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--letters", help="just these letters, e.g. --letters ABEMNST")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--tracker", choices=("holistic", "hands"), default="holistic")
    ap.add_argument("--swap-hands", action="store_true")
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()

    if args.list:
        return cmd_list(args)
    if not args.person:
        ap.error("give a name for who's signing, or use --list")
    return cmd_collect(args)


if __name__ == "__main__":
    raise SystemExit(main())
