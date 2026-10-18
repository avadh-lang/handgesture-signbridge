"""Run the recogniser on a still image (no webcam needed).

    python demo_image.py path/to/hand.jpg
    python demo_image.py path/to/hand.jpg --out annotated.png

Prints the detected gesture(s) and writes an annotated copy next to the input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from fingerspell import classify_letter_best
from gestures import recognize
from hand_tracker import HandTracker


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-hands", type=int, default=2)
    args = ap.parse_args()

    if not args.image.exists():
        print(f"no such file: {args.image}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"could not decode image: {args.image}", file=sys.stderr)
        return 1

    with HandTracker(max_hands=args.max_hands, static=True) as tracker:
        hands = tracker.process(frame)
        tracker.draw(frame, hands)

        if not hands:
            print("no hands detected")
        for i, hand in enumerate(hands, 1):
            state = recognize(hand.landmarks, hand.handedness)
            letter = classify_letter_best(hand.landmarks, hand.handedness)
            spelled = f"{letter.letter}{'*' if letter.motion else ''} ({letter.confidence:.2f})"
            print(f"hand {i}: {hand.handedness:5s}  {state.emoji} {state.gesture}"
                  f"  fingers={state.fingers} count={state.finger_count}"
                  f"  |  ASL letter: {spelled}")
            h, w = frame.shape[:2]
            x = int(hand.landmarks[0].x * w)
            y = int(hand.landmarks[0].y * h)
            cv2.putText(frame, f"{state.gesture} / {letter.letter}",
                        (max(x - 40, 10), max(y - 20, 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 120), 2)

    out = args.out or args.image.with_name(args.image.stem + "_annotated.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out), frame):
        print(f"failed to write {out}", file=sys.stderr)
        return 1
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
