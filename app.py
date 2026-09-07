"""Real-time hand-gesture recognition from a webcam.

    python app.py                 # default camera
    python app.py --camera 1      # pick another camera
    python app.py --no-flip       # already-mirrored source? turn the flip off
    python app.py --no-actions    # don't run the gesture -> action mapper

Keys while running:
    q / Esc   quit
    a         toggle action mapper
    h         toggle the on-screen help / HUD
    s         save the current frame to captures/
"""

from __future__ import annotations

import argparse
import time
from collections import Counter, deque
from pathlib import Path

import cv2

from actions import ActionMapper
from gestures import recognize
from hand_tracker import HandTracker

CAPTURE_DIR = Path(__file__).parent / "captures"


class GestureStabiliser:
    """Majority vote over a short sliding window so labels stop flickering."""

    def __init__(self, window: int = 7):
        self._buf: deque[str] = deque(maxlen=window)

    def update(self, label: str) -> str:
        self._buf.append(label)
        return Counter(self._buf).most_common(1)[0][0]

    def reset(self) -> None:
        self._buf.clear()


def _draw_hud(frame, fps, lines, show_help):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.putText(frame, f"{fps:4.1f} FPS", (w - 110, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2)

    y = 70
    for text, colour in lines:
        box = frame.copy()
        cv2.rectangle(box, (10, y - 26), (10 + 12 * len(text) + 20, y + 10),
                      (0, 0, 0), -1)
        cv2.addWeighted(box, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, text, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
        y += 46

    if show_help:
        tips = ["q quit", "a actions", "h HUD", "s snapshot"]
        cv2.putText(frame, "   ".join(tips), (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--max-hands", type=int, default=2)
    ap.add_argument("--no-flip", dest="flip", action="store_false",
                    help="disable the default selfie mirror")
    ap.add_argument("--no-actions", dest="actions", action="store_false")
    ap.set_defaults(flip=True, actions=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. On macOS, grant camera "
            "access to your terminal in System Settings > Privacy & Security > Camera."
        )

    tracker = HandTracker(max_hands=args.max_hands)
    mapper = ActionMapper(enabled=args.actions)
    stabilisers: dict[str, GestureStabiliser] = {}

    show_help = True
    prev = time.time()
    fps = 0.0

    print("Running. Press 'q' in the window to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Dropped a frame from the camera.")
            break
        if args.flip:
            frame = cv2.flip(frame, 1)

        hands = tracker.process(frame)
        tracker.draw(frame, hands)

        seen = set()
        hud_lines = []
        for hand in hands:
            key = hand.handedness
            seen.add(key)
            state = recognize(hand.landmarks, hand.handedness)
            stab = stabilisers.setdefault(key, GestureStabiliser())
            label = stab.update(state.gesture)

            colour = (0, 255, 120) if label != "Fist" else (0, 165, 255)
            hud_lines.append(
                (f"{hand.handedness}: {label}  [{state.finger_count}]", colour)
            )
            mapper.feed(key, label)

        for gone in set(stabilisers) - seen:
            stabilisers[gone].reset()
        if not hands:
            mapper.feed_all_idle()
            hud_lines.append(("no hand", (160, 160, 160)))

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now
        _draw_hud(frame, fps, hud_lines, show_help)
        if mapper.enabled and mapper.last_action:
            cv2.putText(frame, f"action: {mapper.last_action}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

        cv2.imshow("Hand Gesture Recognition", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("a"):
            mapper.enabled = not mapper.enabled
            print(f"action mapper {'on' if mapper.enabled else 'off'}")
        if key == ord("h"):
            show_help = not show_help
        if key == ord("s"):
            CAPTURE_DIR.mkdir(exist_ok=True)
            path = CAPTURE_DIR / f"snap_{int(now)}.png"
            cv2.imwrite(str(path), frame)
            print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()


if __name__ == "__main__":
    main()
