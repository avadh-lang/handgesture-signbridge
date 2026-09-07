"""SignBridge - a real-time sign-language studio on one webcam.

    python app.py                      # Communicator mode, Holistic tracker
    python app.py --mode coach
    python app.py --tracker hands      # lighter, no body reference
    python app.py --camera 1 --no-flip

Modes (switch any time with Tab):
    1  Communicator  sign / fingerspell -> text; speak it or type it into the
                     focused app.  'v' = reverse channel (type -> gloss cards).
    2  Coach         practise a sign, get a live score, spaced-repetition drills.
    3  Control       a custom sign fires an app / key / shell / text macro.
    4  Pointer       hands-free mouse: point to move, pinch to click / drag,
                     two fingers to scroll  (needs pyautogui + Accessibility).

Global keys:  Tab next mode   `  prev mode   s snapshot   h help   q/Esc quit
Recording a sign:  Esc cancels.

Teach the app your own signs any time with:  python learn_sign.py "pizza"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

import hud
from live import LiveSigner
from modes import Ctx, all_modes
from recorder import SignRecorder
from signbank import DEFAULT_DIR, SignBank
from textinput import TextPrompt
from tracking import make_tracker

CAPTURE_DIR = Path(__file__).parent / "captures"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--tracker", choices=("holistic", "hands"), default="holistic")
    ap.add_argument("--mode", default="communicator",
                    help="communicator | coach | control | pointer  (or 1..4)")
    ap.add_argument("--bank", default=str(DEFAULT_DIR), help="sign bank directory")
    ap.add_argument("--swap-hands", action="store_true",
                    help="use if left/right hands come through mirrored")
    ap.add_argument("--no-flip", dest="flip", action="store_false")
    ap.set_defaults(flip=True)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. On macOS grant Camera access "
            "to your terminal in System Settings > Privacy & Security > Camera."
        )

    bank = SignBank(args.bank)
    live = LiveSigner()
    recorder = SignRecorder()
    prompt = TextPrompt()
    modes = all_modes()
    names = ["communicator", "coach", "control", "pointer"]
    idx = {"1": 0, "2": 1, "3": 2, "4": 3}.get(args.mode, None)
    if idx is None:
        idx = names.index(args.mode) if args.mode in names else 0
    mode = modes[idx]
    mode.activate(Ctx(None, None, time.time(), bank, live, recorder))  # type: ignore

    pending_prompt: tuple | None = None      # (mode, kind)
    show_help = True
    fps = 0.0
    prev = time.time()

    print(f"SignBridge running - mode: {names[idx]}, tracker: {args.tracker}. "
          f"{len(bank)} signs in the bank. Press 'q' in the window to quit.")

    with make_tracker(args.tracker, swap_hands=args.swap_hands) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera frame dropped")
                break
            if args.flip:
                frame = cv2.flip(frame, 1)
            now = time.time()

            pose = tracker.process(frame)
            live.push(pose, now)

            recorder.update(pose, now)
            if recorder.state == "done":
                mode.on_recorded(recorder.result(), Ctx(frame, pose, now, bank, live, recorder))
                recorder.state = "idle"
            elif recorder.state == "cancelled":
                recorder.state = "idle"

            tracker.draw(frame, pose)

            ctx = Ctx(frame, pose, now, bank, live, recorder)
            ctx.busy = prompt.active or recorder.active
            mode.update(ctx)

            if ctx._prompt_request is not None:
                kind, label, initial = ctx._prompt_request
                prompt.start(label, initial)
                pending_prompt = (mode, kind)

            if recorder.active:
                hud.banner(frame, recorder.hint, hud.AMBER, 1.1)
            prompt.draw(frame)

            now2 = time.time()
            fps = 0.9 * fps + 0.1 / max(now2 - prev, 1e-6)
            prev = now2
            if show_help:
                hud.text(frame, f"{mode.keyhint}   |   Tab mode  q quit   {fps:4.1f}fps",
                         (16, frame.shape[0] - 8), 0.5, hud.GREY)

            cv2.imshow("SignBridge", frame)
            code = cv2.waitKey(1) & 0xFF

            if code == 255:
                continue
            if prompt.active:
                state, value = prompt.key(code)
                if state == "submit" and pending_prompt:
                    m, kind = pending_prompt
                    m.on_prompt(kind, value or "", ctx)
                    pending_prompt = None
                    if ctx._prompt_request is not None:
                        k2, l2, i2 = ctx._prompt_request
                        prompt.start(l2, i2)
                        pending_prompt = (m, k2)
                elif state == "cancel" and pending_prompt:
                    m, kind = pending_prompt
                    m.on_prompt(kind, "", ctx)
                    pending_prompt = None
                continue

            if recorder.active and code == 27:
                recorder.cancel()
                continue
            if code in (ord("q"), 27):
                break
            if code == 9:                                  # Tab
                idx = (idx + 1) % len(modes)
                mode = modes[idx]
                mode.activate(ctx)
                live.reset()
                continue
            if code == 96:                                 # backtick
                idx = (idx - 1) % len(modes)
                mode = modes[idx]
                mode.activate(ctx)
                live.reset()
                continue
            if mode.on_key(code, ctx):
                if ctx._prompt_request is not None:
                    kind, label, initial = ctx._prompt_request
                    prompt.start(label, initial)
                    pending_prompt = (mode, kind)
                continue
            if code == ord("h"):
                show_help = not show_help
            elif code == ord("s"):
                CAPTURE_DIR.mkdir(exist_ok=True)
                path = CAPTURE_DIR / f"snap_{int(now)}.png"
                cv2.imwrite(str(path), frame)
                print(f"saved {path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
