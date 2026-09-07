"""Teach SignBridge a sign - or list / delete / test what it already knows.

    python learn_sign.py "pizza"                 # record 3 demos, save
    python learn_sign.py "mom" --reps 4
    python learn_sign.py "flat spin" --bind app:Safari   # also bind for Control
    python learn_sign.py --list
    python learn_sign.py --delete "pizza"
    python learn_sign.py --check "pizza"         # live distance readout

Demonstrate the same sign each time the countdown ends.  A still handshape is
saved as a 'static' sign, a travelling one as 'dynamic' - automatically.
"""

from __future__ import annotations

import argparse
import time

import cv2

import hud
from commands import load_commands, parse_action, save_commands
from recorder import SignRecorder
from signbank import DEFAULT_DIR, SignBank, slugify
from skeleton import SEQ_LEN, sequence_feature
from tracking import make_tracker


def _camera(index: int, w: int = 1280, h: int = 720):
    cap = cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {index}")
    return cap


def cmd_record(args) -> int:
    bank = SignBank(args.bank)
    action = None
    if args.bind:
        action = parse_action(args.bind)
        if action is None:
            raise SystemExit(f"bad --bind spec: {args.bind!r} (use app:/key:/text:/shell:)")

    cap = _camera(args.camera)
    rec = SignRecorder()
    rec.start(args.name, reps=args.reps)
    print(f'recording "{args.name}" - {args.reps} reps. Esc to abort.')

    with make_tracker(args.tracker, swap_hands=args.swap_hands) as tracker:
        while rec.active:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_flip:
                frame = cv2.flip(frame, 1)
            pose = tracker.process(frame)
            tracker.draw(frame, pose)
            rec.update(pose, time.time())
            hud.banner(frame, rec.hint, hud.AMBER, 1.1)
            hud.progress(frame, rec.rep / max(rec.reps_wanted, 1))
            cv2.imshow("learn sign", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                rec.cancel()
    cap.release()
    cv2.destroyAllWindows()

    if rec.state != "done":
        print("aborted - nothing saved")
        return 1
    result = rec.result()
    meta = bank.add(result["name"], result["clips"], kind=result["kind"],
                    hands=result["hands"], action=action)
    print(f'saved "{meta.name}"  kind={meta.kind}  hands={meta.hands}  reps={meta.reps}')
    if action is not None:
        cmds = load_commands()
        cmds[meta.name] = action
        save_commands(cmds)
        print(f'bound -> {action["type"]}: {action["value"]}')
    return 0


def cmd_list(args) -> int:
    bank = SignBank(args.bank)
    if not len(bank):
        print("(sign bank is empty)")
        return 0
    for m in bank.metas():
        act = f'  ->  {m.action["type"]}:{m.action["value"]}' if m.action else ""
        print(f"  {m.name:20s} {m.kind:8s} {m.hands:5s} x{m.reps}{act}")
    return 0


def cmd_delete(args) -> int:
    bank = SignBank(args.bank)
    ok = bank.remove(args.name)
    cmds = load_commands()
    if args.name in cmds:
        cmds.pop(args.name)
        save_commands(cmds)
    print("deleted" if ok else "no such sign")
    return 0 if ok else 1


def cmd_check(args) -> int:
    bank = SignBank(args.bank)
    if bank.get(args.name) is None:
        raise SystemExit(f'no sign called "{args.name}"')
    slug = slugify(args.name)
    cap = _camera(args.camera)
    buf: list = []
    print("hold / perform the sign; Esc to stop")
    with make_tracker(args.tracker, swap_hands=args.swap_hands) as tracker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if not args.no_flip:
                frame = cv2.flip(frame, 1)
            pose = tracker.process(frame)
            tracker.draw(frame, pose)
            buf.append(pose)
            buf[:] = buf[-48:]
            line = "..."
            if len(buf) >= 8:
                clip = sequence_feature(buf, SEQ_LEN)
                ranked = bank.distances(clip)
                dmap = dict(ranked)
                best = ranked[0]
                line = f"{args.name}: {dmap.get(slug, 9):.2f}   best: {best[0]} {best[1]:.2f}"
            hud.text(frame, line, (16, 40), 0.8, hud.GREEN)
            cv2.imshow("check sign", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", help="name of the sign to record")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--bind", help="also bind for Control mode, e.g. app:Safari")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--delete", metavar="NAME")
    ap.add_argument("--check", metavar="NAME")
    ap.add_argument("--bank", default=str(DEFAULT_DIR))
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--tracker", choices=("holistic", "hands"), default="holistic")
    ap.add_argument("--swap-hands", action="store_true")
    ap.add_argument("--no-flip", action="store_true")
    args = ap.parse_args()

    if args.list:
        return cmd_list(args)
    if args.delete:
        args.name = args.delete
        return cmd_delete(args)
    if args.check:
        args.name = args.check
        return cmd_check(args)
    if not args.name:
        ap.error("give a sign name, or use --list / --delete / --check")
    return cmd_record(args)


if __name__ == "__main__":
    raise SystemExit(main())
