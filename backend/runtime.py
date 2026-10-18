"""The camera -> tracker -> mode engine, extracted so both the cv2 CLI
(``app.py``, unchanged) and the WebSocket server (``backend/server.py``) can
eventually share one loop. For now this is a fresh implementation of the
same loop shape, built on ``Mode.compute()`` (the pure-state path added to
``modes.py`` in this pass) rather than ``Mode.update()`` (which also draws
via ``hud.py`` - the cv2-only rendering path ``app.py`` still uses). Keeping
these as two call paths, rather than making ``app.py`` depend on this
``Engine``, was a deliberate choice to avoid touching the working CLI loop
in the same change that introduces the server - see docs/REWORK_ROADMAP.md.

The Engine also encodes and streams the camera frame itself (as JPEG, see
``_JPEG_EVERY``) rather than leaving the browser to grab its own feed via
``getUserMedia``. The original plan left that an open question ("can the
browser and Python read the same camera at once?"); streaming the frame
from the one process that's already reading the camera sidesteps the
question entirely, keeps Python as the single camera owner (consistent
with the rest of the architecture), and costs very little on localhost -
see docs/ARCHITECTURE.md for the updated wire protocol.

Run standalone for a quick camera-free sanity check:
    .venv/bin/python -m backend.runtime
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass

import cv2
import numpy as np

from live import LiveSigner
from modes import Ctx, all_modes
from recorder import SignRecorder
from signbank import DEFAULT_DIR, SignBank
from skeleton import Pose
from tracking import make_tracker

MODE_NAMES = ["communicator", "coach", "control", "pointer"]
_JPEG_EVERY = 2  # stream video at roughly half the landmark rate (~15fps of ~30)


@dataclass
class EngineConfig:
    camera: int = 0
    width: int = 1280
    height: int = 720
    tracker: str = "holistic"
    mode: str = "communicator"
    bank: str = str(DEFAULT_DIR)
    swap_hands: bool = False
    flip: bool = True


def _pose_to_landmarks(pose: Pose) -> dict:
    """``skeleton.Pose`` -> a plain JSON-able dict for the ``landmarks``
    WebSocket message (see docs/ARCHITECTURE.md's wire protocol table)."""

    def _hand(h: np.ndarray | None):
        if h is None:
            return None
        return [[float(x), float(y)] for x, y in h[:, :2]]

    body = None
    if pose.body is not None:
        body = {
            "origin": [float(pose.body.origin[0]), float(pose.body.origin[1])],
            "scale": float(pose.body.scale),
            "angle": float(pose.body.angle),
        }
    return {"left": _hand(pose.left), "right": _hand(pose.right), "body": body}


class Engine:
    """Owns the camera, tracker and all mode state. ``step()`` advances one
    frame and returns everything a frontend needs; ``handle_action()`` /
    ``handle_prompt_submit()`` / ``handle_prompt_cancel()`` take input back
    in, using the same named-action vocabulary as ``Mode.KEYMAP`` (see
    ``modes.py``)."""

    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.cap = cv2.VideoCapture(self.config.camera)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {self.config.camera}")

        self.bank = SignBank(self.config.bank)
        self.live = LiveSigner()
        self.recorder = SignRecorder()
        self.modes = all_modes()

        self.idx = {"1": 0, "2": 1, "3": 2, "4": 3}.get(self.config.mode)
        if self.idx is None:
            self.idx = MODE_NAMES.index(self.config.mode) if self.config.mode in MODE_NAMES else 0
        self.mode = self.modes[self.idx]

        self._tracker_cm = make_tracker(self.config.tracker, swap_hands=self.config.swap_hands)
        self.tracker = self._tracker_cm.__enter__()
        self.mode.activate(Ctx(None, None, time.time(), self.bank, self.live, self.recorder))

        self.pending_prompt: tuple | None = None   # (mode, kind)
        self._ctx: Ctx | None = None
        self._frame_count = 0

    def close(self) -> None:
        self._tracker_cm.__exit__(None, None, None)
        self.cap.release()

    @property
    def mode_name(self) -> str:
        return MODE_NAMES[self.idx]

    def step(self) -> dict:
        """Advance one frame. Always returns a dict with ``ok``; on success
        also ``landmarks``, ``view_state``, ``prompt`` (or None),
        ``recording`` (or None)."""
        ok, frame = self.cap.read()
        if not ok:
            return {"ok": False}
        if self.config.flip:
            frame = cv2.flip(frame, 1)
        now = time.time()

        pose = self.tracker.process(frame)
        self.live.push(pose, now)

        self.recorder.update(pose, now)
        if self.recorder.state == "done":
            self.mode.on_recorded(self.recorder.result(),
                                   Ctx(frame, pose, now, self.bank, self.live, self.recorder))
            self.recorder.state = "idle"
        elif self.recorder.state == "cancelled":
            self.recorder.state = "idle"

        ctx = Ctx(frame, pose, now, self.bank, self.live, self.recorder)
        ctx.busy = self.pending_prompt is not None or self.recorder.active
        view_state = self.mode.compute(ctx)
        self._ctx = ctx

        prompt_msg = None
        if ctx._prompt_request is not None:
            kind, label, initial = ctx._prompt_request
            self.pending_prompt = (self.mode, kind)
            prompt_msg = {"kind": kind, "label": label, "initial": initial}

        # always present, not just while active - otherwise a frontend has no
        # signal to clear its "recording..." banner once a capture finishes.
        recording_msg = {
            "active": self.recorder.active,
            "state": self.recorder.state,
            "hint": self.recorder.hint if self.recorder.active else "",
        }

        landmarks = _pose_to_landmarks(pose)
        self._frame_count += 1
        if self._frame_count % _JPEG_EVERY == 0:
            ok_enc, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok_enc:
                landmarks["jpeg"] = base64.b64encode(buf).decode("ascii")

        return {
            "ok": True,
            "landmarks": landmarks,
            "view_state": view_state,
            "prompt": prompt_msg,
            "recording": recording_msg,
        }

    def handle_action(self, action: str, **kw) -> bool:
        """Global actions (mode switching, recording cancel) plus whatever
        the active mode's ``on_action`` understands."""
        ctx = self._ctx
        if ctx is None:
            return False
        if action in ("mode.next", "mode.prev"):
            step = 1 if action == "mode.next" else -1
            self.idx = (self.idx + step) % len(self.modes)
            self.mode = self.modes[self.idx]
            self.mode.activate(ctx)
            self.live.reset()
            return True
        if action == "mode.set":
            name = kw.get("name")
            if name not in MODE_NAMES:
                return False
            self.idx = MODE_NAMES.index(name)
            self.mode = self.modes[self.idx]
            self.mode.activate(ctx)
            self.live.reset()
            return True
        if action == "recording.cancel" and self.recorder.active:
            self.recorder.cancel()
            return True
        handled = self.mode.on_action(action, ctx, **kw)
        if handled and ctx._prompt_request is not None:
            kind, label, initial = ctx._prompt_request
            self.pending_prompt = (self.mode, kind)
        return handled

    def handle_prompt_submit(self, value: str) -> None:
        if self.pending_prompt is None or self._ctx is None:
            return
        mode, kind = self.pending_prompt
        self.pending_prompt = None
        ctx = self._ctx
        ctx._prompt_request = None
        mode.on_prompt(kind, value, ctx)
        if ctx._prompt_request is not None:
            kind2, _, _ = ctx._prompt_request
            self.pending_prompt = (mode, kind2)

    def handle_prompt_cancel(self) -> None:
        if self.pending_prompt is None or self._ctx is None:
            return
        mode, kind = self.pending_prompt
        self.pending_prompt = None
        mode.on_prompt(kind, "", self._ctx)


if __name__ == "__main__":
    eng = Engine()
    try:
        print(f"Engine up - mode {eng.mode_name}, {len(eng.bank)} signs in the bank. "
              "Ctrl+C to stop.")
        for _ in range(5):
            result = eng.step()
            print("ok" if result.get("ok") else "camera read failed",
                  "- view_state keys:", list(result.get("view_state", {}).keys()))
    finally:
        eng.close()
