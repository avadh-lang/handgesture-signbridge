"""Non-blocking 'demonstrate a sign N times' capture, driven from the app loop.

Call ``update(pose, now)`` once per frame; read ``state`` / ``hint`` to draw the
overlay.  When ``state == "done"`` pull ``result()`` - a dict of fixed-length
feature clips ready for ``SignBank.add``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from skeleton import SEQ_LEN, Pose, motion_energy, sequence_feature

COUNTDOWN = 1.4
RECORD_SECONDS = 1.6
STATIC_MOTION = 0.16        # below this a clip is treated as a still handshape


@dataclass
class SignRecorder:
    name: str = ""
    reps_wanted: int = 3
    state: str = "idle"        # idle | counting | recording | done | cancelled
    rep: int = 0
    hint: str = ""

    _t0: float = 0.0
    _buf: list = field(default_factory=list)
    _clips: list = field(default_factory=list)
    _motion: list = field(default_factory=list)
    _hands_seen: set = field(default_factory=set)

    def start(self, name: str, reps: int = 3) -> None:
        self.name, self.reps_wanted = name, max(1, reps)
        self.state, self.rep = "counting", 0
        self._buf, self._clips, self._motion = [], [], []
        self._hands_seen = set()
        self._t0 = None
        self.hint = f'get ready to sign "{name}"'

    def cancel(self) -> None:
        self.state = "cancelled"
        self.hint = "cancelled"

    @property
    def active(self) -> bool:
        return self.state in ("counting", "recording")

    def update(self, pose: Pose, now: float) -> None:
        if not self.active:
            return
        if self._t0 is None:
            self._t0 = now
        elapsed = now - self._t0

        if self.state == "counting":
            left = COUNTDOWN - elapsed
            self.hint = f'rep {self.rep + 1}/{self.reps_wanted}  -  {max(0, int(left) + 1)}'
            if left <= 0:
                self.state, self._t0, self._buf = "recording", now, []
            return

        # recording
        self._buf.append(pose)
        if pose.left is not None:
            self._hands_seen.add("left")
        if pose.right is not None:
            self._hands_seen.add("right")
        self.hint = f'rep {self.rep + 1}/{self.reps_wanted}  -  signing...'
        if elapsed >= RECORD_SECONDS:
            self._finish_rep(now)

    def _finish_rep(self, now: float) -> None:
        poses = [p for p in self._buf if p.has_hands()] or self._buf
        if len(poses) >= 3:
            self._clips.append(sequence_feature(poses, SEQ_LEN))
            self._motion.append(motion_energy(poses))
            self.rep += 1
        else:
            self.hint = "no hand seen - try again"
        if self.rep >= self.reps_wanted:
            self.state = "done"
            self.hint = f'"{self.name}" learned'
        else:
            self.state, self._t0 = "counting", now

    def result(self) -> dict:
        kind = "static" if (self._motion and np.mean(self._motion) < STATIC_MOTION) else "dynamic"
        if self._hands_seen == {"left", "right"}:
            hands = "both"
        elif self._hands_seen:
            hands = next(iter(self._hands_seen))
        else:
            hands = "any"
        return {"name": self.name, "kind": kind, "hands": hands, "clips": self._clips}
