"""Continuously match the last ~1.5 s of motion against the sign bank.

Keeps a rolling window of poses, extracts a clip on demand, and debounces so
one sign fires once - not every frame it stays recognisable.
"""

from __future__ import annotations

import time
from collections import deque

from signbank import Match, SignBank
from skeleton import SEQ_LEN, Pose, motion_energy, sequence_feature

WINDOW_SECONDS = 1.6
COOLDOWN = 1.0


class LiveSigner:
    def __init__(self, window_seconds: float = WINDOW_SECONDS):
        self.window = window_seconds
        self._buf: deque[tuple[float, Pose]] = deque()
        self._last_fire = 0.0
        self._last_name: str | None = None
        self._empty_since = 0.0

    def push(self, pose: Pose, now: float | None = None) -> None:
        now = time.time() if now is None else now
        self._buf.append((now, pose))
        while self._buf and now - self._buf[0][0] > self.window:
            self._buf.popleft()
        if not pose.has_hands():
            if self._empty_since == 0.0:
                self._empty_since = now
        else:
            self._empty_since = 0.0

    def recent_clip(self):
        poses = [p for _, p in self._buf]
        if len(poses) < 6:
            return None
        return sequence_feature(poses, SEQ_LEN)

    def poll(self, bank: SignBank, now: float | None = None) -> Match | None:
        now = time.time() if now is None else now
        if len(bank) == 0 or len(self._buf) < 8:
            return None
        if now - self._last_fire < COOLDOWN:
            return None
        poses = [p for _, p in self._buf]
        present = sum(p.has_hands() for p in poses) / len(poses)
        if present < 0.6:
            return None
        # allow the same sign again only after the hands have left the frame
        reset = self._empty_since != 0.0 and now - self._empty_since > 0.15

        clip = sequence_feature(poses, SEQ_LEN)
        match = bank.match(clip)
        if match is None:
            return None
        if match.name == self._last_name and not reset:
            return None
        self._last_fire = now
        self._last_name = match.name
        return match

    def reset(self) -> None:
        self._buf.clear()
        self._last_name = None
