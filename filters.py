"""Signal smoothing / debouncing shared by the pointer and the modes.

The One-Euro filter (Casiez, Roussel & Vogel 2012) is the good bit borrowed
from Prathamesh-2803/Real-time-hand-gesture-control-system - it keeps a slow
signal rock-steady while staying responsive the instant it starts moving, which
is exactly what a fingertip cursor needs.

Pure Python, parameter-driven (no global config), camera-free -> unit-tested.
"""

from __future__ import annotations

import math


class OneEuroFilter:
    """Adaptive low-pass filter for one scalar stream.

        min_cutoff  lower  -> smoother when the signal is still
        beta        higher -> less lag when the signal moves fast
        d_cutoff    cutoff for the derivative estimate (1 Hz is usually fine)
    """

    def __init__(self, freq: float = 60.0, min_cutoff: float = 1.0,
                 beta: float = 0.02, d_cutoff: float = 1.0):
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev = 0.0

    def _alpha(self, cutoff: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: float, dt: float | None = None) -> float:
        x = float(x)
        if dt and dt > 0:
            self.freq = 1.0 / dt
        if self._x_prev is None:
            self._x_prev = x
            return x
        dx = (x - self._x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev, self._dx_prev = x_hat, dx_hat
        return x_hat

    def reset(self, x: float | None = None) -> None:
        self._x_prev = None if x is None else float(x)
        self._dx_prev = 0.0


class OneEuroPoint:
    """Two One-Euro filters for an (x, y) point."""

    def __init__(self, **kw):
        self.fx = OneEuroFilter(**kw)
        self.fy = OneEuroFilter(**kw)

    def __call__(self, x: float, y: float, dt: float | None = None) -> tuple[float, float]:
        return self.fx(x, dt), self.fy(y, dt)

    def reset(self, x: float | None = None, y: float | None = None) -> None:
        self.fx.reset(x)
        self.fy.reset(y)


class Hysteresis:
    """Boolean debounce: flips to True only after ``enter`` consecutive True
    inputs, back to False only after ``exit_`` consecutive False inputs."""

    def __init__(self, enter: int = 3, exit_: int = 3, state: bool = False):
        self.enter = enter
        self.exit_ = exit_
        self.state = state
        self._count = 0

    def update(self, signal: bool) -> bool:
        target = not self.state
        need = self.enter if target else self.exit_
        if bool(signal) == target:
            self._count += 1
            if self._count >= need:
                self.state = target
                self._count = 0
        else:
            self._count = 0
        return self.state

    def reset(self, state: bool = False) -> None:
        self.state = state
        self._count = 0
