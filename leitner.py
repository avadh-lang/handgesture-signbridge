"""A tiny Leitner spaced-repetition scheduler for Coach mode.

Each drill item lives in a box 0..5; a correct answer bumps it up a box, a wrong
one knocks it back to box 0.  Higher boxes come due less often.  State persists
to ``data/leitner.json`` so practice carries across sessions.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

LEITNER_PATH = Path(__file__).parent / "data" / "leitner.json"
# seconds until an item in each box is due again
INTERVALS = [0, 30, 180, 900, 7200, 86400]


class Leitner:
    def __init__(self, path: str | Path = LEITNER_PATH):
        self.path = Path(path)
        self.items: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.items = json.loads(self.path.read_text())
            except (json.JSONDecodeError, OSError):
                self.items = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.items, indent=2))

    def ensure(self, keys: list[str]) -> None:
        changed = False
        for k in keys:
            if k not in self.items:
                self.items[k] = {"box": 0, "due": 0.0, "seen": 0}
                changed = True
        if changed:
            self._save()

    def next_due(self, pool: list[str], now: float | None = None) -> str | None:
        now = time.time() if now is None else now
        self.ensure(pool)
        cand = [k for k in pool if self.items[k]["due"] <= now] or pool
        return min(cand, key=lambda k: (self.items[k]["box"], self.items[k]["due"]))

    def grade(self, key: str, correct: bool, now: float | None = None) -> int:
        now = time.time() if now is None else now
        it = self.items.setdefault(key, {"box": 0, "due": 0.0, "seen": 0})
        it["box"] = min(it["box"] + 1, 5) if correct else 0
        it["due"] = now + INTERVALS[it["box"]]
        it["seen"] = it.get("seen", 0) + 1
        self._save()
        return it["box"]

    def box_counts(self) -> list[int]:
        counts = [0] * 6
        for it in self.items.values():
            counts[min(it["box"], 5)] += 1
        return counts
