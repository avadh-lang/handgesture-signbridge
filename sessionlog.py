"""Append-only CSV log of what happened in a session.

Borrowed idea from Prathamesh-2803's gesture logger, repurposed for practice
analytics: Coach writes one row per attempt so you can see, later, which
letters and signs you actually struggle with.

    ts, kind, target, detail, score, result
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

DEFAULT_PATH = Path(__file__).parent / "data" / "practice_log.csv"
_HEADER = ["ts", "kind", "target", "detail", "score", "result"]


class SessionLog:
    def __init__(self, path: str | Path = DEFAULT_PATH, enabled: bool = True):
        self.path = Path(path)
        self.enabled = enabled
        if enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if not self.path.exists():
                with self.path.open("w", newline="") as fh:
                    csv.writer(fh).writerow(_HEADER)

    def log(self, kind: str, target: str = "", detail: str = "",
            score: float | str = "", result: str = "") -> None:
        if not self.enabled:
            return
        row = [f"{time.time():.3f}", kind, target, detail,
               f"{score:.3f}" if isinstance(score, float) else score, result]
        try:
            with self.path.open("a", newline="") as fh:
                csv.writer(fh).writerow(row)
        except OSError:
            pass

    def summary(self) -> dict[str, dict]:
        """Per-target {attempts, passes, last_score} from the log so far."""
        out: dict[str, dict] = {}
        if not self.path.exists():
            return out
        with self.path.open(newline="") as fh:
            for r in csv.DictReader(fh):
                t = r.get("target") or r.get("kind", "?")
                d = out.setdefault(t, {"attempts": 0, "passes": 0, "last_score": None})
                d["attempts"] += 1
                d["passes"] += int(r.get("result") == "pass")
                try:
                    d["last_score"] = float(r["score"])
                except (ValueError, KeyError, TypeError):
                    pass
        return out
