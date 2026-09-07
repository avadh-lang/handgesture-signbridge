"""Offline word prediction for fingerspelling.

``data/words_en.txt`` is ~20k English words, most-common first (line number is
the rank).  Given the letters signed so far we offer the most likely
completions so the signer can pick one instead of spelling every character.

No model, no network - just a ranked prefix lookup over a bundled list.
"""

from __future__ import annotations

import bisect
from functools import lru_cache
from pathlib import Path

WORDS_FILE = Path(__file__).parent / "data" / "words_en.txt"


class Predictor:
    def __init__(self, words_file: str | Path = WORDS_FILE):
        text = Path(words_file).read_text().split()
        self._rank = {w: i for i, w in enumerate(text)}
        self._sorted = sorted(self._rank)          # for prefix scan

    def _prefix_slice(self, prefix: str) -> list[str]:
        lo = bisect.bisect_left(self._sorted, prefix)
        hi = bisect.bisect_left(self._sorted, prefix + "￿")
        return self._sorted[lo:hi]

    def complete(self, prefix: str, k: int = 3) -> list[str]:
        """Up to ``k`` likely words starting with ``prefix`` (case-insensitive)."""
        prefix = prefix.strip().lower()
        if not prefix:
            return []
        cands = self._prefix_slice(prefix)
        cands.sort(key=lambda w: (self._rank.get(w, 10**9), len(w), w))
        # keep the exact word first if it's real
        out = [w for w in cands if w != prefix][:k]
        if prefix in self._rank and prefix not in out:
            out = [prefix] + out[: k - 1]
        return out

    def is_word(self, w: str) -> bool:
        return w.strip().lower() in self._rank


@lru_cache(maxsize=1)
def default_predictor() -> Predictor:
    return Predictor()
