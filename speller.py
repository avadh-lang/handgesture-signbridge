"""Fingerspelling -> words -> a sentence.

A letter is only accepted after it's been held steady for a beat (like waiting
for a slow typist to settle), and the same letter twice in a row needs the hand
to leave the shape in between - otherwise a held "L" becomes "LLLLL".

Word prediction (``predict.Predictor``) offers completions you can accept so
common words don't have to be spelled out in full.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from predict import Predictor, default_predictor

COMMIT_SECONDS = 0.55       # how long a letter must be held to register
RELEASE_SECONDS = 0.20      # gap that counts as "left the handshape"
MIN_CONFIDENCE = 0.12


@dataclass
class Speller:
    predictor: Predictor = field(default_factory=default_predictor)
    commit_seconds: float = COMMIT_SECONDS
    min_confidence: float = MIN_CONFIDENCE

    word: str = ""
    sentence: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    last_committed: str | None = None

    _pending: str | None = None
    _pending_since: float = 0.0
    _released: bool = True
    _idle_since: float | None = None

    # ---- input ------------------------------------------------------
    def feed_letter(self, letter: str | None, confidence: float, now: float | None = None) -> bool:
        """Feed the current best letter (or None if no clear handshape).
        Returns True on the frame a letter is actually committed."""
        now = time.time() if now is None else now

        if letter is None or confidence < self.min_confidence:
            if self._idle_since is None:
                self._idle_since = now
            if now - self._idle_since >= RELEASE_SECONDS:
                self._released = True
                self._pending = None
            return False
        self._idle_since = None

        if letter != self._pending:
            self._pending = letter
            self._pending_since = now
            return False

        if now - self._pending_since < self.commit_seconds:
            return False
        if letter == self.last_committed and not self._released:
            return False

        self._append(letter)
        self.last_committed = letter
        self._released = False
        self._pending_since = float("inf")   # block a repeat until the hand releases
        return True

    def hold_fraction(self, now: float | None = None) -> float:
        """0..1 progress of the letter currently being held toward commit."""
        if self._pending is None or self._pending_since == float("inf"):
            return 0.0
        now = time.time() if now is None else now
        return max(0.0, min(1.0, (now - self._pending_since) / self.commit_seconds))

    def _append(self, letter: str) -> None:
        self.word += letter.lower()
        self._refresh()

    # ---- edits ------------------------------------------------------
    def space(self) -> None:
        if self.word:
            self.sentence.append(self.word)
            self.word = ""
        elif self.sentence and self.sentence[-1] != "":
            pass
        self.last_committed = None
        self._released = True
        self._refresh()

    def accept(self, index: int) -> bool:
        if 0 <= index < len(self.suggestions):
            self.sentence.append(self.suggestions[index])
            self.word = ""
            self.last_committed = None
            self._released = True
            self._refresh()
            return True
        return False

    def backspace(self) -> None:
        if self.word:
            self.word = self.word[:-1]
        elif self.sentence:
            self.word = self.sentence.pop()
            self.word = self.word[:-1] if self.word else ""
        self.last_committed = None
        self._released = True
        self._refresh()

    def clear(self) -> None:
        self.word = ""
        self.sentence.clear()
        self.suggestions.clear()
        self.last_committed = None
        self._released = True

    def pop_sentence(self) -> str:
        text = self.text()
        self.clear()
        return text

    # ---- output ---------------------------------------------------
    def text(self) -> str:
        parts = list(self.sentence) + ([self.word] if self.word else [])
        return " ".join(parts)

    def _refresh(self) -> None:
        self.suggestions = self.predictor.complete(self.word, k=3) if self.word else []
