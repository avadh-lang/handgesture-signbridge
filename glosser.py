"""English sentence -> ASL-gloss cards for the hearing-to-Deaf direction.

This is *not* a grammatical ASL translator - that needs a real corpus.  It's a
fast, offline gloss: drop the words ASL doesn't use (articles, copula,
infinitive "to"), expand contractions, uppercase, and mark anything unknown for
fingerspelling.  Good enough to hold up cards the other person can read while
you speak.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from predict import default_predictor

DROP = {
    "a", "an", "the", "is", "am", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "of", "to", "will", "would", "shall",
}
CONTRACTIONS = {
    "i'm": "i", "you're": "you", "he's": "he", "she's": "she", "it's": "it",
    "we're": "we", "they're": "they", "don't": "not", "doesn't": "not",
    "didn't": "not", "can't": "can not", "won't": "will not", "i'll": "i",
    "you'll": "you", "i've": "i", "you've": "you", "let's": "let us",
    "that's": "that", "what's": "what", "there's": "there",
}
PRONOUN_POINT = {"i": "ME", "me": "ME", "you": "YOU", "my": "MY", "your": "YOUR"}


@dataclass
class Card:
    text: str
    kind: str            # "sign" | "fingerspell" | "pause"

    @property
    def letters(self) -> list[str]:
        return list(self.text) if self.kind == "fingerspell" else []


def _normalise(sentence: str) -> list[str]:
    out: list[str] = []
    for raw in sentence.lower().split():
        raw = CONTRACTIONS.get(raw, raw)
        out.extend(raw.split())
    return out


def gloss(sentence: str) -> list[Card]:
    pred = default_predictor()
    cards: list[Card] = []
    for token in _normalise(sentence):
        word = re.sub(r"[^a-z0-9']", "", token)
        trailing = token[len(token.rstrip(".!?,;:")):]
        if word and word not in DROP:
            if word in PRONOUN_POINT:
                cards.append(Card(PRONOUN_POINT[word], "sign"))
            elif word.isalpha() and not pred.is_word(word):
                cards.append(Card(word.upper(), "fingerspell"))
            else:
                cards.append(Card(word.upper(), "sign"))
        if any(p in trailing for p in ".!?"):
            cards.append(Card("•", "pause"))
    return cards


def gloss_line(sentence: str) -> str:
    return " ".join(c.text for c in gloss(sentence) if c.kind != "pause")
