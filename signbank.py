"""A personal dictionary of signs you taught the app yourself.

You demonstrate a sign a few times; each demo becomes a fixed-length feature
clip (``skeleton.sequence_feature``).  Recognition is nearest-template with a
rejection rule: the closest exemplar wins only if it's *close enough* and
clearly closer than the runner-up.

No training, no dataset, no network.  A sign is added in seconds and is usable
on the very next frame.  Works for a still handshape and for a travelling,
two-handed, body-relative sign alike - DTW doesn't care which.

On disk: one ``<slug>.npz`` per sign (stacked exemplars) + an ``index.json``
manifest, under the bank directory (default ``signs/``).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from dtw import channel_weights, dtw_distance
from skeleton import FRAME_FEATURE_DIM

MAX_EXEMPLARS = 6
DEFAULT_DIR = Path(__file__).parent / "signs"

# distance below which a match is trusted; tune with `learn_sign.py --check`
REJECT_DISTANCE = 0.55
# best must be at most this fraction of the runner-up to be unambiguous
AMBIGUITY_RATIO = 0.85


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "sign"


@dataclass
class SignMeta:
    name: str
    slug: str
    kind: str = "dynamic"       # "dynamic" | "static"
    hands: str = "any"          # "left" | "right" | "both" | "any"
    reps: int = 0
    created: float = 0.0
    action: dict | None = None  # used by control mode: {"type": ..., "value": ...}


@dataclass
class Match:
    name: str
    slug: str
    distance: float
    margin: float               # runner-up distance - best distance
    kind: str
    meta: SignMeta


class SignBank:
    def __init__(self, directory: str | Path = DEFAULT_DIR):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, SignMeta] = {}
        self._clips: dict[str, np.ndarray] = {}      # slug -> (K, L, D)
        self._weights: np.ndarray | None = None
        self._load()

    # ---- persistence ----------------------------------------------------
    @property
    def _index_path(self) -> Path:
        return self.dir / "index.json"

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        raw = json.loads(self._index_path.read_text())
        for entry in raw.get("signs", []):
            meta = SignMeta(**entry)
            npz = self.dir / f"{meta.slug}.npz"
            if not npz.exists():
                continue
            self._meta[meta.slug] = meta
            self._clips[meta.slug] = np.load(npz)["exemplars"]

    def _save_index(self) -> None:
        self._index_path.write_text(json.dumps(
            {"signs": [asdict(m) for m in self._meta.values()]}, indent=2
        ))

    # ---- editing ------------------------------------------------------
    def add(
        self,
        name: str,
        clips: list[np.ndarray],
        kind: str = "dynamic",
        hands: str = "any",
        action: dict | None = None,
    ) -> SignMeta:
        if not clips:
            raise ValueError("need at least one demonstration clip")
        slug = slugify(name)
        stack = np.stack([np.asarray(c, dtype=np.float32) for c in clips])
        if slug in self._clips:
            stack = np.concatenate([self._clips[slug], stack])
        stack = stack[-MAX_EXEMPLARS:]
        self._clips[slug] = stack
        meta = self._meta.get(slug) or SignMeta(name=name, slug=slug, created=time.time())
        meta.name, meta.kind, meta.hands, meta.reps = name, kind, hands, stack.shape[0]
        if action is not None:
            meta.action = action
        self._meta[slug] = meta
        np.savez_compressed(self.dir / f"{slug}.npz", exemplars=stack)
        self._save_index()
        return meta

    def set_action(self, slug: str, action: dict | None) -> None:
        if slug in self._meta:
            self._meta[slug].action = action
            self._save_index()

    def remove(self, name_or_slug: str) -> bool:
        slug = name_or_slug if name_or_slug in self._meta else slugify(name_or_slug)
        if slug not in self._meta:
            return False
        self._meta.pop(slug)
        self._clips.pop(slug, None)
        (self.dir / f"{slug}.npz").unlink(missing_ok=True)
        self._save_index()
        return True

    # ---- query ------------------------------------------------------
    def __len__(self) -> int:
        return len(self._meta)

    def names(self) -> list[str]:
        return [m.name for m in self._meta.values()]

    def metas(self) -> list[SignMeta]:
        return list(self._meta.values())

    def get(self, name_or_slug: str) -> SignMeta | None:
        return self._meta.get(name_or_slug) or self._meta.get(slugify(name_or_slug))

    def _w(self, dim: int) -> np.ndarray:
        if self._weights is None or self._weights.shape[0] != dim:
            self._weights = channel_weights(dim, FRAME_FEATURE_DIM)
        return self._weights

    def distances(self, clip: np.ndarray, band: float = 0.22) -> list[tuple[str, float]]:
        """(slug, best-exemplar distance) for every sign, closest first."""
        clip = np.asarray(clip, dtype=np.float64)
        w = self._w(clip.shape[1])
        out = []
        for slug, stack in self._clips.items():
            d = min(dtw_distance(clip, ex, band=band, weights=w) for ex in stack)
            out.append((slug, d))
        out.sort(key=lambda t: t[1])
        return out

    def match(
        self,
        clip: np.ndarray,
        reject_distance: float = REJECT_DISTANCE,
        ambiguity_ratio: float = AMBIGUITY_RATIO,
        band: float = 0.22,
    ) -> Match | None:
        ranked = self.distances(clip, band=band)
        if not ranked:
            return None
        best_slug, best_d = ranked[0]
        second_d = ranked[1][1] if len(ranked) > 1 else np.inf
        if best_d > reject_distance:
            return None
        if second_d < np.inf and best_d > ambiguity_ratio * second_d:
            return None
        meta = self._meta[best_slug]
        return Match(meta.name, best_slug, best_d,
                     float(second_d - best_d), meta.kind, meta)
