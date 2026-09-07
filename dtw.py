"""Dynamic time warping for comparing two motion clips.

Both clips come in already resampled to the same length (see
``skeleton.sequence_feature``), so DTW here only has to absorb *local* timing
differences - a sign done a bit faster at the start, slower at the end.  A
Sakoe-Chiba band keeps it O(n * band) and stops pathological warps.

Pure NumPy, no SciPy - keeps the dependency list short and the self-test fast.
"""

from __future__ import annotations

import numpy as np

INF = np.inf


def _local_cost(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    """(Ta, Tb) matrix of weighted euclidean distances between frames."""
    diff = a[:, None, :] - b[None, :, :]
    if weights is not None:
        diff = diff * weights
    return np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))


def dtw_distance(
    a: np.ndarray,
    b: np.ndarray,
    band: float = 0.2,
    weights: np.ndarray | None = None,
) -> float:
    """Length-normalised DTW cost between clips ``a`` and ``b`` ((T, D) each).

    ``band`` is the Sakoe-Chiba width as a fraction of the clip length.
    Returns average per-step cost, so it's comparable across clip lengths.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, m = a.shape[0], b.shape[0]
    w = max(int(round(band * max(n, m))), abs(n - m)) + 1

    cost = _local_cost(a, b, weights)
    acc = np.full((n + 1, m + 1), INF)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - w)
        hi = min(m, i + w)
        for j in range(lo, hi + 1):
            acc[i, j] = cost[i - 1, j - 1] + min(
                acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1]
            )
    return float(acc[n, m] / (n + m))


def channel_weights(dim: int, seq_len_dim: int) -> np.ndarray:
    """Down-weight the velocity half of a sequence_feature vector a touch so
    handshape/location dominate the match but movement still counts."""
    w = np.ones(dim)
    if dim == 2 * seq_len_dim:
        w[seq_len_dim:] = 0.6
    return w
