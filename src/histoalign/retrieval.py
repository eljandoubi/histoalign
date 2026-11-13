"""Retrieval metrics for two sets of paired embeddings."""

import numpy as np


def ranks(sim: np.ndarray) -> np.ndarray:
    """Position of the right candidate for each query, 0 is the best. `sim[i, j]` scores query i
    against candidate j and the right answer for query i is candidate i."""
    order = np.argsort(-sim, axis=1, kind="stable")
    return (order == np.arange(len(sim))[:, None]).argmax(axis=1)


def recall_at_k(sim: np.ndarray, ks: tuple[int, ...] = (1, 5, 10)) -> dict[int, float]:
    r = ranks(sim)
    return {k: float((r < k).mean()) for k in ks}


def near_recall_at_k(
    sim: np.ndarray, coords: np.ndarray, radius: float, ks: tuple[int, ...] = (1, 5, 10)
) -> dict[int, float]:
    """Like recall_at_k, but a hit is any of the top k candidates lying within `radius` of the
    right one. Neighbouring spots have close to the same tissue, so the exact spot is a hard target.
    """
    order = np.argsort(-sim, axis=1, kind="stable")
    out = {}
    for k in ks:
        top = order[:, :k]
        dist = np.linalg.norm(coords[top] - coords[:, None, :], axis=-1)
        out[k] = float((dist <= radius).any(axis=1).mean())
    return out


def median_rank(sim: np.ndarray) -> float:
    return float(np.median(ranks(sim)) + 1)
