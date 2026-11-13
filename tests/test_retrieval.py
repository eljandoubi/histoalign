import numpy as np
import pytest

from histoalign.retrieval import median_rank, near_recall_at_k, ranks, recall_at_k


def test_identity_is_perfect():
    sim = np.eye(6)
    assert recall_at_k(sim, ks=(1, 3)) == {1: 1.0, 3: 1.0}
    assert median_rank(sim) == 1.0


def test_known_ranks():
    sim = np.array(
        [
            [0.9, 0.1, 0.0],  # right answer first
            [0.8, 0.5, 0.0],  # right answer second
            [0.9, 0.8, 0.1],  # right answer last
        ]
    )
    assert ranks(sim).tolist() == [0, 1, 2]
    out = recall_at_k(sim, ks=(1, 2, 3))
    assert out[1] == pytest.approx(1 / 3)
    assert out[2] == pytest.approx(2 / 3)
    assert out[3] == 1.0


def test_random_scores_are_near_chance():
    rng = np.random.default_rng(0)
    n = 500
    out = recall_at_k(rng.normal(size=(n, n)), ks=(1, 10))
    assert out[1] < 0.02
    assert out[10] == pytest.approx(10 / n, abs=0.03)


def test_near_recall_accepts_a_neighbour():
    coords = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [300.0, 0.0]])
    # every query prefers the spot next to it, never itself
    sim = np.array(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    assert recall_at_k(sim, ks=(1,))[1] == 0.0
    # spots 0 to 2 are within 15 px of what they retrieve, spot 3 retrieves one 280 px away
    assert near_recall_at_k(sim, coords, radius=15, ks=(1,))[1] == pytest.approx(0.75)
    assert near_recall_at_k(sim, coords, radius=1000, ks=(1,))[1] == 1.0


def test_near_recall_is_never_below_exact_recall():
    rng = np.random.default_rng(1)
    coords = rng.uniform(0, 1000, size=(50, 2))
    sim = rng.normal(size=(50, 50))
    exact = recall_at_k(sim, ks=(1, 5))
    near = near_recall_at_k(sim, coords, radius=0.0, ks=(1, 5))
    assert near == exact
    assert near_recall_at_k(sim, coords, radius=200.0, ks=(5,))[5] >= exact[5]
