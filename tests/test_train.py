import numpy as np
import pytest
import torch

from histoalign.retrieval import recall_at_k
from histoalign.train import Config, Pairs, encode, neighbour_mask, spatial_block_split, train

LATENT, IMAGE_DIM, GENES = 6, 24, 40
_rng = np.random.default_rng(123)
TO_IMAGE = _rng.normal(size=(LATENT, IMAGE_DIM))
TO_EXPR = _rng.normal(size=(LATENT, GENES))


def make_pairs(n, seed, noise=0.3):
    """Image features and expression are two noisy views of the same latent vector."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, LATENT))
    return Pairs(
        image=(z @ TO_IMAGE + noise * rng.normal(size=(n, IMAGE_DIM))).astype(np.float32),
        expr=(z @ TO_EXPR + noise * rng.normal(size=(n, GENES))).astype(np.float32),
        coords=rng.uniform(0, 5000, size=(n, 2)).astype(np.float32),
        slide=np.zeros(n, dtype=np.int64),
    )


FAST = Config(out_dim=16, hidden=64, epochs=40, batch_size=128, lr=3e-3, dropout=0.0)


def held_out_recall(model, pairs, k=5):
    a, e = encode(model, pairs.image, pairs.expr)
    return recall_at_k(a @ e.T, ks=(k,))[k]


def test_training_pairs_up_spots_it_never_saw():
    model, history = train(make_pairs(600, 0), FAST, device="cpu", log=None)
    assert len(history) == FAST.epochs
    assert history[-1] < history[0]
    # 200 candidates, so chance for recall@5 is 2.5 percent
    assert held_out_recall(model, make_pairs(200, 1)) > 0.4


def test_same_seed_same_embeddings():
    pairs = make_pairs(300, 2)
    cfg = Config(out_dim=8, hidden=32, epochs=5, batch_size=64)
    first, _ = train(pairs, cfg, device="cpu", log=None)
    second, _ = train(pairs, cfg, device="cpu", log=None)
    a1, _ = encode(first, pairs.image, pairs.expr)
    a2, _ = encode(second, pairs.image, pairs.expr)
    assert np.array_equal(a1, a2)


def test_sigmoid_loss_also_learns():
    cfg = Config(**{**FAST.__dict__, "loss": "sigmoid"})
    model, _ = train(make_pairs(600, 3), cfg, device="cpu", log=None)
    assert held_out_recall(model, make_pairs(200, 4)) > 0.3


def test_training_with_neighbour_masking_runs():
    pairs = make_pairs(400, 5)
    pairs.coords[:] = np.random.default_rng(0).uniform(0, 300, size=(400, 2))  # all close together
    model, history = train(pairs, FAST, radius_px=100.0, device="cpu", log=None)
    assert np.isfinite(history).all()
    assert held_out_recall(model, make_pairs(200, 6), k=10) > 0.2


def test_unknown_loss_is_refused():
    with pytest.raises(ValueError):
        train(make_pairs(50, 7), Config(loss="triplet"), device="cpu", log=None)


def test_batch_larger_than_the_data_is_shrunk():
    model, history = train(
        make_pairs(40, 8),
        Config(out_dim=4, hidden=8, epochs=2, batch_size=256),
        device="cpu",
        log=None,
    )
    assert len(history) == 2


def test_neighbour_mask_is_same_slide_and_close():
    coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [500.0, 0.0], [10.0, 0.0]])
    slide = torch.tensor([0, 0, 0, 1])
    mask = neighbour_mask(coords, slide, radius_px=20.0)
    assert mask[0, 1] and mask[1, 0]
    assert not mask[0, 2]
    assert not mask[1, 3]  # same place, another slide
    assert torch.equal(mask, mask.T)


def test_spatial_block_split():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 1000, size=(400, 2))
    slide = np.repeat([0, 1], 200)
    train_mask, held = spatial_block_split(coords, slide, frac=0.25, gap_px=100.0)
    assert not (train_mask & held).any()
    for s in (0, 1):
        here = slide == s
        assert held[here].mean() == pytest.approx(0.25, abs=0.02)
        cut = coords[here & held, 0].min()
        # nothing within the gap of the strip is left in training
        assert coords[here & train_mask, 0].max() < cut - 100.0
