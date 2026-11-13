import numpy as np
import pytest
import torch

from histoalign.losses import info_nce
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
    model, info = train(make_pairs(600, 0), FAST, device="cpu", log=None)
    assert len(info["loss"]) == FAST.epochs
    assert info["loss"][-1] < info["loss"][0]
    assert info["best_epoch"] == FAST.epochs and info["val_loss"] == []
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
    model, info = train(pairs, FAST, radius_px=100.0, device="cpu", log=None)
    assert np.isfinite(info["loss"]).all()
    assert held_out_recall(model, make_pairs(200, 6), k=10) > 0.2


def test_unknown_loss_is_refused():
    with pytest.raises(ValueError):
        train(make_pairs(50, 7), Config(loss="triplet"), device="cpu", log=None)


def test_batch_larger_than_the_data_is_shrunk():
    model, info = train(
        make_pairs(40, 8),
        Config(out_dim=4, hidden=8, epochs=2, batch_size=256),
        device="cpu",
        log=None,
    )
    assert len(info["loss"]) == 2


def test_validation_stops_training_and_puts_back_the_best_epoch():
    # few pairs and a lot of noise: the heads learn the training spots by heart
    cfg = Config(
        out_dim=16, hidden=128, epochs=150, batch_size=30, lr=5e-3, dropout=0.0, patience=5
    )
    val = make_pairs(200, 10, noise=1.5)
    model, info = train(make_pairs(60, 9, noise=1.5), cfg, device="cpu", log=None, val=val)
    assert len(info["loss"]) < cfg.epochs
    assert len(info["val_loss"]) == len(info["loss"])
    assert info["best_epoch"] == int(np.argmin(info["val_loss"])) + 1
    # what comes back is the model of that epoch, not the last one
    a, e = encode(model, val.image, val.expr)
    back = info_nce(torch.from_numpy(a), torch.from_numpy(e), model.scale.detach()).item()
    assert back == pytest.approx(min(info["val_loss"]), abs=1e-4)
    assert back < info["val_loss"][-1]


def test_validation_with_neighbour_masking():
    pairs = make_pairs(300, 11)
    val = make_pairs(100, 12)
    pairs.coords[:] = val.coords[:1]  # everything on one point, so everything is a neighbour
    val.coords[:] = val.coords[:1]
    cfg = Config(out_dim=8, hidden=16, epochs=6, batch_size=64, patience=2)
    _, info = train(pairs, cfg, radius_px=10.0, device="cpu", log=None, val=val)
    assert np.isfinite(info["val_loss"]).all()


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
