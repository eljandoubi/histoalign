import numpy as np
import pytest
import torch
from transformers import ViTConfig, ViTModel

from histoalign.encoders import (
    MEAN,
    STD,
    ClsToken,
    FrozenEncoder,
    embed_batches,
    load_features,
    lunit_vit_small,
    save_features,
    to_tensor,
)


def tiny_vit():
    cfg = ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return ViTModel(cfg, add_pooling_layer=False)


def patches(n, seed=0):
    return np.random.default_rng(seed).integers(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)


@pytest.fixture
def encoder():
    return FrozenEncoder(ClsToken(tiny_vit()), "tiny", image_size=32, device="cpu")


def test_white_patch_after_normalisation():
    x = to_tensor(np.full((1, 2, 2, 3), 255, dtype=np.uint8))
    assert x.shape == (1, 3, 2, 2)
    assert x[0, :, 0, 0].numpy() == pytest.approx((1.0 - MEAN) / STD, rel=1e-5)


def test_embedding_shape_and_no_gradients(encoder):
    emb = encoder.embed(patches(5))
    assert emb.shape == (5, 16)
    assert encoder.image_size == 32
    assert not any(p.requires_grad for p in encoder.model.parameters())


def test_same_input_same_output(encoder):
    x = patches(3)
    assert np.array_equal(encoder.embed(x), encoder.embed(x))


def test_batches_match_one_shot(encoder):
    x = patches(7)
    one_shot = encoder.embed(x)
    batched = embed_batches(encoder, iter(list(x)), batch_size=3)
    assert batched.shape == one_shot.shape
    assert np.allclose(batched, one_shot, atol=1e-5)


def test_cache_round_trip(tmp_path):
    emb = np.random.default_rng(1).normal(size=(4, 6)).astype(np.float32)
    barcodes = ["AAAC-1", "AAAG-1", "AACT-1", "ACGT-1"]
    save_features(tmp_path / "f.h5", barcodes, emb, encoder="tiny", patch_px=32)
    got_barcodes, got, attrs = load_features(tmp_path / "f.h5")
    assert got_barcodes == barcodes
    assert np.array_equal(got, emb)
    assert attrs["encoder"] == "tiny" and attrs["patch_px"] == 32


def test_lunit_trunk_gives_one_vector_per_image():
    model = lunit_vit_small().eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 384)
