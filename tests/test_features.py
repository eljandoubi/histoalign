import numpy as np
import pandas as pd
from transformers import ViTConfig, ViTModel

from histoalign.encoders import FrozenEncoder, load_features
from histoalign.features import extract, spot_table
from histoalign.visium import read_slide
from tests.slides import GRID


def tiny_encoder():
    cfg = ViTConfig(
        image_size=32,
        patch_size=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    return FrozenEncoder(name="tiny", model=ViTModel(cfg, add_pooling_layer=False), device="cpu")


def test_spot_table_columns(fake_slide):
    slide = read_slide(fake_slide.root)
    table = spot_table(slide)
    assert list(table.columns) == ["barcode", "slide", "array_row", "array_col", "x", "y"]
    assert len(table) == GRID * GRID
    assert np.array_equal(table[["x", "y"]].to_numpy(), slide.adata.obsm["spatial"])


def test_extract_writes_features_and_table(fake_slide, tmp_path):
    slide = read_slide(fake_slide.root)
    path = extract(slide, tiny_encoder(), tmp_path / "feat", batch_size=4)
    barcodes, emb, attrs = load_features(path)
    assert barcodes == list(slide.adata.obs_names)
    assert emb.shape == (GRID * GRID, 16)
    assert attrs["encoder"] == "tiny" and attrs["source"] == "tiff"
    assert attrs["mpp"] == slide.mpp

    table = pd.read_parquet(path.with_suffix(".parquet"))
    assert table["barcode"].tolist() == barcodes


def test_different_spots_get_different_features(fake_slide, tmp_path):
    slide = read_slide(fake_slide.root)
    _, emb, _ = load_features(extract(slide, tiny_encoder(), tmp_path))
    # the fake image changes with position, so two spots cannot share a patch
    assert not np.allclose(emb[0], emb[4])
