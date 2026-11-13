import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from histoalign.encoders import save_features
from histoalign.experiment import Prepared, block_subsets, pool, prepare, run
from histoalign.genes import log_normalise
from histoalign.train import Config
from tests.slides import SAMPLE

N_GENES, LATENT, IMAGE_DIM = 120, 5, 20
_rng = np.random.default_rng(7)
LOADING = _rng.normal(size=(LATENT, N_GENES))
TO_IMAGE = _rng.normal(size=(LATENT, IMAGE_DIM))
PITCH = 100.0

FAST = Config(out_dim=16, hidden=64, epochs=30, batch_size=64, lr=3e-3, dropout=0.0)


def synthetic_slide(name, seed, n=300, quiet_gene=None):
    """Counts and image features driven by the same latent variables."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, LATENT))
    rate = np.exp(0.8 + 0.5 * (z @ LOADING) / np.sqrt(LATENT))
    counts = rng.poisson(rate).astype(np.float32)
    adata = ad.AnnData(sparse.csr_matrix(counts))
    adata.var_names = [f"G{i}" for i in range(N_GENES)]
    adata.obs_names = [f"{name}_{i}" for i in range(n)]
    image = (z @ TO_IMAGE + 0.3 * rng.normal(size=(n, IMAGE_DIM))).astype(np.float32)
    coords = rng.uniform(0, 5000, size=(n, 2))
    return Prepared(name, log_normalise(adata), image, coords, PITCH)


def everything(p):
    return p, np.arange(len(p.coords))


def test_scores_beat_the_shuffled_control():
    train = [synthetic_slide("a", 1), synthetic_slide("b", 2)]
    test = synthetic_slide("c", 3, n=200)
    out = run([everything(p) for p in train], everything(test), FAST, n_genes=60, k=10, log=None)
    imp = out["imputation"]
    assert imp["ridge"]["pcc_all"] > 0.2
    assert imp["aligned_knn"]["pcc_all"] > 0.15
    assert imp["aligned_knn"]["pcc_all"] > imp["aligned_knn_shuffled_pairs"]["pcc_all"] + 0.1
    assert abs(imp["aligned_knn_shuffled_pairs"]["pcc_all"]) < 0.1
    # retrieval of the exact spot among 200 is far above chance
    assert (
        out["retrieval"]["image_to_expr"]["recall@10"] > 5 * out["retrieval"]["chance"]["recall@10"]
    )


def test_result_describes_the_split():
    train = [synthetic_slide("a", 4)]
    test = synthetic_slide("b", 5, n=150)
    out = run(
        [everything(p) for p in train], everything(test), FAST, n_genes=30, control=False, log=None
    )
    assert out["train"]["slides"] == ["a"] and out["train"]["spots"] == 300
    assert 0 < out["train"]["val_spots"] < out["train"]["fit_spots"]
    assert out["test"] == {"slide": "b", "spots": 150}
    assert "aligned_knn_shuffled_pairs" not in out["imputation"]
    assert len(out["genes"]) == 30


def test_genes_are_chosen_on_training_slides_only():
    train = [synthetic_slide("a", 6)]
    test = synthetic_slide("b", 7, n=150)
    # a gene that is wildly variable in the test slide only
    x = test.adata.X.toarray()
    x[:, 0] = np.where(np.arange(len(x)) % 2, 8.0, 0.0)
    test.adata.X = sparse.csr_matrix(x)
    out = run(
        [everything(p) for p in train], everything(test), FAST, n_genes=20, control=False, log=None
    )
    assert "G0" not in out["genes"]


def test_markers_are_kept_when_present_everywhere():
    train = [synthetic_slide("a", 8)]
    test = synthetic_slide("b", 9, n=100)
    for p in (*train, test):
        names = list(p.adata.var_names)
        names[119] = "ERBB2"
        p.adata.var_names = names
    out = run(
        [everything(p) for p in train],
        everything(test),
        FAST,
        n_genes=5,
        markers=("ERBB2", "NOT_THERE"),
        control=False,
        log=None,
    )
    assert "ERBB2" in out["genes"] and "NOT_THERE" not in out["genes"]
    assert "pcc_ERBB2" in out["imputation"]["ridge"]


def test_block_subsets_do_not_touch():
    p = synthetic_slide("a", 10, n=400)
    (_, train_idx), (_, test_idx) = block_subsets(p, frac=0.25, gap=3.0)
    assert not set(train_idx) & set(test_idx)
    gap = 3.0 * p.pitch_px
    assert p.coords[train_idx, 0].max() < p.coords[test_idx, 0].min() - gap
    assert len(test_idx) == pytest.approx(100, abs=5)


def test_prepare_reads_slide_and_features(fake_slide):
    from histoalign.visium import read_slide

    slide = read_slide(fake_slide.root)
    feats = np.arange(len(slide.adata) * 4, dtype=np.float32).reshape(-1, 4)
    out_dir = fake_slide.root.parent / "feat"
    out_dir.mkdir()
    save_features(out_dir / f"{SAMPLE}.h5", list(slide.adata.obs_names), feats)
    p = prepare(fake_slide.root, out_dir)
    assert p.name == SAMPLE and p.image.shape == (9, 4) and p.coords.shape == (9, 2)
    assert p.pitch_px == pytest.approx(slide.pitch_px)

    # features written in another order are refused
    save_features(out_dir / f"{SAMPLE}.h5", list(slide.adata.obs_names)[::-1], feats)
    with pytest.raises(ValueError):
        prepare(fake_slide.root, out_dir)


def test_validation_strip_picks_the_epochs_and_stays_out_of_the_fit():
    train = [synthetic_slide("a", 11), synthetic_slide("b", 12)]
    test = synthetic_slide("c", 13, n=100)
    cfg = Config(out_dim=16, hidden=64, epochs=80, batch_size=64, lr=3e-3, dropout=0.0, patience=4)
    out = run(
        [everything(p) for p in train],
        everything(test),
        cfg,
        n_genes=40,
        control=False,
        val_frac=0.2,
        log=None,
    )
    seen = out["train"]
    assert seen["val_spots"] > 0.15 * seen["spots"]
    # the spots between the strip and the rest are in neither set
    assert seen["fit_spots"] + seen["val_spots"] < seen["spots"]
    assert 1 <= out["epochs"] <= 80


def test_no_validation_when_the_strip_is_zero():
    train = [synthetic_slide("a", 14)]
    test = synthetic_slide("b", 15, n=100)
    out = run(
        [everything(p) for p in train],
        everything(test),
        FAST,
        n_genes=30,
        val_frac=0,
        log=None,
    )
    assert out["train"]["val_spots"] == 0 and out["train"]["fit_spots"] == 300
    assert out["epochs"] == FAST.epochs
    # the control still runs, on all the spots
    assert "aligned_knn_shuffled_pairs" in out["imputation"]


def test_pooled_slides_keep_their_spots_apart():
    a, b = synthetic_slide("a", 20, n=50), synthetic_slide("b", 21, n=40)
    b.adata.obs_names = a.adata.obs_names[:40]
    both = pool([a, b])
    assert both.name == "a+b" and both.adata.n_obs == 90
    assert both.adata.obs_names.is_unique
    assert np.array_equal(both.image[50:], b.image)
    assert np.array_equal(both.coords[:50], a.coords)
    apart = np.linalg.norm(both.coords[:50, None] - both.coords[None, 50:], axis=2).min()
    assert apart > 1000 * PITCH


def test_pooling_fills_a_missing_gene_with_zeros():
    a, b = synthetic_slide("a", 22, n=30), synthetic_slide("b", 23, n=30)
    b.adata = b.adata[:, 1:].copy()
    g0 = pool([a, b]).adata[:, "G0"].X.toarray().ravel()
    assert (g0[:30] > 0).any() and (g0[30:] == 0).all()


def test_a_pooled_test_set_is_scored_like_one_slide():
    train = [synthetic_slide("a", 24)]
    test = pool([synthetic_slide("b", 25, n=60), synthetic_slide("c", 26, n=70)], "patient X")
    out = run([everything(p) for p in train], everything(test), FAST, n_genes=30, log=None)
    assert out["test"] == {"slide": "patient X", "spots": 130}
