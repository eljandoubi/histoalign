import anndata as ad
import numpy as np
import pytest
from scipy import sparse

from histoalign.genes import Scaler, log_normalise, select_genes, take_genes


def slide(seed, n_spots=300, n_genes=120, n_variable=15, depth=1.0):
    """Counts where the first genes are over-dispersed: the ones a variance filter should find."""
    rng = np.random.default_rng(seed)
    mean = rng.uniform(1.0, 8.0, size=n_genes) * depth
    counts = rng.poisson(mean, size=(n_spots, n_genes)).astype(np.float32)
    boost = rng.gamma(0.5, 2.0, size=(n_spots, n_variable))
    counts[:, :n_variable] = rng.poisson(mean[:n_variable] * boost)
    adata = ad.AnnData(sparse.csr_matrix(counts))
    adata.var_names = [f"G{i}" for i in range(n_genes)]
    adata.obs_names = [f"s{seed}_{i}" for i in range(n_spots)]
    return adata


def test_log_normalise_keeps_the_counts():
    raw = slide(0)
    out = log_normalise(raw)
    assert np.array_equal(out.layers["counts"].toarray(), raw.X.toarray())
    # after normalise_total every spot sums to the target, undoing the log gives it back
    sums = np.expm1(out.X.toarray()).sum(axis=1)
    assert sums == pytest.approx(np.full(raw.n_obs, 1e4), rel=1e-3)
    # the input is untouched
    assert raw.X.max() > 20


def test_select_genes_finds_the_planted_ones():
    train = [log_normalise(slide(1)), log_normalise(slide(2))]
    genes = select_genes(train, n_top=15, min_frac=0.0)
    assert len(genes) == 15
    planted = {f"G{i}" for i in range(15)}
    assert len(planted & set(genes)) >= 11


def test_select_genes_works_with_one_slide():
    genes = select_genes([log_normalise(slide(3))], n_top=10, min_frac=0.0)
    assert len(genes) == 10


def test_rare_genes_are_dropped():
    adata = slide(4)
    x = adata.X.toarray()
    x[:, 0] = 0
    x[:3, 0] = 50  # seen in 1% of the spots only
    adata.X = sparse.csr_matrix(x)
    genes = select_genes([log_normalise(adata)], n_top=20, min_frac=0.05)
    assert "G0" not in genes


def test_missing_genes_are_zero():
    adata = slide(5, n_genes=10, n_variable=2)
    out = take_genes(adata, ["G1", "NOPE", "G3"])
    assert out.shape == (adata.n_obs, 3)
    assert (out[:, 1] == 0).all()
    assert np.array_equal(out[:, 0], adata.X[:, 1].toarray().ravel())


def test_scaler_is_fitted_on_the_training_slides():
    train = [log_normalise(slide(6)), log_normalise(slide(7))]
    genes = [f"G{i}" for i in range(20)]
    scaler = Scaler(genes).fit(train)
    z = np.vstack([scaler.transform(a) for a in train])
    assert z.mean(axis=0) == pytest.approx(np.zeros(20), abs=1e-4)
    assert z.std(axis=0) == pytest.approx(np.ones(20), abs=1e-2)
    # another slide is moved by the training statistics, not centred on itself
    other = log_normalise(slide(8, depth=3.0))
    assert abs(scaler.transform(other).mean()) > 0.01
