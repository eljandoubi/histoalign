import numpy as np
import pytest

from histoalign.predict import (
    gene_correlation,
    knn_impute,
    ridge_impute,
    score,
    spot_correlation,
)


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_knn_with_one_neighbour_copies_its_expression():
    reference = unit(np.eye(4))
    expr = np.arange(8, dtype=np.float32).reshape(4, 2)
    query = unit(np.array([[0.1, 0.0, 1.0, 0.0], [1.0, 0.1, 0.0, 0.0]]))
    out = knn_impute(query, reference, expr, k=1)
    assert np.array_equal(out, expr[[2, 0]])


def test_knn_averages_the_k_best():
    reference = unit(np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]))
    expr = np.array([[0.0], [2.0], [100.0]], dtype=np.float32)
    out = knn_impute(unit(np.array([[1.0, 0.0]])), reference, expr, k=2)
    assert out[0, 0] == pytest.approx(1.0)


def test_chunk_size_does_not_matter():
    rng = np.random.default_rng(0)
    reference = unit(rng.normal(size=(40, 8)))
    query = unit(rng.normal(size=(25, 8)))
    expr = rng.normal(size=(40, 5)).astype(np.float32)
    assert np.allclose(
        knn_impute(query, reference, expr, k=5, chunk=4),
        knn_impute(query, reference, expr, k=5, chunk=1000),
    )


def test_k_larger_than_the_reference_is_capped():
    reference = unit(np.eye(3))
    expr = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
    out = knn_impute(unit(np.ones((1, 3))), reference, expr, k=50)
    assert out[0, 0] == pytest.approx(2.0)


def test_gene_correlation():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=(100, 4))
    truth[:, 3] = 5.0  # a constant gene has no correlation
    r = gene_correlation(truth, truth)
    assert r[:3] == pytest.approx(np.ones(3))
    assert np.isnan(r[3])
    assert gene_correlation(-truth, truth)[:3] == pytest.approx(-np.ones(3))


def test_spot_correlation():
    rng = np.random.default_rng(2)
    truth = rng.normal(size=(10, 30))
    assert spot_correlation(2 * truth + 1, truth) == pytest.approx(np.ones(10))


def test_score_uses_the_most_variable_genes_and_markers():
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(200, 6)) * np.array([5, 4, 3, 0.1, 0.1, 0.1])
    pred = truth.copy()
    pred[:, 3:] = rng.normal(size=(200, 3))  # wrong on the quiet genes only
    genes = [f"g{i}" for i in range(6)]
    out = score(pred, truth, genes, markers=("g0", "g4", "missing"), top=3)
    assert out["pcc_top"] == pytest.approx(1.0)
    assert out["pcc_all"] < 0.7
    assert out["pcc_g0"] == pytest.approx(1.0)
    assert abs(out["pcc_g4"]) < 0.3
    assert "pcc_missing" not in out


def test_ridge_recovers_a_linear_map():
    rng = np.random.default_rng(4)
    w = rng.normal(size=(12, 5))
    x = rng.normal(size=(300, 12))
    y = x @ w + 0.05 * rng.normal(size=(300, 5))
    pred = ridge_impute(x[:200], y[:200], x[200:])
    r = gene_correlation(pred, y[200:])
    assert r.min() > 0.95
