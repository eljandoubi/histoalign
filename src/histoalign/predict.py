"""Predict the expression of spots from their image, and score the prediction."""

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def knn_impute(
    query: np.ndarray,
    reference: np.ndarray,
    reference_expr: np.ndarray,
    k: int = 50,
    chunk: int = 256,
) -> np.ndarray:
    """Average the expression of the k reference spots that look most like each query.

    Similarity is the dot product, so `query` and `reference` are expected to be L2 normalised.
    The same call serves the aligned space and the raw image features, which is what makes the
    second one a fair baseline.
    """
    k = min(k, len(reference))
    out = np.empty((len(query), reference_expr.shape[1]), dtype=np.float32)
    for start in range(0, len(query), chunk):
        sim = query[start : start + chunk] @ reference.T
        top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
        out[start : start + chunk] = reference_expr[top].mean(axis=1)
    return out


def ridge_impute(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    alphas: tuple[float, ...] = (1.0, 10.0, 100.0, 1e3, 1e4, 1e5),
) -> np.ndarray:
    """Linear baseline on the frozen features, with the penalty picked by leave-one-out."""
    model = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(train_x, train_y)
    return model.predict(test_x).astype(np.float32)


def _pearson(a: np.ndarray, b: np.ndarray, axis: int) -> np.ndarray:
    a = a - a.mean(axis=axis, keepdims=True)
    b = b - b.mean(axis=axis, keepdims=True)
    num = (a * b).sum(axis=axis)
    den = np.sqrt((a * a).sum(axis=axis) * (b * b).sum(axis=axis))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def gene_correlation(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pearson correlation of each gene across spots. A constant gene gives nan."""
    return _pearson(pred, truth, axis=0)


def spot_correlation(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Pearson correlation of each spot across genes."""
    return _pearson(pred, truth, axis=1)


def score(
    pred: np.ndarray,
    truth: np.ndarray,
    genes: list[str],
    markers: tuple[str, ...] = (),
    top: int = 50,
) -> dict[str, float]:
    """Mean correlation over all genes, over the `top` genes with the largest variance in the
    truth, over spots, and the correlation of each marker gene that is in `genes`."""
    per_gene = gene_correlation(pred, truth)
    top_genes = np.argsort(-truth.var(axis=0))[:top]
    out = {
        "pcc_all": float(np.nanmean(per_gene)),
        "pcc_top": float(np.nanmean(per_gene[top_genes])),
        "pcc_spot": float(np.nanmean(spot_correlation(pred, truth))),
    }
    for name in markers:
        if name in genes:
            out[f"pcc_{name}"] = float(per_gene[genes.index(name)])
    return out
