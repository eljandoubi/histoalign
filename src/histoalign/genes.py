"""Gene selection and scaling.

The most variable genes and the mean and spread of each gene are computed from the training
spots. `experiment.run` then adds the breast cancer markers that every slide has, the test slide
included: that reads the names of the genes of the test slide, never their expression, so the
markers can depend on the held out patient.
"""

import anndata as ad
import numpy as np
import scanpy as sc
from scipy import sparse


def log_normalise(adata: ad.AnnData, target_sum: float = 1e4) -> ad.AnnData:
    """Library size normalisation and log1p, on a copy. The raw counts stay in layers["counts"]."""
    out = adata.copy()
    out.layers["counts"] = out.X.copy()
    sc.pp.normalize_total(out, target_sum=target_sum)
    sc.pp.log1p(out)
    return out


def select_genes(train: list[ad.AnnData], n_top: int = 1000, min_frac: float = 0.05) -> list[str]:
    """Highly variable genes of the training slides.

    The slides are treated as batches, so a gene has to be variable across slides and not in one
    of them only. Genes seen in fewer than `min_frac` of the spots are dropped first.
    """
    slides = {str(i): a for i, a in enumerate(train)}
    merged = ad.concat(slides, join="inner", label="batch", index_unique="-")
    counts = merged.X
    seen = np.asarray((counts > 0).mean(axis=0)).ravel()
    merged = merged[:, seen >= min_frac].copy()
    sc.pp.highly_variable_genes(
        merged,
        n_top_genes=n_top,
        flavor="seurat",
        batch_key="batch" if len(train) > 1 else None,
    )
    return list(merged.var_names[merged.var["highly_variable"]])


def take_genes(adata: ad.AnnData, genes: list[str]) -> np.ndarray:
    """Dense spots x genes matrix in the order of `genes`. A gene the slide lacks is all zeros."""
    index = adata.var_names.get_indexer(genes)
    present = index >= 0
    block = adata.X[:, index[present]]
    block = block.toarray() if sparse.issparse(block) else np.asarray(block)
    out = np.zeros((adata.n_obs, len(genes)), dtype=np.float32)
    out[:, present] = block
    return out


class Scaler:
    """Per gene mean and standard deviation, fitted on training spots."""

    def __init__(self, genes: list[str]):
        self.genes = genes
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, train: list[ad.AnnData]) -> "Scaler":
        x = np.vstack([take_genes(a, self.genes) for a in train])
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        return self

    def transform(self, adata: ad.AnnData) -> np.ndarray:
        return ((take_genes(adata, self.genes) - self.mean) / self.std).astype(np.float32)
