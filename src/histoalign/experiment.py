"""One experiment: fit on some spots, score on others."""

import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import anndata as ad
import numpy as np

from histoalign.encoders import load_features
from histoalign.genes import Scaler, log_normalise, select_genes
from histoalign.predict import knn_impute, ridge_impute, score
from histoalign.retrieval import median_rank, near_recall_at_k, recall_at_k
from histoalign.train import Config, Pairs, encode, spatial_block_split, train
from histoalign.visium import Slide, read_slide

# genes with a known pattern in breast tumours, kept whatever the variance filter says
MARKERS = ("ERBB2", "ESR1", "PGR", "MKI67", "KRT8", "KRT18", "EPCAM", "COL1A1", "CD74", "MS4A1")
KS = (1, 5, 10)
FAR = 1e7  # pixels between two pooled slides, much more than any slide is wide


@dataclass
class Prepared:
    """A slide with log-normalised counts and cached image features, spots in the same order."""

    name: str
    adata: ad.AnnData
    image: np.ndarray
    coords: np.ndarray
    pitch_px: float


Subset = tuple[Prepared, np.ndarray]  # a slide and the indices of the spots that are used


def prepare(raw_dir: str | Path, feature_dir: str | Path) -> Prepared:
    return prepare_slide(read_slide(raw_dir), feature_dir)


def prepare_slide(slide: Slide, feature_dir: str | Path) -> Prepared:
    barcodes, image, _ = load_features(Path(feature_dir) / f"{slide.sample_id}.h5")
    if barcodes != list(slide.adata.obs_names):
        raise ValueError(f"{slide.sample_id}: cached features are not in the order of the spots")
    return Prepared(
        slide.sample_id,
        log_normalise(slide.adata),
        image,
        slide.adata.obsm["spatial"],
        slide.pitch_px,
    )


def pool(parts: list[Prepared], name: str | None = None) -> Prepared:
    """Several slides as one, to score all the sections of a patient together.

    Each slide is moved far away from the others in x, so that no spot counts as a neighbour of
    a spot of another slide. A gene that a slide lacks is zero in its spots.
    """
    adata = ad.concat({p.name: p.adata for p in parts}, join="outer", index_unique="-")
    coords = np.vstack([p.coords + [i * FAR, 0.0] for i, p in enumerate(parts)])
    return Prepared(
        name or "+".join(p.name for p in parts),
        adata,
        np.vstack([p.image for p in parts]),
        coords,
        float(np.mean([p.pitch_px for p in parts])),
    )


def block_subsets(p: Prepared, frac: float = 0.2, gap: float = 2.0) -> tuple[Subset, Subset]:
    """Train on the left of one slide and test on a strip at its right edge.

    `gap` is in spot pitches: the spots next to the strip are left out of training.
    """
    slide = np.zeros(len(p.coords), dtype=int)
    train_mask, held = spatial_block_split(p.coords, slide, frac, gap_px=gap * p.pitch_px)
    return (p, np.flatnonzero(train_mask)), (p, np.flatnonzero(held))


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def _recalls(sim: np.ndarray, coords: np.ndarray, radius: float) -> dict[str, float]:
    out = {f"recall@{k}": v for k, v in recall_at_k(sim, KS).items()}
    out["near_recall@1"] = near_recall_at_k(sim, coords, radius, (1,))[1]
    out["near_recall@10"] = near_recall_at_k(sim, coords, radius, (10,))[10]
    out["median_rank"] = median_rank(sim)
    return out


def run(
    train_sets: list[Subset],
    test_set: Subset,
    cfg: Config,
    n_genes: int = 1000,
    markers: tuple[str, ...] = MARKERS,
    k: int = 50,
    control: bool = True,
    val_frac: float = 0.15,
    val_gap: float = 2.0,
    log=print,
) -> dict:
    """Fit on `train_sets`, score on `test_set`.

    The genes are the most variable ones of the training spots plus the `markers` that every slide
    has, the test slide included: that reads the names of its genes, never their expression, so the
    markers can depend on the held out patient. The scaling is fitted on the training spots. The
    expression and the images of the test spots are not used to choose genes, scaling or epochs.
    """
    t0 = time.time()
    test_p, test_idx = test_set
    train_adatas = [p.adata[idx] for p, idx in train_sets]
    test_adata = test_p.adata[test_idx]

    genes = select_genes(train_adatas, n_top=n_genes)
    everywhere = train_adatas + [test_adata]
    genes += [m for m in markers if m not in genes and all(m in a.var_names for a in everywhere)]
    scaler = Scaler(genes).fit(train_adatas)

    pairs = Pairs(
        image=np.vstack([p.image[idx] for p, idx in train_sets]),
        expr=np.vstack([scaler.transform(a) for a in train_adatas]),
        coords=np.vstack([p.coords[idx] for p, idx in train_sets]),
        slide=np.concatenate([np.full(len(idx), i) for i, (_, idx) in enumerate(train_sets)]),
    )
    test_image = test_p.image[test_idx]
    truth = scaler.transform(test_adata)
    test_coords = test_p.coords[test_idx]
    pitch = float(np.mean([p.pitch_px for p, _ in train_sets]))
    radius_px = cfg.neighbour_radius * pitch

    fit, val = pairs, None
    if val_frac > 0:
        keep, held = spatial_block_split(pairs.coords, pairs.slide, val_frac, val_gap * pitch)
        fit, val = pairs.take(keep), pairs.take(held)
    model, info = train(fit, cfg, radius_px=radius_px, log=log, val=val)
    img_test, expr_test = encode(model, test_image, truth)
    _, expr_train = encode(model, pairs.image, pairs.expr)

    near = 1.5 * test_p.pitch_px
    sim = img_test @ expr_test.T
    chance = np.random.default_rng(0).normal(size=sim.shape)
    imputation = {
        "aligned_knn": score(
            knn_impute(img_test, expr_train, pairs.expr, k), truth, genes, markers
        ),
        "raw_knn": score(
            knn_impute(_unit(test_image), _unit(pairs.image), pairs.expr, k), truth, genes, markers
        ),
        "ridge": score(ridge_impute(pairs.image, pairs.expr, test_image), truth, genes, markers),
    }
    if control:
        order = np.random.default_rng(cfg.seed).permutation(len(pairs))
        shuffled = Pairs(pairs.image, pairs.expr[order], pairs.coords, pairs.slide)
        fixed = replace(cfg, epochs=info["best_epoch"])
        fit_c = shuffled.take(keep) if val is not None else shuffled
        model_c, _ = train(fit_c, fixed, radius_px=radius_px, log=None)
        img_c, _ = encode(model_c, test_image, truth)
        _, expr_c = encode(model_c, shuffled.image, shuffled.expr)
        imputation["aligned_knn_shuffled_pairs"] = score(
            knn_impute(img_c, expr_c, shuffled.expr, k), truth, genes, markers
        )

    return {
        "config": asdict(cfg),
        "train": {
            "slides": [p.name for p, _ in train_sets],
            "spots": len(pairs),
            "fit_spots": len(fit),
            "val_spots": 0 if val is None else len(val),
        },
        "test": {"slide": test_p.name, "spots": len(test_idx)},
        "genes": genes,
        "epochs": info["best_epoch"],
        "final_loss": info["loss"][-1],
        "retrieval": {
            "image_to_expr": _recalls(sim, test_coords, near),
            "expr_to_image": _recalls(sim.T, test_coords, near),
            "chance": _recalls(chance, test_coords, near),
        },
        "imputation": imputation,
        "seconds": round(time.time() - t0, 1),
    }
