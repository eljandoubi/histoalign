import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import tifffile
from scipy import sparse

SAMPLE = "V1_Fake_Section_1"
GRID = 3  # spots per side
PITCH = 300  # pixels between spot centres
FIRST = 150  # pixel position of the first spot centre
SPOT_DIAMETER = 100.0
IMAGE_SIDE = 900
GENES = [f"GENE{i}" for i in range(12)]


def write_10x_h5(path, counts, barcodes, genes):
    """counts is genes x spots, stored the way Space Ranger does (CSC, one column per barcode)."""
    mat = sparse.csc_matrix(counts)
    with h5py.File(path, "w") as f:
        m = f.create_group("matrix")
        m["barcodes"] = np.array(barcodes, dtype="S")
        m["data"] = mat.data.astype("int32")
        m["indices"] = mat.indices.astype("int64")
        m["indptr"] = mat.indptr.astype("int64")
        m["shape"] = np.array(mat.shape, dtype="int32")
        feats = m.create_group("features")
        feats["id"] = np.array([f"ENSG{i:011d}" for i in range(len(genes))], dtype="S")
        feats["name"] = np.array(genes, dtype="S")
        feats["feature_type"] = np.array(["Gene Expression"] * len(genes), dtype="S")
        feats["genome"] = np.array(["GRCh38"] * len(genes), dtype="S")


@dataclass
class FakeSlide:
    root: Path
    counts: np.ndarray  # genes x spots


def write_slide(root, seed=0):
    """A 3 x 3 grid of spots plus one barcode outside the tissue that has no counts."""
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    barcodes = [f"AAAC{i:02d}-1" for i in range(GRID * GRID)]
    counts = rng.poisson(3.0, size=(len(GENES), len(barcodes)))
    write_10x_h5(root / f"{root.name}_filtered_feature_bc_matrix.h5", counts, barcodes, GENES)

    spatial = root / "spatial"
    spatial.mkdir(exist_ok=True)
    (spatial / "scalefactors_json.json").write_text(
        json.dumps({"spot_diameter_fullres": SPOT_DIAMETER, "tissue_hires_scalef": 0.1})
    )
    rows = []
    for i, bc in enumerate(barcodes):
        r, c = divmod(i, GRID)
        rows.append(f"{bc},1,{r},{c},{FIRST + PITCH * r},{FIRST + PITCH * c}")
    rows.append("OUTSIDE-1,0,5,5,10,10")
    (spatial / "tissue_positions_list.csv").write_text("\n".join(rows) + "\n")

    # red channel carries x, green carries y, so a crop shows where it came from
    ys, xs = np.mgrid[0:IMAGE_SIDE, 0:IMAGE_SIDE]
    image = np.stack([xs % 256, ys % 256, np.full_like(xs, 128)], axis=-1).astype(np.uint8)
    tifffile.imwrite(root / f"{root.name}_image.tif", image)
    return FakeSlide(root, counts)
