import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
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


def write_her2st(root, sections=("A1", "A2", "B1"), seed=0):
    """Sections in the layout of the unpacked HER2ST archives, 3 x 3 spots each.

    Each selection file also lists a spot that is not under the tissue and a selected spot that
    has no counts. Returns the counts of every section, spots x genes.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for name in sections:
        grid = [(10 + c, 20 + r) for r in range(GRID) for c in range(GRID)]
        ids = [f"{x}x{y}" for x, y in grid]
        counts = pd.DataFrame(rng.poisson(3.0, (len(ids), len(GENES))), ids, GENES)
        (root / "count-matrices").mkdir(parents=True, exist_ok=True)
        counts.to_csv(root / "count-matrices" / f"{name}.tsv.gz", sep="\t")
        out[name] = counts

        rows = [
            (x, y, x + 0.1, y - 0.1, FIRST + PITCH * (x - 10), FIRST + PITCH * (y - 20), 1)
            for x, y in grid
        ]
        rows += [(40, 40, 40.0, 40.0, 20.0, 20.0, 0), (41, 41, 41.0, 41.0, 880.0, 880.0, 1)]
        columns = ["x", "y", "new_x", "new_y", "pixel_x", "pixel_y", "selected"]
        (root / "spot-selections").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=columns).to_csv(
            root / "spot-selections" / f"{name}_selection.tsv.gz", sep="\t", index=False
        )

        ys, xs = np.mgrid[0:IMAGE_SIDE, 0:IMAGE_SIDE]
        image = np.stack([xs % 256, ys % 256, np.full_like(xs, 128)], axis=-1).astype(np.uint8)
        (root / "images" / "HE").mkdir(parents=True, exist_ok=True)
        Image.fromarray(image).save(root / "images" / "HE" / f"{name}.jpg", quality=95)
    return out
