"""Read the HER2ST sections: 36 sections of 8 patients on Spatial Transcriptomics arrays.

Andersson et al., Nature Communications 2021, data at https://zenodo.org/record/3957257. The
spots are 100 microns wide and 200 microns apart. The image is a JPEG and nothing in the data
gives its pixel size, so it is worked out from the distance between neighbouring spots.
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pyzipper
from scipy import sparse
from scipy.spatial import cKDTree

from histoalign.visium import Slide

SPOT_DIAMETER_UM = 100.0
SPOT_PITCH_UM = 200.0

# the authors publish these next to the data, see https://github.com/almaan/her2st
PASSWORDS = {
    "count-matrices.zip": b"zNLXkYk3Q9znUseS",
    "images.zip": b"zNLXkYk3Q9znUseS",
    "meta.zip": b"yUx44SzG6NdB32gY",
    "spot-selections.zip": b"yUx44SzG6NdB32gY",
}


def unpack(zips: str | Path, out: str | Path) -> Path:
    """Decrypt the four archives of the Zenodo record into `out`.

    Two of them keep their files at the top level, they get a folder named after the archive.
    A file that is already there with the right size is left alone.
    """
    zips, out = Path(zips), Path(out)
    for name, password in PASSWORDS.items():
        with pyzipper.AESZipFile(zips / name) as archive:
            archive.setpassword(password)
            folder = out if all("/" in n for n in archive.namelist()) else out / Path(name).stem
            for info in archive.infolist():
                target = folder / info.filename
                done = target.exists() and target.stat().st_size == info.file_size
                if done and not info.is_dir():
                    continue
                archive.extract(info, folder)
    return out


def sections(root: str | Path) -> list[str]:
    """Names of the sections that have a count matrix, like A1 or H3."""
    found = (Path(root) / "count-matrices").glob("*.tsv.gz")
    return sorted(p.name.removesuffix(".tsv.gz") for p in found)


def patient(section: str) -> str:
    """The letter of the patient a section comes from."""
    return section[0]


def read_section(root: str | Path, section: str) -> Slide:
    """Counts of the spots under the tissue, their pixel positions and the path of the image."""
    root = Path(root)
    counts = pd.read_csv(root / "count-matrices" / f"{section}.tsv.gz", sep="\t", index_col=0)
    table = pd.read_csv(root / "spot-selections" / f"{section}_selection.tsv.gz", sep="\t")
    table.index = table["x"].astype(str) + "x" + table["y"].astype(str)
    table = table[table["selected"] == 1]

    keep = counts.index[counts.index.isin(table.index)]
    counts, spots = counts.loc[keep], table.loc[keep]
    adata = ad.AnnData(
        X=sparse.csr_matrix(counts.to_numpy(dtype=np.float32)),
        obs=pd.DataFrame(index=keep),
        var=pd.DataFrame(index=counts.columns),
    )
    adata.var_names_make_unique()
    adata.obs["array_row"] = spots["y"].to_numpy()
    adata.obs["array_col"] = spots["x"].to_numpy()
    adata.obs["slide"] = section
    # x then y, in pixels of the full resolution image
    adata.obsm["spatial"] = spots[["pixel_x", "pixel_y"]].to_numpy(dtype=float)

    centres = adata.obsm["spatial"]
    pitch_px = float(np.median(cKDTree(centres).query(centres, k=2)[0][:, 1]))
    return Slide(
        sample_id=section,
        adata=adata,
        spot_diameter_px=pitch_px * SPOT_DIAMETER_UM / SPOT_PITCH_UM,
        image_path=root / "images" / "HE" / f"{section}.jpg",
        hires_path=None,
        hires_scale=1.0,
        spot_diameter_um=SPOT_DIAMETER_UM,
        spot_pitch_um=SPOT_PITCH_UM,
    )
