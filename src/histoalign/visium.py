"""Read one 10x Visium slide: counts, spot positions and the paths of the H&E images."""

import json
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import pandas as pd
import scanpy as sc

SPOT_DIAMETER_UM = 55.0
SPOT_PITCH_UM = 100.0

POSITION_COLUMNS = ["in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]


@dataclass
class Slide:
    sample_id: str
    adata: ad.AnnData
    spot_diameter_px: float
    image_path: Path | None
    hires_path: Path | None
    hires_scale: float

    @property
    def mpp(self) -> float:
        """Microns per pixel of the full resolution image."""
        return SPOT_DIAMETER_UM / self.spot_diameter_px

    @property
    def pitch_px(self) -> float:
        """Distance between two neighbouring spot centres, in full resolution pixels."""
        return SPOT_PITCH_UM / self.mpp


def read_positions(spatial_dir: Path) -> pd.DataFrame:
    """Spot table indexed by barcode.

    Space Ranger 1.x writes tissue_positions_list.csv without a header, 2.x writes
    tissue_positions.csv with one. The column order is the same in both.
    """
    new = spatial_dir / "tissue_positions.csv"
    if new.exists():
        table = pd.read_csv(new, header=0, index_col=0).iloc[:, :5]
    else:
        table = pd.read_csv(spatial_dir / "tissue_positions_list.csv", header=None, index_col=0)
    table.columns = POSITION_COLUMNS
    return table


def read_slide(root: str | Path, sample_id: str | None = None) -> Slide:
    root = Path(root)
    sample_id = sample_id or root.name
    adata = sc.read_10x_h5(next(root.glob("*filtered_feature_bc_matrix.h5")))
    adata.var_names_make_unique()

    spatial = root / "spatial"
    spots = read_positions(spatial).loc[adata.obs_names]
    adata.obs["array_row"] = spots["array_row"].to_numpy()
    adata.obs["array_col"] = spots["array_col"].to_numpy()
    adata.obs["slide"] = sample_id
    # x then y, in pixels of the full resolution image
    adata.obsm["spatial"] = spots[["pxl_col", "pxl_row"]].to_numpy(dtype=float)

    scales = json.loads((spatial / "scalefactors_json.json").read_text())
    tifs = sorted(root.glob("*_image.tif"))
    hires = spatial / "tissue_hires_image.png"
    return Slide(
        sample_id=sample_id,
        adata=adata,
        spot_diameter_px=scales["spot_diameter_fullres"],
        image_path=tifs[0] if tifs else None,
        hires_path=hires if hires.exists() else None,
        hires_scale=scales["tissue_hires_scalef"],
    )
