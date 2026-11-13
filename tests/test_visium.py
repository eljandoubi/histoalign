from dataclasses import replace

import numpy as np
import pytest

from histoalign.visium import SPOT_DIAMETER_UM, read_slide
from tests.slides import FIRST, GRID, PITCH, SAMPLE, SPOT_DIAMETER


def test_only_spots_in_the_matrix_are_kept(fake_slide):
    slide = read_slide(fake_slide.root)
    assert slide.adata.n_obs == GRID * GRID
    assert "OUTSIDE-1" not in slide.adata.obs_names


def test_spatial_is_x_then_y(fake_slide):
    slide = read_slide(fake_slide.root)
    # the second spot sits one step to the right of the first, so only x moves
    first, second = slide.adata.obsm["spatial"][:2]
    assert tuple(first) == (FIRST, FIRST)
    assert tuple(second) == (FIRST + PITCH, FIRST)


def test_scale_and_paths(fake_slide):
    slide = read_slide(fake_slide.root)
    assert slide.sample_id == SAMPLE
    assert slide.mpp == pytest.approx(SPOT_DIAMETER_UM / SPOT_DIAMETER)
    assert slide.pitch_px == pytest.approx(100.0 / slide.mpp)
    assert slide.image_path.name.endswith("_image.tif")
    assert slide.hires_path is None


def test_counts_survive_the_round_trip(fake_slide):
    slide = read_slide(fake_slide.root)
    assert np.array_equal(slide.adata.X.toarray(), fake_slide.counts.T)


def test_positions_file_with_header(fake_slide):
    spatial = fake_slide.root / "spatial"
    old = (spatial / "tissue_positions_list.csv").read_text()
    header = "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
    (spatial / "tissue_positions.csv").write_text(header + old)
    (spatial / "tissue_positions_list.csv").unlink()
    slide = read_slide(fake_slide.root)
    assert slide.adata.obsm["spatial"].shape == (GRID * GRID, 2)
    assert slide.adata.obs["array_col"].tolist()[:3] == [0, 1, 2]


def test_a_slide_can_carry_its_own_spot_size(fake_slide):
    slide = replace(read_slide(fake_slide.root), spot_diameter_um=100.0, spot_pitch_um=200.0)
    assert slide.mpp == pytest.approx(100.0 / SPOT_DIAMETER)
    assert slide.pitch_px == pytest.approx(200.0 / slide.mpp)
