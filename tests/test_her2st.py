import numpy as np
import pandas as pd
import pytest
import pyzipper

from histoalign import her2st
from histoalign.encoders import save_features
from histoalign.experiment import prepare_slide
from tests.slides import FIRST, GENES, GRID, PITCH, write_her2st


@pytest.fixture
def raw(tmp_path):
    write_her2st(tmp_path / "raw")
    return tmp_path / "raw"


def test_sections_and_their_patients(raw):
    names = her2st.sections(raw)
    assert names == ["A1", "A2", "B1"]
    assert [her2st.patient(s) for s in names] == ["A", "A", "B"]


def test_only_selected_spots_that_have_counts_are_kept(raw):
    slide = her2st.read_section(raw, "A1")
    assert slide.adata.n_obs == GRID * GRID
    assert slide.adata.n_vars == len(GENES)
    assert not {"40x40", "41x41"} & set(slide.adata.obs_names)


def test_positions_follow_the_spot_not_the_row(tmp_path):
    write_her2st(tmp_path)
    path = tmp_path / "spot-selections" / "A1_selection.tsv.gz"
    pd.read_csv(path, sep="\t").iloc[::-1].to_csv(path, sep="\t", index=False)
    slide = her2st.read_section(tmp_path, "A1")
    i = list(slide.adata.obs_names).index("12x22")
    assert tuple(slide.adata.obsm["spatial"][i]) == (FIRST + 2 * PITCH, FIRST + 2 * PITCH)


def test_counts_survive_the_round_trip(tmp_path):
    counts = write_her2st(tmp_path)["A2"]
    slide = her2st.read_section(tmp_path, "A2")
    assert list(slide.adata.obs_names) == list(counts.index)
    assert np.array_equal(slide.adata.X.toarray(), counts.to_numpy())


def test_spatial_is_pixel_x_then_y(raw):
    first, second = her2st.read_section(raw, "A1").adata.obsm["spatial"][:2]
    assert tuple(first) == (FIRST, FIRST)
    assert tuple(second) == (FIRST + PITCH, FIRST)


def test_pixel_size_comes_from_the_spacing_of_the_spots(raw):
    slide = her2st.read_section(raw, "A1")
    # the two odd spots are far from the grid and must not move the median
    assert slide.pitch_px == pytest.approx(PITCH)
    assert slide.mpp == pytest.approx(200.0 / PITCH)
    assert slide.spot_diameter_px == pytest.approx(PITCH / 2)
    assert slide.image_path == raw / "images" / "HE" / "A1.jpg" and slide.hires_path is None


def test_a_section_is_prepared_like_a_visium_slide(raw, tmp_path):
    slide = her2st.read_section(raw, "A1")
    feats = np.arange(slide.adata.n_obs * 4, dtype=np.float32).reshape(-1, 4)
    save_features(tmp_path / "A1.h5", list(slide.adata.obs_names), feats)
    p = prepare_slide(slide, tmp_path)
    assert p.name == "A1" and p.image.shape == (GRID * GRID, 4)
    assert p.pitch_px == pytest.approx(slide.pitch_px)


def make_archives(folder):
    layout = {
        "count-matrices.zip": {"count-matrices/A1.tsv.gz": b"counts"},
        "images.zip": {"images/HE/A1.jpg": b"jpeg"},
        "meta.zip": {"A1_labeled_coordinates.tsv": b"labels"},
        "spot-selections.zip": {"A1_selection.tsv.gz": b"selection"},
    }
    folder.mkdir()
    for name, files in layout.items():
        with pyzipper.AESZipFile(
            folder / name, "w", compression=pyzipper.ZIP_DEFLATED, encryption=pyzipper.WZ_AES
        ) as archive:
            archive.setpassword(her2st.PASSWORDS[name])
            for member, data in files.items():
                archive.writestr(member, data)


def test_unpack_decrypts_into_one_folder_per_kind(tmp_path):
    make_archives(tmp_path / "zips")
    out = her2st.unpack(tmp_path / "zips", tmp_path / "raw")
    assert (out / "count-matrices" / "A1.tsv.gz").read_bytes() == b"counts"
    assert (out / "images" / "HE" / "A1.jpg").read_bytes() == b"jpeg"
    assert (out / "meta" / "A1_labeled_coordinates.tsv").read_bytes() == b"labels"
    assert (out / "spot-selections" / "A1_selection.tsv.gz").read_bytes() == b"selection"


def test_unpack_leaves_files_that_are_there(tmp_path):
    make_archives(tmp_path / "zips")
    out = her2st.unpack(tmp_path / "zips", tmp_path / "raw")
    target = out / "images" / "HE" / "A1.jpg"
    stamp = target.stat().st_mtime_ns
    her2st.unpack(tmp_path / "zips", out)
    assert target.stat().st_mtime_ns == stamp
