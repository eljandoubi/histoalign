import json
import tarfile

import numpy as np
import pytest

from histoalign import cli
from histoalign.encoders import save_features
from histoalign.visium import read_slide
from tests.slides import write_slide


def make_server(tmp_path, sample="V1_Fake_Section_1"):
    """A folder that looks like the 10x download area, reachable through file:// urls."""
    src = tmp_path / "server_src" / sample
    write_slide(src)
    folder = tmp_path / "server" / sample
    folder.mkdir(parents=True)
    for f in src.glob("*.h5"):
        (folder / f.name).write_bytes(f.read_bytes())
    (folder / f"{sample}_image.tif").write_bytes(next(src.glob("*_image.tif")).read_bytes())
    with tarfile.open(folder / f"{sample}_spatial.tar.gz", "w:gz") as tar:
        tar.add(src / "spatial", arcname="spatial")
    return (tmp_path / "server").as_uri()


def test_download_unpacks_and_skips_what_is_there(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "BASE", make_server(tmp_path))
    out = tmp_path / "raw"
    root = cli.download("V1_Fake_Section_1", out)
    assert (root / "spatial" / "scalefactors_json.json").exists()
    assert read_slide(root).adata.n_obs == 9
    tif = root / "V1_Fake_Section_1_image.tif"
    stamp = tif.stat().st_mtime_ns
    cli.download("V1_Fake_Section_1", out)
    assert tif.stat().st_mtime_ns == stamp


def test_download_without_the_image(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "BASE", make_server(tmp_path))
    root = cli.download("V1_Fake_Section_1", tmp_path / "raw", image=False)
    assert not list(root.glob("*_image.tif"))


def test_flags_become_a_config():
    args = cli.build_parser().parse_args(
        ["run", "--train", "a", "b", "--test", "c", "--epochs", "7", "--loss", "sigmoid"]
    )
    assert (args.epochs, args.loss, args.lr) == (7, "sigmoid", None)
    assert args.train == ["a", "b"]


def test_train_without_test_is_refused():
    with pytest.raises(SystemExit):
        cli.main(["run", "--train", "a"])


def test_run_end_to_end_on_tiny_slides(tmp_path, capsys):
    raw, feats = tmp_path / "raw", tmp_path / "feat"
    feats.mkdir()
    rng = np.random.default_rng(0)
    for name, seed in (("S1", 1), ("S2", 2), ("S3", 3)):
        write_slide(raw / name, seed=seed)
        slide = read_slide(raw / name)
        save_features(
            feats / f"{name}.h5",
            list(slide.adata.obs_names),
            rng.normal(size=(9, 6)).astype(np.float32),
        )
    out = tmp_path / "runs" / "tiny.json"
    cli.main(
        ["run", "--raw", str(raw), "--features", str(feats), "--train", "S1", "S2", "--test", "S3"]
        + ["--genes", "8", "--k", "3", "--epochs", "2", "--hidden", "8", "--out-dim", "4"]
        + ["--out", str(out)]
    )
    shown = capsys.readouterr().out
    assert "ridge" in shown and "aligned_knn_shuffled_pairs" in shown
    result = json.loads(out.read_text())
    assert result["test"]["slide"] == "S3" and result["train"]["slides"] == ["S1", "S2"]
    assert result["config"]["epochs"] == 2
