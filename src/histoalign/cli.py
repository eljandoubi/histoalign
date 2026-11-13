"""Command line: download slides, cache image features, run an experiment."""

import argparse
import json
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from histoalign import her2st
from histoalign.encoders import load_encoder
from histoalign.experiment import block_subsets, pool, prepare, prepare_slide, run
from histoalign.features import extract
from histoalign.train import Config
from histoalign.visium import read_slide

BASE = "https://cf.10xgenomics.com/samples/spatial-exp/1.1.0"
COUNTS = "{s}_filtered_feature_bc_matrix.h5"
SPATIAL = "{s}_spatial.tar.gz"
IMAGE = "{s}_image.tif"

# every field of train.Config that can be set on the command line, with its type
CONFIG_FLAGS = {
    "out_dim": int,
    "hidden": int,
    "dropout": float,
    "input_dropout": float,
    "lr": float,
    "weight_decay": float,
    "epochs": int,
    "batch_size": int,
    "loss": str,
    "temperature": float,
    "neighbour_radius": float,
    "patience": int,
    "seed": int,
}


def fetch(url: str, dest: Path) -> Path:
    """Download url to dest, unless a file of the right size is already there."""
    with urllib.request.urlopen(url) as resp:
        size = int(resp.headers.get("Content-Length", -1))
        if dest.exists() and dest.stat().st_size == size:
            return dest
        part = dest.with_suffix(dest.suffix + ".part")
        with open(part, "wb") as f:
            shutil.copyfileobj(resp, f, 1 << 20)
    part.replace(dest)
    return dest


def download(sample: str, out: Path, image: bool = True) -> Path:
    """Counts, spatial folder and (the 1.7 GB) full resolution image of a 10x Visium sample."""
    root = out / sample
    root.mkdir(parents=True, exist_ok=True)
    names = [COUNTS, SPATIAL] + ([IMAGE] if image else [])
    for name in names:
        name = name.format(s=sample)
        print("fetching", name, flush=True)
        fetch(f"{BASE}/{sample}/{name}", root / name)
    with tarfile.open(root / SPATIAL.format(s=sample)) as tar:
        tar.extractall(root, filter="data")
    return root


def code_version() -> str:
    """Short hash of the commit the code is at, marked +dirty if src has uncommitted changes."""
    here = Path(__file__).resolve().parent
    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        changed = subprocess.run(
            ["git", "status", "--porcelain", "--", str(here)],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return head + ("+dirty" if changed else "")


def summary(result: dict) -> str:
    lines = [
        f"train {result['train']['slides']} ({result['train']['spots']} spots), "
        f"test {result['test']['slide']} ({result['test']['spots']} spots), "
        f"{len(result['genes'])} genes",
        "",
        "retrieval, test spots against test spots",
    ]
    for direction, row in result["retrieval"].items():
        cells = "  ".join(
            f"{k} {v:.3f}" if "rank" not in k else f"{k} {v:.0f}" for k, v in row.items()
        )
        lines.append(f"  {direction:<14} {cells}")
    lines += ["", "expression imputation, Pearson r"]
    for method, row in result["imputation"].items():
        cells = "  ".join(f"{k[4:]} {v:.3f}" for k, v in row.items())
        lines.append(f"  {method:<28} {cells}")
    return "\n".join(lines)


def cmd_download(args) -> None:
    for sample in args.samples:
        download(sample, Path(args.out), image=not args.no_image)


def cmd_unpack(args) -> None:
    print(her2st.unpack(args.zips, args.out))


def cmd_features(args) -> None:
    names = args.slides or (her2st.sections(args.her2st) if args.her2st else [])
    if not names:
        raise SystemExit("give the slides to embed")
    encoder = load_encoder(args.encoder, args.device)
    for name in names:
        slide = her2st.read_section(args.her2st, name) if args.her2st else read_slide(name)
        print(extract(slide, encoder, args.out, args.batch_size, args.hires))


def split_by_patient(root: str, patient: str) -> tuple[list[str], list[str]]:
    """Names of the HER2ST sections to train on and to test on when `patient` is left out."""
    names = her2st.sections(root)
    held = [s for s in names if her2st.patient(s) == patient]
    if not held:
        raise SystemExit(f"no section of patient {patient!r} in {root}")
    return [s for s in names if her2st.patient(s) != patient], held


def cmd_run(args) -> None:
    cfg = Config(**{k: getattr(args, k) for k in CONFIG_FLAGS if getattr(args, k) is not None})
    feats = Path(args.features)

    def load(name):
        if args.her2st:
            return prepare_slide(her2st.read_section(args.her2st, name), feats)
        return prepare(Path(args.raw) / name, feats)

    if args.block:
        train_set, test_set = block_subsets(load(args.block), args.frac, args.gap)
        train_sets = [train_set]
    else:
        train_names, test_names, label = args.train, args.test, None
        if args.leave_out:
            train_names, test_names = split_by_patient(args.her2st, args.leave_out)
            label = f"patient {args.leave_out}"
        train_sets = [(p, np.arange(len(p.coords))) for p in map(load, train_names)]
        parts = [load(s) for s in test_names]
        test = parts[0] if len(parts) == 1 else pool(parts, label)
        test_set = (test, np.arange(len(test.coords)))
    result = run(
        train_sets,
        test_set,
        cfg,
        n_genes=args.genes,
        k=args.k,
        control=not args.no_control,
        val_frac=args.val_frac,
    )
    print(summary(result))
    result["code"] = code_version()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="histoalign")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="fetch 10x Visium samples")
    p.add_argument("samples", nargs="+")
    p.add_argument("--out", default="data/raw")
    p.add_argument("--no-image", action="store_true", help="skip the 1.7 GB tiff")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("unpack", help="decrypt the HER2ST archives from Zenodo")
    p.add_argument("zips", help="folder with the four zip files")
    p.add_argument("--out", default="data/her2st/raw")
    p.set_defaults(func=cmd_unpack)

    p = sub.add_parser("features", help="embed the patch around every spot and cache it")
    p.add_argument("slides", nargs="*", help="folders made by `download`, or HER2ST sections")
    p.add_argument("--her2st", help="unpacked HER2ST folder, slides are its sections (all if none)")
    p.add_argument("--encoder", default="lunit/dino-vit-s16")
    p.add_argument("--out", default="data/features")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device")
    p.add_argument("--hires", action="store_true", help="use the small png instead of the tiff")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("run", help="train on some spots, score on others")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--her2st", help="unpacked HER2ST folder, samples are its sections")
    p.add_argument("--features", default="data/features")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--train", nargs="+", help="samples to train on, needs --test")
    where.add_argument(
        "--block", help="one sample, trained on its left and tested on a right strip"
    )
    where.add_argument("--leave-out", help="HER2ST patient to test on, the others train")
    p.add_argument("--test", nargs="+", help="samples to test on, scored together")
    p.add_argument("--frac", type=float, default=0.2, help="share of the slide held out (--block)")
    p.add_argument(
        "--gap", type=float, default=2.0, help="spot pitches left out around it (--block)"
    )
    p.add_argument("--genes", type=int, default=1000)
    p.add_argument("--k", type=int, default=50)
    p.add_argument("--val-frac", type=float, default=0.15, help="strip of training used to stop")
    p.add_argument("--no-control", action="store_true", help="skip the shuffled pairs control")
    p.add_argument("--out", help="write the full result as json")
    for name, kind in CONFIG_FLAGS.items():
        p.add_argument(f"--{name.replace('_', '-')}", dest=name, type=kind)
    p.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "run" and args.train and not args.test:
        raise SystemExit("--train needs --test")
    if args.command == "run" and args.leave_out and not args.her2st:
        raise SystemExit("--leave-out needs --her2st")
    args.func(args)


if __name__ == "__main__":
    main()
