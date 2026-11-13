"""Command line: download slides, cache image features, run an experiment."""

import argparse
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

from histoalign.encoders import load_encoder
from histoalign.experiment import block_subsets, prepare, run
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


def cmd_features(args) -> None:
    encoder = load_encoder(args.encoder, args.device)
    for raw in args.slides:
        print(extract(read_slide(raw), encoder, args.out, args.batch_size, args.hires))


def cmd_run(args) -> None:
    cfg = Config(**{k: getattr(args, k) for k in CONFIG_FLAGS if getattr(args, k) is not None})
    raw, feats = Path(args.raw), Path(args.features)
    if args.block:
        train_set, test_set = block_subsets(prepare(raw / args.block, feats), args.frac, args.gap)
        train_sets = [train_set]
    else:
        train_sets = [
            (p, np.arange(len(p.coords))) for p in (prepare(raw / s, feats) for s in args.train)
        ]
        test = prepare(raw / args.test, feats)
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

    p = sub.add_parser("features", help="embed the patch around every spot and cache it")
    p.add_argument("slides", nargs="+", help="folders made by `download`")
    p.add_argument("--encoder", default="lunit/dino-vit-s16")
    p.add_argument("--out", default="data/features")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device")
    p.add_argument("--hires", action="store_true", help="use the small png instead of the tiff")
    p.set_defaults(func=cmd_features)

    p = sub.add_parser("run", help="train on some spots, score on others")
    p.add_argument("--raw", default="data/raw")
    p.add_argument("--features", default="data/features")
    where = p.add_mutually_exclusive_group(required=True)
    where.add_argument("--train", nargs="+", help="samples to train on, needs --test")
    where.add_argument(
        "--block", help="one sample, trained on its left and tested on a right strip"
    )
    p.add_argument("--test")
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
    args.func(args)


if __name__ == "__main__":
    main()
