"""Print the results folder as markdown: mean and spread over seeds, one table per split.

uv run python scripts/results_table.py results [variant ...]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

METHODS = ["aligned_knn", "raw_knn", "ridge", "aligned_knn_shuffled_pairs"]
SCORES = [("pcc_all", "all genes"), ("pcc_top", "top 50 genes"), ("pcc_spot", "per spot")]


def load(folder: Path) -> dict[str, dict[str, list[dict]]]:
    """results/<split>__<variant>__seed<k>.json grouped as runs[split][variant] = [result, ...]"""
    runs: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(folder.glob("*.json")):
        split, variant, _ = path.stem.split("__")
        runs[split][variant].append(json.loads(path.read_text()))
    return runs


def fmt(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{np.mean(values):.3f} ± {np.std(values):.3f}"


def table(runs: dict[str, dict[str, list[dict]]], only: list[str] | None = None) -> str:
    out = []
    for split, variants in runs.items():
        variants = {name: r for name, r in variants.items() if not only or name in only}
        if not variants:
            continue
        first = next(iter(variants.values()))[0]
        slides = first["train"]["slides"]
        train = slides if len(slides) <= 3 else f"{len(slides)} slides"
        out.append(f"### {split}: {train} to {first['test']['slide']}")
        out.append(
            f"train {first['train']['spots']} spots, test {first['test']['spots']} spots, "
            f"{len(first['genes'])} genes. Mean and standard deviation over seeds.\n"
        )
        header = ["variant", "method"] + [label for _, label in SCORES]
        out.append("| " + " | ".join(header) + " |")
        out.append("|" + "---|" * len(header))
        for variant, results in variants.items():
            for method in METHODS:
                if method not in results[0]["imputation"]:
                    continue
                cells = [fmt([r["imputation"][method][key] for r in results]) for key, _ in SCORES]
                out.append(f"| {variant} | {method} | " + " | ".join(cells) + " |")
        out.append("")
        out.append("| variant | epochs | recall@10 | near recall@10 | median rank |")
        out.append("|---|---|---|---|---|")
        for variant, results in variants.items():
            row = [r["retrieval"]["image_to_expr"] for r in results]
            epochs = fmt([float(r["epochs"]) for r in results])
            out.append(
                f"| {variant} | {epochs} | {fmt([x['recall@10'] for x in row])} "
                f"| {fmt([x['near_recall@10'] for x in row])} "
                f"| {fmt([x['median_rank'] for x in row])} |"
            )
        chance = first["retrieval"]["chance"]
        out.append(
            f"| chance | | {chance['recall@10']:.3f} | {chance['near_recall@10']:.3f} "
            f"| {chance['median_rank']:.0f} |\n"
        )
    return "\n".join(out)


def across_splits(runs: dict[str, dict[str, list[dict]]], variant: str = "default") -> str:
    """One row per method: the score of each split (a mean over its seeds), then their mean and
    standard deviation over the splits, for instance over the held out patients."""
    scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    splits = [s for s, variants in runs.items() if variant in variants]
    for split in splits:
        results = runs[split][variant]
        for method in METHODS:
            if method in results[0]["imputation"]:
                for key, _ in SCORES:
                    values = [r["imputation"][method][key] for r in results]
                    scores[method][key].append(float(np.mean(values)))
    header = ["method"] + [label for _, label in SCORES]
    out = [f"### {variant}: mean and standard deviation over {len(splits)} splits", ""]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "---|" * len(header))
    for method, by_key in scores.items():
        out.append(f"| {method} | " + " | ".join(fmt(by_key[key]) for key, _ in SCORES) + " |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--across"]
    folder = Path(args[0] if args else "results")
    runs = load(folder)
    if "--across" in sys.argv[1:]:
        print(across_splits(runs, *args[1:2]))
    else:
        print(table(runs, args[1:]))
