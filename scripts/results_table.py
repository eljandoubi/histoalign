"""Print the results folder as markdown: mean and spread over seeds, one table per split.

uv run python scripts/results_table.py results
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


def table(runs: dict[str, dict[str, list[dict]]]) -> str:
    out = []
    for split, variants in runs.items():
        first = next(iter(variants.values()))[0]
        out.append(f"### {split}: {first['train']['slides']} to {first['test']['slide']}")
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


if __name__ == "__main__":
    print(table(load(Path(sys.argv[1] if len(sys.argv) > 1 else "results"))))
