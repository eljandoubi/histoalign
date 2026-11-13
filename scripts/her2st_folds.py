"""Leave one patient out on HER2ST: the sections of the seven other patients train the model and
all the sections of the eighth are the test, three seeds for each of the eight patients.

The features must be cached first:
    uv run histoalign unpack data/her2st
    uv run histoalign features --her2st data/her2st/raw --out data/features/her2st
    uv run python scripts/her2st_folds.py

Every slide is read once and shared by the 24 runs. A result that is already in the output
folder is skipped, so an interrupted run can be started again.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from histoalign import her2st
from histoalign.cli import code_version, split_by_patient
from histoalign.experiment import pool, prepare_slide, run
from histoalign.train import Config

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--root", default="data/her2st/raw")
parser.add_argument("--features", default="data/features/her2st")
parser.add_argument("--out", default="results/her2st")
parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
args = parser.parse_args()

slides = {
    name: prepare_slide(her2st.read_section(args.root, name), args.features)
    for name in her2st.sections(args.root)
}
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

for patient in sorted({her2st.patient(name) for name in slides}):
    train_names, test_names = split_by_patient(args.root, patient)
    train_sets = [(slides[n], np.arange(len(slides[n].coords))) for n in train_names]
    test = pool([slides[n] for n in test_names], f"patient {patient}")
    for seed in args.seeds:
        path = out / f"{patient}__default__seed{seed}.json"
        if path.exists():
            continue
        result = run(train_sets, (test, np.arange(len(test.coords))), Config(seed=seed), log=None)
        result["code"] = code_version()
        path.write_text(json.dumps(result, indent=1))
        print(path, flush=True)
