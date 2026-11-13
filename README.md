# histoalign

Align H&E patches with spatial gene expression. A frozen pathology encoder turns the patch around
each Visium spot into a vector, a small network does the same for the expression of that spot, and
two projection heads trained with a contrastive loss put both in one space. The aligned space is
then used to guess the expression of spots from the image alone.

The evaluation matters as much as the model. Every score sits next to a simple baseline and a
control, and the way the data is split is written down.

## Pipeline

1. `histoalign download` fetches 10x Visium samples: counts (HDF5), spatial folder, full
   resolution H&E.
2. `histoalign features` cuts a 224 px patch at 0.5 microns per pixel around every spot, runs a
   frozen ViT and caches the vectors in HDF5, with the spot table in Parquet.
3. `histoalign run` picks genes and fits the scaling on the training spots only, trains the two
   heads (InfoNCE or the SigLIP sigmoid loss, with an option to leave neighbouring spots out of
   the negatives), chooses the number of epochs on a held-out strip of the training slides, then
   scores the test spots.

Three reference points are computed on the same split: kNN on the raw image features, ridge
regression on the same features, and the aligned model trained on shuffled pairs, which should
score close to zero.

## Data

Two sections of one 10x Genomics Visium human breast cancer sample (fresh frozen, whole
transcriptome, Space Ranger 1.1.0): `V1_Breast_Cancer_Block_A_Section_1` and
`V1_Breast_Cancer_Block_A_Section_2`, with 3,798 and 3,987 spots under tissue and 36,601 genes.
The data is released by 10x Genomics under CC BY 4.0.

The image encoder is the ViT-S/16 that Lunit trained with DINO on TCGA patches (Kang et al., CVPR
2023), loaded from their GitHub release. The code also loads a Hugging Face ViT by name
(`--encoder owkin/phikon`, for instance), but that path has not been run: the weights could not be
downloaded from the network these experiments ran on.

## Reproduce

```
uv sync
uv run histoalign download V1_Breast_Cancer_Block_A_Section_1 V1_Breast_Cancer_Block_A_Section_2
uv run histoalign features data/raw/V1_Breast_Cancer_Block_A_Section_1 \
    data/raw/V1_Breast_Cancer_Block_A_Section_2 --out data/features/lunit
sh scripts/run_all.sh
uv run python scripts/results_table.py results
```

The download is 3.6 GB, mostly the two H&E images. Features for both slides take about six
minutes on an M-series Mac. `run_all.sh` took about an hour for its 36 runs on a laptop that was
busy with other work; training and scoring one run takes 47 seconds at the median.
Each file in `results/` holds the whole output of one run, its configuration and the hash of the
commit that produced it. The tests need no download: `uv run pytest`.

## How the test spots are kept out

Nothing about the test spots is used to choose genes, to fit the scaling, to pick the number of
epochs or to set a hyperparameter. The genes are the 1,000 most variable ones of the training
spots (plus ten breast cancer markers when every slide has them), the scaling is the training
mean and standard deviation, and the epochs come from the loss on a strip at the edge of the
training slide with two spot pitches left out around it. The other settings are the defaults of
`histoalign run`; the variants in the tables are reported for information, not tuned.

Three splits:

- `a1_block`: section 1 alone, trained on its left side and tested on a strip at its right edge,
  with two spot pitches between them.
- `a1_to_a2` and `a2_to_a1`: trained on one section, tested on the other.

Scores are Pearson correlations across the test spots, one per gene, averaged over all genes,
over the 50 genes with the largest variance in the test slide, and per spot across genes.
Retrieval asks which of the test spots a given image belongs to; "near" accepts any of the top
candidates within 1.5 spot pitches of the right one.

## Limits

- Both sections come from the same block, they are neighbours in the tissue. They are not two
  samples of a patient population, so the cross-section numbers are optimistic and nothing here
  says how the model transfers to another patient, tissue or scanner. A split by patient needs
  more than one patient.
- The image encoder is frozen. The heads are trained, the encoder is not fine-tuned.
- A 224 px patch at 0.5 microns per pixel covers about 112 microns, a Visium spot is 55 microns:
  each patch also shows the neighbourhood of the spot.
- The standard deviations are over seeds (three per variant), not over patients or sections.
- The mean correlation over a thousand genes is dominated by genes with little signal; the
  files in `results/` hold the score of each marker.

## Layout

`src/histoalign`: `visium.py` (reader), `patches.py` (crops), `genes.py` (selection and scaling),
`encoders.py` (frozen encoders, HDF5 cache), `features.py`, `models.py` (heads), `losses.py`,
`train.py`, `retrieval.py`, `predict.py` (kNN, ridge, scores), `experiment.py`, `cli.py`.
`tests/` builds small synthetic slides on the fly.
