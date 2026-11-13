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

## Results

On the two Visium sections, first. Pearson correlation between predicted and measured expression
on the test spots, default settings, mean and standard deviation over three seeds. The two
baselines have no seed and give the same score every time.

| split | method | all genes | top 50 genes | per spot |
|---|---|---|---|---|
| `a1_block` | aligned space, kNN | 0.135 ± 0.004 | 0.119 ± 0.005 | 0.141 ± 0.006 |
|  | raw features, kNN | 0.148 | 0.154 | 0.163 |
|  | raw features, ridge | 0.135 | 0.132 | 0.167 |
|  | control: aligned space, shuffled pairs | 0.021 ± 0.010 | 0.038 ± 0.036 | 0.021 ± 0.011 |
| `a1_to_a2` | aligned space, kNN | 0.140 ± 0.005 | 0.221 ± 0.008 | 0.170 ± 0.007 |
|  | raw features, kNN | 0.138 | 0.210 | 0.174 |
|  | raw features, ridge | 0.137 | 0.196 | 0.145 |
|  | control: aligned space, shuffled pairs | -0.006 ± 0.006 | -0.001 ± 0.003 | -0.012 ± 0.006 |
| `a2_to_a1` | aligned space, kNN | 0.134 ± 0.013 | 0.164 ± 0.025 | 0.173 ± 0.009 |
|  | raw features, kNN | 0.159 | 0.172 | 0.183 |
|  | raw features, ridge | 0.134 | 0.122 | 0.143 |
|  | control: aligned space, shuffled pairs | -0.013 ± 0.007 | -0.035 ± 0.007 | -0.008 ± 0.016 |

Retrieval: for each test image, the rank of its own spot among all test spots by similarity in the
aligned space. In brackets, what a random ranking gives.

| split | recall@10 | near recall@10 | median rank |
|---|---|---|---|
| `a1_block` | 0.054 ± 0.001 (0.022) | 0.278 ± 0.014 (0.167) | 172 (382) |
| `a1_to_a2` | 0.029 ± 0.003 (0.003) | 0.180 ± 0.014 (0.029) | 416 (1999) |
| `a2_to_a1` | 0.019 ± 0.007 (0.003) | 0.135 ± 0.024 (0.031) | 634 (1938) |

What the numbers say:

- The control sits near zero. Trained on shuffled pairs, with the same genes, split and scoring, the
  aligned space scores between -0.035 and 0.038: the scores above come from the pairing of image and
  expression.
- From section 1 to section 2 (`a1_to_a2`) the aligned space does about as well as the raw features:
  0.140 against 0.138 (kNN) and 0.137 (ridge) over all genes, 0.221 against 0.210 and 0.196 over the
  50 most variable genes. Per spot it is a little lower than the raw kNN (0.170 against 0.174). The
  gaps are about as large as the spread over seeds (0.005).
- In the other direction (`a2_to_a1`) and on the strip of one section (`a1_block`), the kNN on the
  raw features is better over all genes: 0.159 against 0.134 and 0.148 against 0.135.
- The three variants (neighbour mask, input dropout, sigmoid loss) do not help: against the default,
  the score over all genes changes by -0.034 at worst and 0.000 at best. Their tables are in
  `results/tables.md`.
- So the contrastive step does not predict expression better than a kNN on the features it starts
  from. It does build a shared space: across sections the right spot is among the ten closest for
  2.9% of the images in one direction and 1.9% in the other, where a random ranking gives 0.3%.

## Across patients

The two Visium sections above are neighbours in one block, the easy case. On HER2ST each of the 8
patients is held out in turn: the model is trained on the sections of the seven others and scored on
all the sections of that patient, three seeds each. Mean and standard deviation over the 8 patients,
each patient counting once (the mean of its three seeds).

| method | all genes | top 50 genes | per spot |
|---|---|---|---|
| aligned space, kNN | 0.058 ± 0.030 | 0.060 ± 0.030 | 0.096 ± 0.046 |
| raw features, kNN | 0.052 ± 0.029 | 0.056 ± 0.031 | 0.084 ± 0.046 |
| raw features, ridge | 0.066 ± 0.032 | 0.073 ± 0.035 | 0.107 ± 0.054 |
| control: aligned space, shuffled pairs | 0.003 ± 0.005 | 0.001 ± 0.005 | 0.011 ± 0.027 |

Each patient on its own, over all genes, and the retrieval recall@10 of the aligned space (in
brackets, what a random ranking gives). The files in `results/her2st/` hold the spread over seeds.

| patient | test spots | aligned space, kNN | raw features, kNN | raw features, ridge | control | recall@10 |
|---|---|---|---|---|---|---|
| A | 2,065 | 0.031 | 0.030 | 0.035 | 0.002 | 0.014 (0.005) |
| B | 1,712 | 0.119 | 0.111 | 0.124 | 0.013 | 0.029 (0.004) |
| C | 1,086 | 0.083 | 0.077 | 0.097 | 0.005 | 0.041 (0.012) |
| D | 1,833 | 0.053 | 0.041 | 0.069 | 0.000 | 0.025 (0.005) |
| E | 1,729 | 0.023 | 0.025 | 0.027 | -0.002 | 0.012 (0.006) |
| F | 2,098 | 0.029 | 0.019 | 0.029 | 0.000 | 0.018 (0.006) |
| G | 1,371 | 0.067 | 0.060 | 0.076 | -0.004 | 0.052 (0.007) |
| H | 1,726 | 0.061 | 0.054 | 0.071 | 0.009 | 0.034 (0.003) |

What the numbers say:

- The control sits at zero: 0.003, 0.001 and 0.011 on the three scores.
- The aligned space does better than a kNN on the raw features for 7 of 8 patients over all genes
  (+0.006 on average), for 6 of 8 on the 50 most variable genes and for 7 of 8 per spot. The gain is
  small next to the spread between patients (0.030).
- A ridge regression on the raw features does better than the aligned space for 7 of 8 patients over
  all genes (+0.008 on average) and for 8 of 8 on the 50 most variable genes.
- The shared space is not empty. For every patient the right spot is among the ten closest more
  often than a random ranking would put it, 1.9 to 9.7 times as often.
- The scores are low, 0.06 on average over all genes, and not comparable with the Visium numbers
  above: other arrays, another resolution, and a patient the model has never seen.

## Data

Two sections of one 10x Genomics Visium human breast cancer sample (fresh frozen, whole
transcriptome, Space Ranger 1.1.0): `V1_Breast_Cancer_Block_A_Section_1` and
`V1_Breast_Cancer_Block_A_Section_2`, with 3,798 and 3,987 spots under tissue and 36,601 genes.
The data is released by 10x Genomics under CC BY 4.0.

HER2ST (Andersson et al., Nature Communications 2021, https://doi.org/10.1038/s41467-021-26271-2)
is used for the split by patient: 36 sections of 8 patients with HER2-positive breast tumours, on
Spatial Transcriptomics arrays (spots of 100 microns, 200 microns apart), 13,620 spots under the
tissue and 14,861 to 15,842 genes per section, 11,871 of them in every section. The images are
JPEGs. Their pixel size, between 0.7 and 1.0 microns, is worked out from the distance between
neighbouring spots, nothing in the data gives it. The data is released under CC BY 4.0 on Zenodo
(https://doi.org/10.5281/zenodo.3957257). The four archives are encrypted; the authors publish the
passwords in https://github.com/almaan/her2st and `histoalign unpack` uses them.

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

For the split by patient, the four zip files of the HER2ST record (0.6 GB) go in `data/her2st`:

```
mkdir -p data/her2st
for f in count-matrices images meta spot-selections; do
  curl -L -o data/her2st/$f.zip "https://zenodo.org/records/3957257/files/$f.zip?download=1"
done
uv run histoalign unpack data/her2st
uv run histoalign features --her2st data/her2st/raw --out data/features/her2st
uv run python scripts/her2st_folds.py
uv run python scripts/results_table.py results/her2st --across
```

Unpacking takes about 30 seconds. Features for the 36 sections took about 18 minutes and the 24
runs of `her2st_folds.py` about 19 minutes, as three processes side by side, on a laptop that was
busy with other work. A run takes 84 seconds at the median, most of it spent on the 12,000
training spots.

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

On HER2ST there is one fold per patient. The sections of the seven other patients are the training
slides, each one a slide of its own for the gene selection and the validation strip, and all the
sections of the held out patient are the test, scored together. Sections of one patient are
neighbours in the tissue, so they are never split between training and test.

Scores are Pearson correlations across the test spots, one per gene, averaged over all genes,
over the 50 genes with the largest variance in the test slide, and per spot across genes.
Retrieval asks which of the test spots a given image belongs to; "near" accepts any of the top
candidates within 1.5 spot pitches of the right one.

## Limits

- The two Visium sections come from the same block, they are neighbours in the tissue, so the
  cross-section numbers are optimistic. The HER2ST folds split by patient, but all eight patients
  have HER2-positive breast tumours and one kind of array: nothing here says how the model
  transfers to another tissue, protocol or scanner.
- The image encoder is frozen. The heads are trained, the encoder is not fine-tuned.
- A 224 px patch at 0.5 microns per pixel covers about 112 microns, a Visium spot is 55 microns:
  each patch also shows the neighbourhood of the spot. On HER2ST the spots are 100 microns wide and
  the images 0.7 to 1.0 microns per pixel, so the patch is about 110 to 160 pixels wide before it
  is enlarged to 224.
- In the Visium tables the standard deviations are over seeds (three per variant), not over
  sections. In the HER2ST table they are over the 8 patients, each patient being the mean of three
  seeds.
- The mean correlation over a thousand genes is dominated by genes with little signal; the
  files in `results/` hold the score of each marker.

## Layout

`src/histoalign`: `visium.py` (reader), `her2st.py` (HER2ST reader and decryption), `patches.py`
(crops), `genes.py` (selection and scaling), `encoders.py` (frozen encoders, HDF5 cache),
`features.py`, `models.py` (heads), `losses.py`, `train.py`, `retrieval.py`, `predict.py` (kNN,
ridge, scores), `experiment.py`, `cli.py`. `scripts/` has the runs and the tables.
`tests/` builds small synthetic slides on the fly.
