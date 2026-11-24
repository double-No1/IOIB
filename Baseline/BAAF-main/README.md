# BAAF: Universal Transformation of One-Class Classifiers for Unsupervised Image Anomaly Detection

Official implementation of **Bootstrap Aggregation Anomaly Filtering (BAAF)**, ECCV 2026.

[ECCV](https://eccv.ecva.net/virtual/2026/poster/5489) · [arXiv](https://arxiv.org/abs/2602.13091) · [PDF](https://arxiv.org/pdf/2602.13091)

> Declan McIntosh and Alexandra Branzan Albu
> University of Victoria

BAAF turns any one-class classifier (OCC) anomaly detector into a fully unsupervised one: it filters anomalies out of a corrupted training set using bagged instances of the detector itself, then trains the unmodified detector on the filtered set. Inference is unchanged.

<p align="center">
  <img src="assets/fig1_qualitative.png" width="720" alt="PatchCore with and without BAAF on training corruptions.">
</p>

## Method

1. Split the training images into *n* non-overlapping bags.
2. Train an independent copy of the OCC on each bag.
3. Each copy predicts on images it was not trained on; scores are min-max normalized and a two-component Gaussian mixture (weighted toward nominal predictions) sets the filter threshold at the crossover of the two Gaussians.
4. Majority-vote across bags and *K* independent repeats ("votes"); images voted anomalous are removed.
5. Train one final OCC on the filtered set.

Notation: **BAAF(K/n)+OCC**, e.g. `BAAF(1/4)+PatchCore` is one vote, four bags. Training costs about *K·n + 1* OCC fits; inference cost is unchanged.

<p align="center">
  <img src="assets/fig2_method.png" width="720" alt="BAAF pipeline with n=2 bags.">
</p>

## Installation

Requires **Python 3.12**. A GPU is recommended for PatchCore and EfficientAD.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .

# Determinism env vars; must be set before Python starts.
set -a && source baaf_deterministic.env && set +a
```

CPU-only machines: install torch/torchvision from `https://download.pytorch.org/whl/cpu` instead. Verify the install with `python -m baaf --help`.

EfficientAD teacher weights download automatically on first use. For the optional ImageNet penalty term (paper default), place [ImageNette](https://github.com/fastai/imagenette) at `./datasets/imagenette/imagenette2` or pass `--imagenet-path`.

## Data

Datasets use the MVTec AD folder layout (`<category>/{train/good, test/<defect>, ground_truth/<defect>}`, PNG images):

| Dataset | Source | Preparation |
| --- | --- | --- |
| [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) | Official site | Use as-is |
| [VisA](https://github.com/amazon-science/spot-diff) | Amazon | `python -m baaf prepare-visa --source ... --output ...` |
| [MVTec LOCO AD](https://www.mvtec.com/company/research/datasets/mvtec-loco) | Official site | `python -m baaf prepare-loco --source ... --output ...` |

**Corruption protocol.** Benchmarks use the overlapping unsupervised setting from the paper: `--corruption 0.1` copies test anomalies amounting to 10% of the training-set size into `train/good` (the same anomalies remain in the test set). `--corruption 0` is the clean OCC case; `--votes 0` skips filtering (OCC baseline).

## Usage

Single category, paper default `BAAF(1/4)+PatchCore`:

```bash
python -m baaf run --data-dir ./datasets/MVTecAD --category bottle \
  --method patchcore --votes 1 --bags 4 --corruption 0.1 --seed 0 --gpu 0
```

Full benchmarks and summary:

```bash
# MVTec AD (SOTA config in the paper: 3 votes, 4 bags)
python -m baaf benchmark --dataset mvtec --data-dir ./datasets/MVTecAD \
  --votes 3 --bags 4 --corruption 0.1 --seed 0 --gpu 0 --skip-existing

# Logical anomalies on MVTec LOCO AD with EfficientAD
python -m baaf benchmark --dataset loco --data-dir ./datasets/LocoAD_MVTec \
  --method efficientad --votes 1 --bags 4 --corruption 0.1 --gpu 0 --skip-existing

python -m baaf summarize ./results --dataset mvtec
```

Each run writes a JSON with I-AUROC, P-AUROC, AUPRO, and filter precision/recall. Pixel metrics follow the paper: maps and masks are resized to 256×256 and center-cropped to 224×224. `--save-qualitative` writes overlay images; `--save-gmm-plots` writes per-bag threshold plots.

| Flag | Paper name | Default | Notes |
| --- | --- | ---: | --- |
| `--bags` | *n* | 4 | 2 bags under-filter on MVTec AD; 8 starve small classes |
| `--votes` | *K* | 1 | 3 is slightly stronger and ~3× slower; 5 does not help |
| `--corruption` | — | 0.1 | Injected training anomaly rate |
| `--seed` | — | 0 | Seeds bags, GMM, and detectors |

Python API:

```python
from baaf import run_baaf
from wrappers.patchcore import PatchCoreWrapper

_, path, results = run_baaf(
    category="bottle", method=PatchCoreWrapper(), data_dir="./datasets/MVTecAD/",
    seed=0, k=4, votes=1, num_per_negative=0.1, output_dir="./results/",
)
print(results["img_results"]["auroc"])
```

This repo ships two runnable detectors, `--method patchcore` (WideResNet-101 + ResNeXt-101 + DenseNet-201 ensemble, 320², 1% coreset) and `--method efficientad` (medium PDN, 256², 70k steps). The remaining paper detectors were wrapped with the same three-method interface; cite them via [`CITATION.bib`](CITATION.bib).

## Wrapping your own OCC method

Subclass `AD_Method_Wrapper` (in `wrappers/base.py`) and implement three methods; BAAF never inspects the model, it only changes which images are in the training folder.

```python
from wrappers.base import AD_Method_Wrapper

class MyDetector(AD_Method_Wrapper):
    def reset(self):
        """Build a fresh, untrained model. Called before every fit."""

    def train_on_class(self, seed, data_path):
        """Fit on PNGs in `data_path/train/good/`. Never read test labels."""

    def predict_on_image(self, image_path):
        """Return (image_score: float, anomaly_map: np.ndarray of shape (1, H, W)).
        Higher scores are more anomalous; filtering uses only the image score."""
```

Your wrapper holds the model as instance state; BAAF repeatedly calls, on the same instance:

1. `reset()` — discard any previous model and build a fresh, untrained one. Called before every fit (each bag and the final model), so no state may leak between fits.
2. `train_on_class(seed, data_path)` — `data_path` is a complete MVTec-layout category folder with a trailing slash; fit on the PNGs in `data_path + "train/good/"` and use `seed` for the method's own RNG.
3. `predict_on_image(image_path)` — score a single PNG with the most recently trained model.

**Why data is staged as temporary folders on disk.** Every training set BAAF constructs — the corrupted category and each bag — is materialized as a real MVTec-layout category folder (temporary copies written next to the source data and deleted as soon as that bag or run finishes). This is deliberate: `train_on_class` receives only a folder path, so BAAF stays agnostic to however the wrapped OCC loads data. A detector can keep its existing dataset class, dataloader, augmentation pipeline, or even its original training CLI completely unchanged — it never has to accept in-memory arrays or conform to a loader API. The cost is transient disk space of a few extra copies of one category.

Pass your wrapper as `--method module.path:ClassName`. A CPU-only template that exercises the full pipeline ships as `examples/custom_method.py`. If the wrapped method selects checkpoints or thresholds using the test set, remove that path first — neither the paper nor this code ever touches test labels during training.

For custom datasets, point `--data-dir` at a root of MVTec-layout category folders. Without a labelled test set, use `--corruption 0 --no-eval`; the training folder is treated as an uncurated stream and filtered.

**Reproducibility.** With the pinned environment, `baaf_deterministic.env` sourced, and a fixed `--seed`, re-runs on the same machine reproduce exactly; a different GPU, driver/CUDA version, Python version, or OS will change results for the same seed.

## Repository layout

```
baaf/                    BAAF algorithm, GMM threshold, data, metrics, CLI
wrappers/                OCC interface + PatchCore and EfficientAD wrappers
methods/                 Vendored detectors (official PatchCore; nelson1425 EfficientAD)
examples/                CPU-only template wrapper
logs/                    Scripts that produced the shipped reference results
results/                 Reference benchmark runs (JSON)
assets/                  Figures used in this README
baaf_deterministic.env   Env vars to source before launching Python
```

## Citation

```bibtex
@inproceedings{mcintosh2026baaf,
  title     = {BAAF: Universal Transformation of One-Class Classifiers for Unsupervised Image Anomaly Detection},
  author    = {McIntosh, Declan and Branzan Albu, Alexandra},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

If you report results with a wrapped OCC method, please also cite that method ([`CITATION.bib`](CITATION.bib)).

## License

BAAF code is MIT. Vendored code under `methods/patchcore` and `methods/EffecientAd` is Apache-2.0. See [`LICENSE`](LICENSE).
