# CT-MUSIQ Research

Research code and thesis materials for no-reference quality assessment of low-dose CT images. The project investigates a multi-scale transformer (CT-MUSIQ), a convolutional/transformer ensemble, and their agreement with radiologist quality scores.

**Research status:** The original thesis and defense notes are historical drafts, not verified results. The repository's original 0900–0999 test split is drawn from the challenge's training collection and contains images highly similar to training images. Its results cannot be ranked directly against the official challenge leaderboard. See [the reproducibility audit](research_audit/README.md) before using any reported result.

The current compact 448-pixel baseline achieves a released-test aggregate of
2.7200 ± 0.0133 across three training seeds, versus 2.4617 ± 0.0179 for the
earlier transformer pipeline. These are exploratory results on one fixed
300-image collection, with a provisional grouped development split. The
compact architecture is an established baseline; no novelty, clinical
readiness or state-of-the-art claim is made. [Results and limitations](research_audit/README.md).

## Layout

- `ct_musiq/` — model, data pipeline, training, evaluation and ablation code.
- `dataset/` — training labels and local image files in `dataset/image/`; the released 300-image test set is in `dataset/challenge_test/`. Both image directories and the local test labels are not tracked by Git.
- `research_audit/` — exploratory evaluation and image-grouping tools, with documented limitations.
- `thesis/` — original thesis source and artifacts, retained as historical material.
- `docs/` — original defense preparation notes, retained as historical material.
- `index.html`, `site.js`, `site.css` — documentation viewer; serve the repository root over HTTP.
- `results/` — generated checkpoints, logs, and predictions (not tracked).

## Environment

Create a Python environment, install a PyTorch build suitable for your machine, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

The original code expects 1,000 TIFF images named `0000.tif`–`0999.tif` in `dataset/image/`, with labels in `dataset/train.json`. Its filename-based split is retained for reproducing historical experiments; it should not be used as evidence of independent patient-level evaluation.

## Original entry points

Run commands from the repository root:

```bash
python -m ct_musiq.train --help
python -m ct_musiq.evaluate --help
python -m ct_musiq.ablation --help
```

Training and evaluation produce artifacts in `results/`. Inspect the configuration and experimental protocol before running; these commands retain the original study's data split and do not automatically implement the released 300-image challenge test set.

## Reproducibility work

The [audit notes](research_audit/README.md) describe the released test data, exploratory checkpoint evaluation, and provisional grouping of visually related training images. Inferred image groups are not patient identifiers. The original data and clinical claims require verification before publication.

The challenge dataset and original evaluation protocol are described by [Lee et al.](https://doi.org/10.1016/j.media.2024.103343).
