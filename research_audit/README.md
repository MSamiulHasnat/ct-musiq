# Reproduction audit

See [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) for the current
development-run protocol and limitations, and [CLAIMS.md](CLAIMS.md) for the
evidence supporting or contradicting proposed paper claims.

## Completed conditional-ranking study, 2026-11-25

All nine declared arms and all three seeds are complete (27 models), with
300 released LDCT test cases, 1,666 annotated external CBCT-IQ cases and
1,509 nonambiguous expert reconstruction pairs per model. No external labels
fit or calibrate a predictor. Adaptive controls are explicitly exploratory.

| Configuration | LDCT aggregate | External signed Spearman | Expert-pair agreement |
|---|---:|---:|---:|
| Scalar baseline | 2.7200 ± 0.0133 | −0.142 ± 0.253 | 42.5% ± 15.8% |
| Scalar conditional blur .75 | 2.7197 ± 0.0040 | 0.348 ± 0.307 | 74.5% ± 14.9% |
| Scalar unconditional blur .75 | 2.7070 ± 0.0139 | −0.075 ± 0.485 | 49.3% ± 26.4% |
| Ordinal baseline | 2.6931 ± 0.0201 | −0.222 ± 0.328 | 37.6% ± 21.3% |
| Ordinal conditional blur 1.5 | 2.6912 ± 0.0194 | 0.426 ± 0.255 | 73.7% ± 14.4% |
| Ordinal unconditional blur 1.5 | 2.6928 ± 0.0236 | 0.391 ± 0.063 | 72.8% ± 5.9% |
| Ordinal conditional blur .75 | 2.7016 ± 0.0168 | −0.097 ± 0.215 | 48.4% ± 9.2% |
| ResNet-18 baseline | 2.6256 ± 0.0075 | −0.603 ± 0.130 | 3.4% ± 5.5% |
| ResNet-18 conditional blur .75 | 2.6064 ± 0.0371 | −0.470 ± 0.116 | 11.8% ± 11.1% |

All ± values are sample SD across three fixed training seeds. They are not
patient confidence intervals. Quality conditioning helps in the scalar
mild-blur comparison, but the strong ordinal gate contrast includes zero
in both within-phantom bootstrap intervals. ResNet transfer remains poor.
The ordinal mild-blur model passes the illustrated synthetic checks yet
transfers poorly: synthetic consistency is not a substitute for human-rated
external evaluation. No SOTA, noninferiority or clinical-readiness claim follows.

[Machine-readable summary](results/robustness_summary.json),
[all checkpoint hashes and environment](results/completed_study_manifest.json),
[paired external intervals](results/external_gpu/paired_source_slice_bootstrap.json),
and [working paper](../paper/main.pdf) retain the complete evidence.
Final GPU results are in `results/external_gpu/`; CPU and initially rotated
stored-array results are retained separately, not pooled into the final table.
The fixed axial correction was based on NIfTI geometry and visual QA.

The complete export (251 files without images or weights) has SHA-256
`b455720f959746baad8d1a42ec1ba8b60622b7cd6ad90c092c9d2a01bc9d3fd8`.
All endpoint model hashes match, all nine unit tests pass, and the generic
backbone engine exactly reproduces the reference model on all 1,666 cases.
The remote queue has finished; there are no further planned experiment runs.

## Fresh-run interim results

### Compact follow-up completed 2026-11-25

The fixed 448-pixel EfficientNet-B0 protocol was repeated with three seeds
after selection on development validation. No test-based calibration, score
rounding, seed selection or ensemble was used. All runs use the same 790/210
provisional inferred-group partition; these are not verified patient IDs.

| Run | Best validation aggregate | Test aggregate | Test MAE |
|---|---:|---:|---:|
| Compact 448, seed 42 | 2.8342 | 2.7287 | 0.3014 |
| Compact 448, seed 123 | 2.8318 | 2.7047 | 0.3029 |
| Compact 448, seed 456 | 2.8315 | 2.7266 | 0.2987 |
| Aligned transformer, seed 42 | 2.7888 | 2.5885 | 0.4050 |

The compact three-seed test aggregate is **2.7200 ± 0.0133**, and MAE is
**0.3010 ± 0.0021** (sample standard deviations). Its mean aggregate exceeds
the earlier batch-32 transformer's 2.4617 by 0.2583. These pipelines differ
in backbone, preprocessing, augmentation, prediction head and optimization;
the comparison does not identify which component caused the improvement.
The aligned transformer has only one seed and does not establish stability.

The compact model has 4,335,741 trainable parameters versus 88,325,379 for
the aligned transformer (20.4 times fewer). In one RTX 4090 FP32 benchmark,
GPU forwards at batch 32 took 24.91 ms versus 33.66 ms per batch, about 1.35
times the throughput. At batch 1, forwards took 1.95 ms versus 2.64 ms.
These exclude preprocessing, loading and transfers. Peak allocated GPU memory
at batch 32 was higher for the compact model (1.72 GB versus 0.73 GB), so fewer
parameters must not be described as lower activation memory at every batch.
See `results/screening/benchmark.json` and `benchmark_screening.py`.

The result archive SHA-256 is
`965c51c85e21f524abdf408f5b24071070f0d2541721966ff81a6abfa287c71c`;
extracted predictions, metrics, training logs and launcher exit records are
under `results/screening/`. Training and all four declared evaluations exited
successfully. The frozen training source is recorded in the protocol.
The compact checkpoints remain on the SSH server under
`/home/samiul/ct-musiq-repro-20261124/development_screening/results/compact448_seed{42,123,456}/best.pth`;
their hashes are in `results/screening/remote_checkpoint_sha256.txt`.
Slow downloads were stopped and incomplete local checkpoint copies removed.
`python -m research_audit.summarize_screening` recomputes descriptive metrics
from all saved predictions, verifies the shared image IDs and targets, and
checks agreement with each saved evaluation record.

This is a stronger reproduced baseline, **not a novel-method or SOTA claim**.
EfficientNet-B0 at 448 pixels is already a comparator in the
[MedSigLIP author repository](https://github.com/itu-biai/medsiglip-ldct-iqa),
which reports 2.6810 ± 0.007 under a different five-fold protocol.
[ClinReadNet](https://doi.org/10.1016/j.neunet.2025.108535), published in
Neural Networks in 2026, reports 2.7690. Those results are literature context,
not matched local replications. A useful contribution and fair comparisons
remain necessary before claiming readiness for a strong journal. Simple
backbone substitution alone is insufficient.

### Precision control: no clear aggregate benefit

The declared control quantizes clipped pixel values to 256 levels before
the same floating-point resize and otherwise preserves the compact protocol.
Its test aggregates for seeds 42, 123 and 456 are 2.7223, 2.7108 and 2.7178,
respectively. The mean is **2.7170 ± 0.0058**, versus **2.7200 ± 0.0133**
for float inputs. The mean difference is only +0.0030; this small study does
not establish a reliable aggregate improvement from retaining sub-byte
precision. It also does not prove equivalence. Quantized-input MAE averages
0.2882 ± 0.0128, compared with 0.3010 ± 0.0021 for float inputs. Do not
choose a revised primary protocol on those test errors.

This control does not explain the much larger transformer-to-compact gap.
The architectures, augmentation and optimization still differ, and the
legacy integer-resize path is not reproduced by this isolated quantization
arm. The float primary result remains the declared primary result.

All six precision-control training/evaluation processes exited successfully.
Their source snapshot is recorded in the protocol. Results and validation
predictions are in `results/precision_control/`; the archive SHA-256 is
`7bb81eb8349e6edc5198203617e2b9cefde728bffad28985703cea5478575134`.
Remote checkpoint hashes are saved there as well. With the new precision flag
disabled, an old float checkpoint produced byte-identical test prediction CSVs
before and after the code addition. Metrics recomputed from every prediction
file agree with the saved records within four-decimal rounding.

The limited screening and precision comparison are complete; no broad sweep
remains queued. A submission-ready novelty claim has not been established.

### Fixed three-model average (exploratory follow-up)

After a further improvement request, an equal average of the existing float
compact seeds 42,123,456 was evaluated on validation first. Its aggregate
2.841767 and MAE 0.173226 improve on each individual validation checkpoint.
The same fixed average then obtained test aggregate **2.728080** and MAE
**0.294445**, without fitted weights, rounding or calibration. The original
single-model mean remains 2.720026 and MAE 0.300991. This ensemble is one
predictor; it does not have a three-seed standard deviation of its own.

Its test aggregate is close to seed 42 alone (2.7287), and an exploratory
paired inferred-group bootstrap of ensemble minus seed 42 gives aggregate
difference interval [-0.0191,0.0190] and MAE difference [-0.0185,0.0036].
Neither establishes an improvement over that individual checkpoint. These
are inferred-slice-group intervals, not patient-level confidence intervals.
The ensemble needs three forward passes and 13,007,223 parameters in total;
wall time was not rebenchmarked. This modest gain over the single-model mean
does not establish a new architecture or a reliable SOTA result. Source-file
hashes and metrics are in `results/compact_ensemble_validation/` and
`results/compact_ensemble_test/`; use `ensemble_predictions.py` to reproduce.

### Earlier exploratory runs

These are exploratory results on the same released 300-image test archive;
the validation partition remains the provisional grouped 210-image set.

| Run | Best validation aggregate | Test aggregate | Test MAE |
|---|---:|---:|---:|
| Batch 32, seed 42 | 2.6878 | 2.4536 | 0.4447 |
| Batch 32, seed 123 | 2.6764 | 2.4493 | 0.4642 |
| Batch 32, seed 456 | 2.6500 | 2.4823 | 0.4867 |
| Adjusted batch 64, seed 42 | 2.7002 | 2.5039 | 0.9692 |
| Adjusted batch 64, seed 123 | 2.5701 | 2.4656 | 0.5181 |
| Archived Swin-T + ResNet50 ensemble | Historical selection | 2.5833 | 0.4054 |

Across the three batch-32 training seeds, the test aggregate is
2.4617 ± 0.0179 and MAE is 0.4652 ± 0.0210 (sample standard deviations).
These describe training-seed variation on one fixed test collection, not
population confidence intervals or three independent test datasets.

The initial, unadjusted batch-64 run stopped at epoch 21 with best validation
aggregate 0.9396. The adjusted batch-64 run changes EMA, minimum epochs and
checkpoint selection as documented in the protocol. Its validation MSE at its
best-correlation epoch was 1.2308, so its score offset is also visible on
development data. A higher correlation does not establish better absolute
score accuracy. The archived ensemble was trained under a different historical
partition and is not a matched retraining comparator.

The original repository's 0000–0999 dataset is the challenge's 1,000-image
training collection. Its 0900–0999 split is not the challenge's 300-image
test set. The original results must not be ranked directly against challenge
leaderboard results obtained on the other test set.

The archived ablation CSV records A1 as an error and has no A6 result. It
reports A2 aggregate 2.2327 and A4 2.3052 on the historical 100-image split,
not the improvement asserted in parts of the thesis. These runs used the
leaky historical split and do not establish a reliable KL-loss effect.

An exploratory evaluation of the archived CT-MUSIQ checkpoint on the released
300-image test set uses `evaluate_released_test.py`. The script retains the
repository's actual image preprocessing, loads weights strictly, uses no test
time augmentation or calibration, and writes per-image predictions and run
metadata. This is a check of an existing checkpoint, not a fresh training run
or a pre-registered final comparison.
The flat-layout source used on the server is preserved in
`snapshots/historical_evaluation_source.tar.gz` (SHA-256
`ab3e78a4d50a43d80a8083a71e9361eac1e934b2ba98c697b730ac34b0e33bd7`),
along with `snapshots/historical_released_test.py`, the evaluation entry point.

From the repository root, the local released test copy can be evaluated with:

```bash
python -m research_audit.evaluate_released_test --data dataset/challenge_test --checkpoint results/ct_musiq/ct_musiq_best_weights_only.pth --output results/exploratory_test
```

The checkpoint must be compatible with PyTorch's restricted
`weights_only=True` loader; the historical archive's original checkpoint
contains a NumPy scalar and must first be converted in a trusted environment.

The archived CT-MUSIQ checkpoint was evaluated on the released 300-image set
without test-time augmentation or calibration. Its Pearson, Spearman and
Kendall correlations were 0.8925, 0.8876 and 0.7116 (overall 2.4917); mean
absolute error was 0.4475 on the 0–4 scale. The per-image predictions and
environment metadata are in `results/exploratory_released_test/`. This is one
exploratory checkpoint, not an estimate from multiple training seeds or a
fresh grouped-validation study.
The transferred weights-only checkpoint was checked against the archived
checkpoint: all 164 model tensors matched exactly. Optimizer state was
omitted for transfer; a NumPy scalar in the metadata was converted to a
Python float for restricted loading.
A second evaluation on the same server produced a byte-identical prediction
CSV (SHA-256 `f8340f4c5b7763cbb41c52316da828d79fb9d650e838df64338ef2690515c8fb`).
That repeat command took 4.77 seconds wall time, including startup and file
loading; this is a single observed timing, not a cross-hardware benchmark.

In a post-hoc descriptive check, the mean signed error was +0.148 and RMSE
was 0.567. For the 79 images with target scores from 2 inclusive to 3
exclusive, mean signed error was +0.447 and MAE was 0.584. This is an
exploratory subgroup summary; the images may not be statistically independent.
The released 300 images also form 30 visually inferred groups of 10 at the
same similarity threshold. Resampling those groups 2,000 times gives an
exploratory percentile interval of 2.405–2.593 for the aggregate and
0.384–0.514 for MAE; see `group_bootstrap.json` and `bootstrap_test.py`.
These are slice-group intervals, not patient-level confidence intervals:
multiple groups can come from the same patient, so the intervals may still
understate uncertainty.
The correlation p-values in the saved metrics are ordinary image-level
outputs from SciPy; they assume independent observations and should not be
used as patient-level significance evidence for this dataset.

The test archive was obtained from the test-data link in
https://github.com/itu-biai/medsiglip-ldct-iqa and has SHA-256
`e017e8d555edb0193785f60efcdedce70f60cd2a2419deceb1aa5eda0582d4df`.
Its 300 labels match those independently available in
https://github.com/Surya-Hariharan/Deep-Learning-for-CT-Image-Quality-Assessment.
The official challenge paper describes patient-separated splits:
https://doi.org/10.1016/j.media.2024.103343.
The official scoring procedure sums the absolute Pearson, Spearman and Kendall
correlations across the test predictions:
https://ldctiqac2023.grand-challenge.org/submission-and-evaluation/.
The TIFFs are already normalized; this repository additionally converts them
to 8-bit before bicubic resizing. That model-specific preprocessing must be
reported, not described as the radiologists' display windowing.
An experiment that preserves floating-point detail may be worthwhile, but it
must be compared on the development split before choosing a final protocol.

The repository's Swin-T + ResNet50 baseline is not an exact reproduction of
the winning agaldran submission, which used Swin and BiT-ResNeXt50 with
multiple heads. Describe it as an independently implemented baseline.

For subsequent model development, identify source-slice and patient groups,
split those groups before selecting models, and keep the released test set out
of model selection. The TIFF files alone do not establish patient identities.

`development_split_seed42.json` is a reproducible provisional split of 790
training and 210 validation images. It keeps the inferred image groups
disjoint, but is not a verified patient-level split. The respective mean
quality labels are 2.088 and 2.016; this check does not establish balanced
patient, site or artifact distributions. A new CT-MUSIQ run may use:

```bash
python -m ct_musiq.train --split-manifest research_audit/development_split_seed42.json --output-dir results/development_seed42 --seed 42 --pretrained-file PATH
```

The pretrained file is a locally obtained ViT safetensors checkpoint; a
requested local file must load successfully. The separate released challenge
test set is not used by this development training command.

As an exploratory development baseline, `noise_baseline.py` computes the mean
squared Laplacian of each float image and fits isotonic regression using the
training partition only. On the 210-image provisional validation partition it
achieved PLCC 0.8622, SROCC 0.8446, KROCC 0.7006, aggregate 2.4074 and MAE
0.4355. This is a development-set result, not directly comparable with the
300-image test result above; direction and the isotonic mapping were selected
using development training data. See `results/noise_baseline_development/`.
The same fixed baseline, fitted on those 790 training images, was also
evaluated exploratorily on the released 300 images: aggregate 2.2879 and MAE
0.5373. The historical transformer checkpoint obtained 2.4917 and 0.4475 on
those images. Their development training partitions differ, so this is not a
controlled architecture comparison. A paired bootstrap over inferred test
slice groups gives an aggregate difference interval of 0.067–0.345 in favor
of CT-MUSIQ; it is not a patient-level confidence interval. Raw predictions
and `paired_noise_bootstrap.json` are preserved with the audit outputs.

`infer_slice_groups.py` proposes conservative groups by thresholding the
correlation of 64-by-64 image thumbnails and taking connected components. At
the default threshold, this produced 99 groups: 98 groups of 10 and one of 20.
This finding suggests related images but does not prove that every group is
one source slice, and different groups can belong to the same patient. For
example, the original 0902 test image and 0146 training image have
full-resolution pixel correlation 0.999676. The mapping in
`inferred_slice_groups.csv` must be validated against source metadata before
being described as a patient-level split.
At this threshold all 100 images in the historical 0900–0999 split share an
inferred image group with at least one image in the historical 0000–0699
training split. This is image-similarity evidence, not verified patient IDs.

The configured transformer uses eight attention heads, whereas the
`vit_base_patch32_224` checkpoint normally uses twelve. The loader copies
attention tensors by shape, so the transferred projection weights are
partitioned into a different number of heads. A fresh study should document
and test this design choice rather than assume the pretrained attention
operation is preserved exactly.
