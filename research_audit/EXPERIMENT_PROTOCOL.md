# Development experiment protocol

## Current priority: limited stronger-method screening

The author prioritizes a stronger method with limited implementation work,
using recent literature rather than treating the 2023 leaderboard as current
state of the art. The queued broad component sweep was cancelled before it
started. Existing seed replications may finish and remain pilot evidence.

Screen two small development alternatives, initially seed 42 only: (1) the
multi-scale transformer with twelve-head ViT transfer, transferred CLS and
final norm, pretrained input normalization, float resizing and flip-only
augmentation; (2) an ImageNet-pretrained EfficientNet-B0 at 448 pixels with a
small bounded regression head, flip-only augmentation, MSE and ranking loss.
Use the same grouped development partition and select follow-up work using
validation evidence. These changes are baseline engineering, not a novelty
claim. Do not promise journal acceptance or a fast review from a higher score.

The screening source snapshot is `snapshots/development_screening_source.tar.gz`
(SHA-256 `0dbf09ac9e6fe934c85e0329cbd205eea0323546a380064c78d27b45e686e6c0`).
EfficientNet-B0 pretrained weights are from `timm/efficientnet_b0.ra_in1k`
(SHA-256 `d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd`).
All external downloads are performed locally and copied to the server.

Recent comparison context checked on 2026-11-25 includes the published TFKT V2
study (https://pmc.ncbi.nlm.nih.gov/articles/PMC12116730/), the MedSigLIP
preprint and author code (https://arxiv.org/abs/2511.12256 and
https://github.com/itu-biai/medsiglip-ldct-iqa), and ClinReadNet, published in
Neural Networks 197 (2026), 108535
(https://doi.org/10.1016/j.neunet.2025.108535; author manuscript at
https://arxiv.org/abs/2606.10372). Reported numbers from these works
are not assumed to be comparable until training, ensembling, score rounding,
splits and model-selection procedures are checked. They are context for
screening, not proof that this project is competitive or novel.

## Question and scope

### Compact follow-up frozen on 2026-11-25

The seed-42 compact run completed after 51 epochs, selecting epoch 31 using
the signed development correlation sum (aggregate 2.8342, MAE 0.1807).
Before inspecting this candidate's released-test predictions, repeat the
unchanged 448-pixel protocol with seeds 123 and 456 on the same partition.
Report all three seeds and their mean and sample standard deviation; do not
choose a seed, ensemble weight, calibration, rounding rule or hyperparameter
from test performance. Evaluate the aligned-transformer screen as well and
retain its result regardless of which model is stronger. No new component
claim is supported by this comparison because several settings differ.

This is a limited baseline and transfer-learning study. EfficientNet-B0 at
448 pixels is already a published author-repository comparator, so using it
is not itself a novel method. A useful efficiency or methodological finding
must be demonstrated before drafting a strong-journal novelty claim. The
current MedSigLIP author repository also now links released checkpoints;
earlier notes about unavailable checkpoints should not be treated as current.
Single-split, inferred-group validation remains a limitation.

The follow-up launcher is `run_screening_followup.py`. It runs one additional
compact seed on each GPU after any existing screen finishes, uses the frozen
screening source, and records process exit codes and wall times. It downloads
nothing. The released test is already exploratory for this project.

### One controlled precision comparison

After completing the compact three-seed screen, perform one additional
three-seed arm: seeds 42, 123 and 456 with the same compact model, partition,
augmentation, optimizer, batch size, maximum epochs and selection criterion.
The sole input change is clipping to [0,1], flooring to 256 intensity levels
by `uint8(255*x)/255`, then using the SAME floating-point bicubic resize as
the main arm. This isolates quantization; it does not reproduce the complete
legacy integer-resize pipeline. Inference uses each checkpoint's stored flag.
Report every seed and both validation and released-test results regardless of
direction, without tuning the quantization rule on test scores. This is
exploratory follow-up after prior test use, not preregistered research.
The hypothesis was motivated by the earlier preprocessing audit; a benefit
is not assumed. Do not expand this into another architecture search.
The source snapshot is `snapshots/precision_comparison_source.tar.gz`,
SHA-256 `3e320878df458159cb496beeedff2f7a88ff6c79c6914b208ad864708c6e5dae`.
`run_precision_comparison.py` checks that hash and verifies the quantization
range and intensity-level count on a development training image before
launching the three runs on two GPUs. The main float runs keep their original
frozen source; old compact checkpoints default to float input at evaluation.

Both screens, the compact seed replications and all three precision-control
runs have now completed. Every declared evaluation succeeded. Their outcomes
are retained in README.md, including the small aggregate difference and lower
MAE of the quantized control. This does not establish a new method or a
precision benefit. No further architecture or component sweep is queued.

Evaluate CT-MUSIQ for predicting radiologist image-quality scores on the
LDCTIQAC2023 dataset, and compare it with a simple high-frequency baseline.
This is retrospective algorithm research; no claim of clinical readiness is
supported by these experiments.

## Data and separation

- The 1,000 released training images are development data. The separate
  released 300-image challenge test collection is used only for evaluation.
- `development_split_seed42.json` assigns 790 images to training and 210 to
  validation, disjoint by image-similarity groups. These groups are inferred
  from the TIFFs, not known patient IDs; some patients may cross the split.
- The public test set has already been used for exploratory evaluation of the
  archived checkpoint and scalar baseline. It is therefore not an untouched
  holdout for this project. Report this history and avoid selecting settings
  using its observed scores.

## Initial development run

Train the two-scale CT-MUSIQ model from locally supplied ImageNet-pretrained
ViT-B/32 weights, using seed 42, KL weight 0.10 and at most 100 epochs with
the repository's existing early-stopping and checkpoint-selection logic. Run
batch sizes 32 and 64 concurrently on separate GPUs, using four data-loading
workers each. The split and seed are held fixed; batch size changes the number
of optimizer updates and is therefore a substantive experimental factor.
These two runs do not establish stability across random seeds.

After the initial batch-64 run stopped at epoch 21 with its best validation
score at epoch 1, an additional exploratory batch-64 run was started on GPU 1.
It uses EMA decay 0.993617254677532 (approximately the original per-example
decay), a 40-epoch minimum, and signed correlation-sum checkpoint selection.
These are changes to multiple settings, so the adjusted run is not a clean
isolated batch-size comparison. Its code snapshot is
`snapshots/grouped_batch64_adjusted_source.tar.gz` (SHA-256
`ca0f5b0949061ebc2dfdee39fc87da19d43e8a95a906e9ef0fac650b8b01053a`).

The copied run source snapshot is `snapshots/grouped_parallel_source.tar.gz`
(SHA-256 `a4e3ffbf3c767abbbc47648ca235a4cea675a3a13e58efe68d5c5232a94b4fc4`).
The input training archive SHA-256 is
`44ea6b1acbd285e64a8033026863be08f7847138f8ac48b1fd3b125dd8900089`;
the pretrained safetensors SHA-256 is
`b0a4866d4f722434e9d26c529181f1e7072d80edcf75843f2af4e091617a4098`.
The remote runner checks both input hashes before starting.

The code uses 8-bit conversion before resizing and an eight-head attention
architecture initialized from a twelve-head ViT. Both are methodological
choices to report. The training code uses augmentation, ranking loss, and
scale-consistency loss; no official test images are used during fitting.

For a controlled study of split leakage, a later experiment should compare
image-level and inferred-slice-group development splits with matched partition
sizes, identical training settings, and repeated seeds. Assess both validation
and independent test performance. The existing historical checkpoint and the
new grouped run have different partitions and cannot isolate the cause of any
performance difference. Patient-level separation remains unverified.

## Cancelled component-run proposal

This proposal was cancelled before any of its queued runs started. It is
retained as experimental history, not completed evidence or a pending queue.

The component study uses the batch-32 setup, 100 epochs maximum, EMA 0.999,
the original early-stopping rule, and the same grouped development split.
For each seed 42, 123 and 456, compare two scales without KL, one scale
without KL, and two scales without ranking loss against the full two-scale
model. The three-scale configuration is initially a single-seed exploratory
run (42), not part of a three-seed conclusion. Evaluate all completed arms;
do not select an ablation result based on test performance.

The model's scale pooling and positional-embedding capacity were corrected
so the scale list is explicit. Single-, two- and three-scale gradient tests
pass; a two-scale historical checkpoint gives byte-identical predictions
before and after this change. The frozen source and queue are in
`snapshots/ablation_study_source.tar.gz` (SHA-256
`f40adc2fe86da37d4d5413b8c65a70e1b80482b76921e660a3c991f1892a12b2`).

## Baseline and evaluation

The scalar baseline is mean squared Laplacian response, with isotonic mapping
and direction fitted only on development training labels. Compute signed
Pearson, Spearman and Kendall correlations, the challenge's sum of their
absolute values, MAE, and per-image errors. Prefer paired comparisons over
inferred source-slice groups when available; this does not establish
patient-level statistical independence. The published challenge leaderboard
may be presented as context, with differences in training budgets,
preprocessing and evaluation history disclosed.

## Before a publication claim

Obtain or verify patient/source metadata where possible, conduct multiple
independently seeded runs and controlled component ablations, compare with
appropriate published methods, and investigate errors and calibration.
External validation on additional patients, acquisition conditions and sites
would materially strengthen generalization claims. Keep all exploratory
choices and test use transparent. Do not describe the project's Swin-T +
ResNet50 model as an exact implementation of the original challenge winner.

## Publication constraint

The author prefers no APC at all and can cover at most US$120 in total.
Candidate hybrid journals must be checked for a subscription option and any
separate mandatory charges at submission; do not select an author-paid
open-access option that exceeds this limit. Working
candidates are Computers in Biology and Medicine, Computerized Medical Imaging
and Graphics, and the higher-reach Medical Image Analysis. Journal choice
depends on the completed evidence and the supervisor's agreement.
