# Development experiment protocol

## Completion status, 2026-11-25

The fixed nine-arm study is complete: 27 models, each evaluated on 300 LDCT
test images and 1,666 expert-annotated external phantom images. All six
development transformations, both quality strata and all declared controls
are retained. The final numerical summary, model/environment manifest and
verification record are in `results/`. The remote queue has finished and no
further experiment runs are planned. Earlier proposals below are retained as
development history, not an active queue. The working manuscript is in
`../paper/`; authorship details and institutional declarations require review.

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

### Ordinal and conditional degradation study, 2026-11-25

The author requests the smallest feasible significant study. Compare the
existing scalar compact model with an ordinal 21-bin expectation head and
with that same ordinal model plus a conditional synthetic-blur ranking term.
Use seeds 42,123,456, the SAME grouped partition, float 448 input, flips,
AdamW, learning rate, batch 32, epoch limit 100 and patience as the compact
baseline. New arms select checkpoints with unrounded signed correlation sum
(the older compact arm used four-decimal components; disclose this difference).
Ordinal training uses MSE + 0.5 margin rank + mean CDF squared error over the
20 nonconstant thresholds. Targets linearly interpolate MOS onto the 0.2
grid; these are not observed reader-vote distributions.

The conditional arm adds weight 0.1 times relu(0.1-original+blurred), only on
training images with original MOS>=3. Blur sigma is fixed at 1.5 at 448
pixels; auxiliary predictions use evaluation-mode BN/dropout with gradients
enabled, preserving running statistics. This is a synthetic preference
assumption on relatively high-rated slices, not a clinical ground truth.
Do not adjust this weight or threshold based on test outcomes.

Probe EVERY baseline and new-arm checkpoint on development validation with
Gaussian blur sigma .75,1.5,2.5, a box filter width 5 and additive noise
standard deviation .01,.03 in normalized units. Report score changes overall,
on original MOS>=3 and MOS<=1; transformed images have no human MOS. Two
Gaussian settings and the box family are not used in the constraint. These
are stress tests and do not establish clinical safety or denoiser validity.
Retain every seed, including failures to improve. Run the new test evaluations
only after all development training and probes complete; no test selection,
calibration, rounding or learned ensemble weights. Prior public test use makes
the entire study exploratory. Published prior art already identifies smoothness
bias and ordinal losses, so novelty must be evaluated rather than assumed.
Frozen source: `snapshots/ordinal_study_source.tar.gz`, SHA-256
`26539b6534307ec09359ff58b801bcfd9d20eb475aafd0f4744ab143835e9231`.
Three ordinal-loss/gradient checks passed in the remote experiment environment
before starting the study. External downloads remain local-only.

Follow-up declared before inspecting ANY new ordinal-arm test evaluations:
development probes show mild blur (sigma .75) raises scores in 73–100% of
the 49 original MOS>=3 images across baseline seeds, whereas sigma 1.5 is
already often penalized. The first conditional seed's auxiliary training
loss often reaches zero. Retain this original arm and run a targeted mild
sigma .75 arm at the same weight .1 and seeds 42,123,456. No other optimizer,
target threshold or ranking-margin setting changes. Gaussian 1.5 and 2.5 and
box smoothing remain nontraining stress settings for this mild arm. This
development-driven follow-up is exploratory, not a prespecified initial arm.
Source SHA-256: `d4fa992bbdb4ade32d10b09f5cfb27839e1c00abcfcc4634265588319fadc6a9`
in `snapshots/mild_ordinal_study_source.tar.gz`.

Investigate independent CBCT-IQ phantom data (Zenodo 20124496, CC BY 4.0),
downloaded locally. The actual annotations.csv has 1,666 rows, MD5
37733870d1f6cb796272b5f778a98c0c. Verify matching, input normalization and
reader-score direction before any external inference. Primary external
endpoint is Spearman correlation against human OIQ_MOS_ZSCORE, overall and
separately by phantom anatomy; do not use algorithm-derived IQA-rating columns
as human labels. No external labels may fit weights or select settings. This
phantom/domain transfer experiment cannot establish patient-level clinical
generalization. Record any missing cases rather than silently excluding them.
Secondary external endpoint, fixed before external inference: agreement with
at least two of three readers on reconstruction pairs matched on anatomy,
slice, kernel, voltage and pulse width. Exclude ambiguous/tied reader pairs
only for this secondary endpoint, reporting their count; model ties within
1e-6 receive half credit. All 1,666 images remain in the primary endpoint.
No reference-image ratings are invented for unannotated acquisitions.

External input correction: visual QA found that directly presenting the stored
NIfTI x/y array as image rows/columns rotates the anatomy relative to axial
training TIFFs. Every affine reports RAS axes. The fixed mapping
`flipud(array.T)` puts the physical x axis along columns and anterior at the
top, based on geometry and visual QA, not score optimization. Initial runs
were stopped; partial and completed outputs are preserved in
`results/external_cbct_iq_stored_array/`. Every declared model is rerun on
`dataset/external_cbct_iq/prepared_axial/`. The initial negative correlations
must not be presented as primary external results. No external setting,
normalization or model is selected by the corrected test outcomes.

Final external inference is moved to the two original GPUs for speed and
framework consistency, using the SAME fixed axial preprocessing and all 12
models. Uploads originate from the local computer. CPU runs are retained as
cross-environment checks; GPU results form the final complete external table.
This change is declared before viewing GPU external outcomes. Source archive
`snapshots/external_gpu_source.tar.gz` SHA-256 is
`0902005727533ddbfb0b8c5c071ae2a189fbb34410e5e573e9a035a28740f814`.
The published CBCT-IQ reader rubric was checked: 1 means very poor and 4 very
good (paper page 9); the human z-score direction is retained. Readers saw a
reference image while assigning these scores, whereas models receive only the
test image. This annotation-task difference is part of the transfer limitation.

### Scalar-head component control

To test whether the ordinal head is necessary for the constraint's effect,
add a scalar head plus the same mild constraint (.75 sigma, .1 weight, .1
margin, original MOS>=3) with seeds 42,123,456 and all other settings fixed.
This completes the head/constraint component comparison without another
backbone search. It is an exploratory mechanistic follow-up after observing
earlier benchmark and initial external baselines, not a new untouched-holdout
claim. No external labels fit its parameters or choose its hyperparameters.
Source snapshot `snapshots/scalar_constraint_source.tar.gz`, SHA-256
`3f361a1c582fd0595cc9adf7c2e2c02d558015af38b9936f755536de43e53bd9`.
The original scalar baseline's rounded selection differs slightly from the
new unrounded selection and is disclosed for this control as well.

### Second-backbone confirmation

As a final portability check of the same constraint, compare scalar ResNet-18
with and without mild conditioning at 448, seeds 42,123,456. Keep all data,
loss weights, threshold, margin, sigma and optimizer settings fixed. Only the
backbone changes to ImageNet `timm/resnet18.a1_in1k`; no head/loss search is
performed. This tests a training constraint across two standard CNN families,
not a new architecture. It is added during the exploratory study after initial
external outcomes are known; external labels never fit or choose its settings.
Retain every seed and report the same benchmark/probe/external endpoints.
Frozen source `snapshots/resnet_constraint_source.tar.gz`, SHA-256
`493b3eb168a964ad1d4062f97fad07268348d27b202761d9cb1cc1a60811ded3`.
Offline ResNet-18 weights SHA-256
`80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`.

### Unconditional-ranking ablation

The final mechanistic ablation applies the SAME scalar EfficientNet mild-blur
ranking term to all training images (threshold 0 instead of 3), seeds
42,123,456. This tests whether quality conditioning contributes beyond an
ordinary degradation-ranking prior. It is required to assess the central
interpretation, not a hyperparameter search for a higher test score. All
other settings, including auxiliary BN/dropout handling, are fixed. Retain
the outcome even if it contradicts the conditioning hypothesis. Synthetic
distortion-ranking prior art includes RankIQA (ICCV 2017) and distortion-guided
opinion-unaware CT IQA (ISBI 2026); cite both and do not claim that degradation
ranking itself is new. This is the last planned method component; complete
the statistics and manuscript after these fixed controls.
Frozen source `snapshots/unconditional_source.tar.gz`, SHA-256
`68203ebef5e59898ac121dc84d6cd9fa454d51201acfeeac86fb0ce4f64945ee`.

### Fixed ensemble follow-up on 2026-11-25

Following the author's request for further improvements, first evaluate the
equal arithmetic mean of the three existing FLOAT compact models (seeds 42,
123,456) on development validation. No weights are fitted and no subsets,
rounding or calibration rules are searched. Validation aggregate is 2.841767
and MAE 0.173226, improving on all three individual validation checkpoints.
Freeze this same three-model average for subsequent released-test evaluation.
This is exploratory follow-up after prior test use; the single-model
three-seed result remains the original primary estimate. The ensemble is one
predictor, not three independent ensemble replications. Report three forward
passes and approximately three times the parameter storage as its costs.
Use `ensemble_predictions.py` and retain prediction-source hashes.

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

## Final matched strong-blur control (exploratory)

After the original external CPU outcomes were available, retain the original
ordinal Gaussian sigma 1.5 arm and compare it with the identical constraint
applied to every training image (threshold 0 instead of 3). Use seeds 42, 123,
456, weight 0.1, margin 0.1 and unchanged optimization and checkpoint selection.
This isolates the gate for the initial strong-blur arm; it is an adaptive
follow-up, not an untouched confirmatory comparison. No external labels fit
weights, calibrations or hyperparameters. Report all nine arms (27 models),
including unsuccessful synthetic and transfer outcomes, without selecting a
new best configuration from external results.

Frozen source `snapshots/strong_unconditional_source.tar.gz`, SHA-256
`aa037982b80388c4ef124ad47d435be6b37c704bdcbbd9672b845a03aa26d807`.
The runner waits for current training and the original 12 external GPU runs
before allocating additional training memory, and then evaluates all 15
follow-up models externally with the same fixed axial inputs and GPU framework.

## Publication constraint

The author prefers no APC at all and can cover at most US$120 in total.
Candidate hybrid journals must be checked for a subscription option and any
separate mandatory charges at submission; do not select an author-paid
open-access option that exceeds this limit. Working
candidates are Computers in Biology and Medicine, Computerized Medical Imaging
and Graphics, and the higher-reach Medical Image Analysis. Journal choice
depends on the completed evidence and the supervisor's agreement.
