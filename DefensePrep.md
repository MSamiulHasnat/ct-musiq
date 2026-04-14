# Defense Preparation Guide: CT-MUSIQ Thesis

This guide is a practical, defense-focused walkthrough of your full project so you can explain every mechanism clearly and confidently.

## 1. Defense Goal

Your target in the viva is not to recite code, but to show:
- You understand the problem and clinical motivation.
- You can justify each architectural and training choice.
- You can explain results honestly, including strengths and weaknesses.
- You can defend reproducibility and fairness.

## 2. One-Minute Thesis Pitch

Use this as your opening:

"My thesis proposes CT-MUSIQ, a multi-scale transformer adaptation for no-reference quality assessment of low-dose brain CT images. The model predicts radiologist-like quality scores from a single image without a reference scan. The pipeline uses multi-scale patch tokens, hash-based coordinate positional encoding, and a composite objective combining regression, ranking, and scale-consistency learning. I evaluate with PLCC, SROCC, KROCC, and aggregate score under a deterministic train/val/test split and compare against a strong ensemble baseline for fairness."

## 3. End-to-End Pipeline (What Happens to One Image)

1. Load one grayscale CT slice and its quality score.
2. Build a multi-scale pyramid (default: 224 and 384).
3. Split each scale into non-overlapping 32x32 patches.
4. Concatenate all patches into one token sequence.
5. Build coordinate tuples per patch: (scale_idx, row_idx, col_idx).
6. Replicate grayscale to 3 channels for pretrained compatibility.
7. Feed tokens + coordinates to CT-MUSIQ.
8. Produce:
   - Global quality score.
   - Per-scale quality scores.
9. Compute total loss.
10. Update parameters through staged training.
11. Evaluate on validation and test/blind split with correlation metrics.

## 4. Data and Splits You Must Memorize

- Total images: 1000
- Train: 700 (0000-0699)
- Val: 200 (0700-0899)
- Test/Blind: 100 (0900-0999)
- Label file format: keys like 0042.tif
- Score range: 0 to 4

Why this matters:
- Deterministic split improves reproducibility.
- Prevents accidental leakage from random re-sampling.

## 5. CT-MUSIQ Architecture Explained

### 5.1 Multi-scale tokenization

Default scales are [224, 384], with patch size 32:
- 224 -> 7x7 = 49 patches
- 384 -> 12x12 = 144 patches
- Total tokens = 193

Motivation:
- Lower scale captures global structure.
- Higher scale captures fine-grained noise and detail.

### 5.2 Patch embedding

A Conv2d projection maps each 32x32 patch into embedding space.

Motivation:
- Efficient and compatible with ViT-style patch embeddings.

### 5.3 Hash-based positional encoding

Each token receives:

PE = E_scale(s) + E_row(r) + E_col(c)

Motivation:
- Works naturally with variable multi-scale token sequences.
- Encodes both spatial and scale identity.

### 5.4 Transformer encoder + heads

- Transformer learns interactions among all tokens.
- [CLS] branch predicts global quality.
- Per-scale heads produce scale-specific quality scores.

Motivation:
- Global prediction is final output.
- Per-scale heads enable consistency regularization.

## 6. Loss Function (Core Mathematical Defense)

Total objective:

L_total = L_mse + lambda_kl * L_kl + lambda_rank * L_rank

### 6.1 MSE loss

- Optimizes absolute score error.
- Useful for direct regression to radiologist score.

### 6.2 KL scale-consistency loss

- Converts scalar scores to soft distributions over bins.
- Penalizes disagreement between per-scale predictions and global prediction.

Why this is defensible:
- Different scales observe same anatomy; their quality assessment should be coherent.
- Reduces scale-specific shortcut behavior.

### 6.3 Pairwise ranking loss

- Optimizes ordering between image pairs.

Why this matters:
- IQA metrics include rank correlation (SROCC, KROCC), so ranking supervision is aligned with evaluation.

## 7. Training Strategy and Why It Is Reasonable

### 7.1 Two-stage training

Stage 1:
- Freeze transformer encoder.
- Train CT-specific front/back layers first.

Stage 2:
- Unfreeze all parameters.
- Warmup and cosine scheduling for stable full fine-tuning.

Why this is good:
- Prevents unstable early gradients from damaging pretrained representations.

### 7.2 Stability tools

- Mixed precision on CUDA.
- Gradient clipping.
- Early stopping.
- EMA of weights (for CT-MUSIQ).

## 8. Evaluation Protocol

Metrics:
- PLCC
- SROCC
- KROCC
- Aggregate = |PLCC| + |SROCC| + |KROCC|

Options:
- TTA (horizontal flip averaging).
- Isotonic calibration on validation predictions.

Output artifacts:
- Per-image prediction CSV.
- Summary result CSV.

## 9. Baseline and Fairness Story

Baseline model:
- agaldran_combo = average of Swin-T and ResNet50 scores.

Fairness principles:
- Same dataset splits.
- Same evaluation metrics.
- Same reporting format.

Defense framing:
- Baseline is intentionally strong, so comparison is meaningful.

## 10. Ablation Logic (Most Important for Examiners)

Ablations test causal contribution of design choices:
- A1: single-scale, no KL
- A2: two-scale, no KL
- A3/A4/A5: KL weight sensitivity
- A6: third scale stress test (VRAM tradeoff)

How to narrate results:
1. Show multi-scale benefit: A1 -> A2.
2. Show consistency-loss benefit: A2 -> A4.
3. Show hyperparameter sensitivity: A3/A4/A5.
4. Show practical hardware limit: A6 if OOM.

## 11. High-Value Questions You Must Be Ready For

1. Why no-reference IQA for CT?
- In deployment, a pristine reference image is unavailable.

2. Why multi-scale?
- Quality artifacts and anatomy cues exist at different frequencies and extents.

3. Why not only MSE?
- MSE optimizes absolute error, but ranking metrics demand ordering quality too.

4. Why KL consistency?
- Prevents contradictory predictions across scales and regularizes learning.

5. Why this split strategy?
- Deterministic and reproducible, avoids leakage and unstable random splits.

6. Why baseline can outperform in some runs?
- Ensemble backbones are very strong; thesis value is principled adaptation plus ablation-backed insight, not just one metric peak.

## 12. Weaknesses and Honest Limitations (Say These Proactively)

- Dataset size is limited for deep models.
- Evaluation is currently from one fixed split.
- External-domain generalization not fully validated.
- Calibration and TTA may improve metric stability but add inference complexity.

Then add improvement plan:
- Cross-hospital validation.
- Multi-reader reliability analysis.
- Lightweight deployment version.

## 13. 7-Day Defense Drill Plan

Day 1:
- Master data pipeline and split logic.

Day 2:
- Master architecture block-by-block with diagrams.

Day 3:
- Master loss math and intuition.

Day 4:
- Master training schedule, optimization, and stability choices.

Day 5:
- Master evaluation metrics and result interpretation.

Day 6:
- Master ablation narrative and likely examiner criticisms.

Day 7:
- Full mock viva with timed answers.

## 14. 30-Second Answers Template (Use in Viva)

Use this structure for almost any question:
1. "The goal is ..."
2. "We chose X because ..."
3. "This affects performance by ..."
4. "Evidence from ablation/evaluation shows ..."
5. "A limitation is ... and next step is ..."

## 15. Numbers to Keep in Memory

- 1000 images total.
- 700/200/100 split.
- Scales 224 and 384.
- Patch size 32.
- 193 tokens per image.
- Score range 0-4.
- Aggregate metric formula.
- Your latest test metrics for both ct_musiq and agaldran_combo.

## 16. Final Defense Mindset

- Be precise and concise.
- Never bluff: state what is known, unknown, and future work.
- Defend design choices with mechanism plus evidence.
- Keep linking implementation decisions to clinical and statistical rationale.

If you can explain each section in this file clearly without notes, your defense quality will be in a very strong position.

## 17. Rehearsal Scripts (Same File)

### 17.1 One-Minute Script (Quick Opening)

"My thesis addresses no-reference quality assessment for low-dose brain CT, where no clean reference image is available in clinical workflow. I propose CT-MUSIQ, a multi-scale transformer adaptation that processes two image scales, tokenizes each into patches, and uses hash-based coordinate positional encoding to preserve scale and spatial identity. The model predicts a global quality score and per-scale scores. Training uses a composite objective combining MSE for absolute score regression, KL consistency between scale and global predictions, and ranking loss for pairwise ordering. Evaluation uses PLCC, SROCC, KROCC, and aggregate score on a deterministic 700/200/100 train/val/test split. I also compare against a strong Swin-T plus ResNet50 ensemble baseline and run ablations to isolate the value of multi-scale learning and consistency regularization." 

### 17.2 Three-Minute Script (Method + Evidence)

"The core problem is blind IQA for brain CT, where model predictions should correlate with radiologist quality scores from 0 to 4. In this project, each image is resized into two scales, 224 and 384, and split into 32x32 patches, resulting in 193 tokens per image. Every token receives a coordinate tuple (scale, row, column), and positional encoding is formed by summing learned embeddings for each component. This lets the transformer reason jointly over local and global structures while remaining scale-aware.

The architecture produces one global score and per-scale scores. The loss is designed to match both regression and ranking goals: MSE learns numeric accuracy, ranking loss helps ordering, and KL consistency discourages contradictory scale-specific behavior. Training is two-stage: first freeze encoder to stabilize adaptation layers, then unfreeze all for full fine-tuning with warmup and cosine scheduling. Additional stabilization includes gradient clipping, early stopping, mixed precision on GPU, and EMA.

For evaluation, I use PLCC, SROCC, KROCC, and aggregate score, plus optional TTA and isotonic calibration. I report results with saved prediction CSVs and comparison tables. Importantly, I run controlled ablations: single-scale vs multi-scale, with and without KL, and KL-weight sensitivity. This provides causal evidence that the method design is justified rather than accidental." 

### 17.3 Ten-Minute Script (Full Defense Narrative)

Use this 10-minute structure in slides:

1. Clinical Motivation (1 min)
- Low-dose CT reduces radiation but may degrade perceptual quality.
- Manual quality scoring is expensive and subjective.
- Blind IQA is practical because reference images are unavailable in deployment.

2. Problem Definition (1 min)
- Input: single brain CT slice.
- Output: quality score in [0, 4].
- Objective: maximize correlation with radiologist labels.

3. Dataset and Splits (1 min)
- 1000 images total.
- Deterministic split: 700 train, 200 val, 100 test.
- Why deterministic: reproducibility and fair comparison.

4. Data Pipeline (1 min)
- Multi-scale resizing: 224 and 384.
- Patch extraction with size 32.
- Token count: 49 + 144 = 193.
- Coordinate tracking per patch: (scale, row, col).

5. Model Architecture (2 min)
- Patch embedding to transformer dimension.
- Hash positional encoding:
   PE = E_scale(s) + E_row(r) + E_col(c).
- Transformer encoder captures global dependencies across all tokens.
- Two prediction outputs: global head and per-scale heads.

6. Loss Design (1.5 min)
- Total objective combines MSE, KL consistency, and ranking.
- MSE: numeric regression.
- KL: enforces coherence across scales.
- Ranking: aligns optimization with rank-based metrics.

7. Training Strategy (1 min)
- Stage 1 freeze for stable adaptation.
- Stage 2 unfreeze with warmup + cosine schedule.
- Stability tools: clipping, early stopping, mixed precision, EMA.

8. Evaluation + Baseline + Ablation (1.5 min)
- Metrics: PLCC, SROCC, KROCC, aggregate.
- Strong baseline: Swin-T + ResNet50 score ensemble.
- Ablation evidence: multi-scale effect, KL effect, weight sensitivity.
- Discuss both wins and limitations transparently.

## 18. Rapid-Fire Viva Questions (With Model Answers)

1. What is your core contribution?
- A scale-aware transformer IQA pipeline for brain CT with hash coordinate encoding and a composite loss that jointly targets regression, ranking, and scale consistency.

2. Why use two scales only?
- Two scales capture complementary detail while remaining computationally feasible on limited VRAM; this balance is supported by ablation and hardware constraints.

3. Why not use only one scale at 384?
- Single-scale can miss either global structure or fine detail depending on choice; two-scale fusion improves robustness to different artifact patterns.

4. Why replicate grayscale to RGB?
- To leverage pretrained ImageNet backbones and patch embeddings expecting 3 channels, while preserving identical intensity information across channels.

5. Why is KL computed on distributions, not raw scores?
- KL divergence requires valid probability distributions. Scores are mapped to soft bins to compare uncertainty-aware consistency across heads.

6. Why ranking loss if you already have MSE?
- MSE optimizes point error; ranking loss directly improves pairwise ordering, which is better aligned with SROCC and KROCC.

7. How do you prevent overfitting?
- Two-stage fine-tuning, dropout, early stopping, deterministic splits, strong validation monitoring, and ablation-backed hyperparameter control.

8. Why deterministic index split instead of random k-fold?
- It ensures reproducibility and stable benchmarking. In a thesis benchmark setting, deterministic protocol improves comparability across experiments.

9. Why compare with agaldran_combo?
- It is a strong ensemble-style baseline with modern backbones, making comparison more credible than weak baselines.

10. What is your biggest limitation?
- Limited dataset size and single-domain evaluation. Cross-domain or multi-center validation is the highest-priority next step.

## 19. Final Week Checklist (Execution-Focused)

1. Rehearse 1-minute, 3-minute, and 10-minute scripts daily.
2. Memorize key numbers: split, scales, patch count, metrics, and top results.
3. Prepare one slide each for architecture, loss, and ablation.
4. Prepare one slide titled "Limitations and Future Work".
5. Do 2 mock vivas with interruption-style questioning.
6. Practice concise answers: max 30-40 seconds per question unless asked for detail.

## 20. Technical Deep Dive: Exact Tensor Flow

This section explains the exact tensor contracts so you can defend implementation details without ambiguity.

### 20.1 Dataloader output contract

For each batch from the dataset pipeline:
- patches: [B, N, 3, 32, 32]
- coords: [B, N, 3]
- score: [B]
- image_id: list of length B

Where N is total patches across scales.
For default scales [224, 384], N = 49 + 144 = 193.

### 20.2 Forward pass tensor shapes in CT-MUSIQ

1. Flatten patch axis:
- [B, N, 3, 32, 32] -> [B*N, 3, 32, 32]

2. Patch embedding (Conv2d with kernel=stride=32):
- [B*N, 3, 32, 32] -> [B*N, 768, 1, 1] -> flatten -> [B*N, 768]

3. Restore token axis:
- [B*N, 768] -> [B, N, 768]

4. Positional encoding from coordinates:
- coords [B, N, 3] -> pos [B, N, 768]
- token embeddings + pos encoding -> [B, N, 768]

5. Prepend cls token:
- cls parameter [1, 1, 768] expanded to [B, 1, 768]
- concat -> [B, N+1, 768]

6. Transformer encoder stack:
- input [B, N+1, 768] -> output [B, N+1, 768]

7. Heads:
- global head from token index 0 -> [B, 1]
- per-scale heads from pooled scale token slices -> list of [B, 1]

### 20.3 Scale slicing logic

For each scale s with image size S_s and patch size P:
- grid size g_s = S_s / P
- token count n_s = g_s^2

Default:
- S_1 = 224, g_1 = 7, n_1 = 49
- S_2 = 384, g_2 = 12, n_2 = 144

Scale token windows are contiguous in sequence by construction, so each scale head receives the correct token subset.

## 21. Mathematical Details You Should Be Able to Derive

### 21.1 Score-to-distribution transform for KL term

Given score q and bin centers b_k:

p_tilde_k = exp( - (q - b_k)^2 / (2*sigma^2) )

p_k = p_tilde_k / (sum_j p_tilde_j + eps)

This makes p a valid probability simplex vector.

### 21.2 KL consistency term

For each scale prediction q_s and global prediction q_g:

L_kl,s = KL( p(q_s) || p(q_g) )

Overall:

L_kl = (1 / num_scales) * sum_s L_kl,s

Interpretation:
- If scale-specific prediction drifts from global prediction, KL rises.
- This discourages contradictory scale behavior.

### 21.3 Pairwise ranking term

For pair (i, j), with target sign y_ij = sign(t_i - t_j):

L_rank,ij = max(0, m - y_ij * (s_i - s_j))

Batch objective averages this over all pairs.

Interpretation:
- If predicted order matches true order with margin m, loss is 0.
- Otherwise, loss pushes relative ordering to correct direction.

### 21.4 Total training objective

L_total = L_mse + lambda_kl * L_kl + lambda_rank * L_rank

Gradient signal behavior:
- L_mse controls absolute calibration.
- L_rank controls monotonic ordering.
- L_kl controls cross-scale coherence.

## 22. Optimization Mechanics Under the Hood

### 22.1 Why AdamW

AdamW separates weight decay from gradient-based adaptive moments, which is generally more stable for transformer fine-tuning than classic Adam with l2 coupling.

### 22.2 Two-stage schedule as an optimization constraint

Stage 1 freezing reduces parameter search space:
- only adaptation-sensitive layers move early
- pretrained self-attention statistics are protected

Stage 2 unfreezing expands optimization space after heads stabilize.

### 22.3 Warmup and cosine interaction

- Warmup avoids abrupt learning-rate jump at stage transition.
- Cosine annealing then gradually reduces step size to improve convergence smoothness.

### 22.4 Gradient clipping role

Clip by global norm prevents rare large updates from destabilizing attention blocks, especially with mixed-precision scaling dynamics.

### 22.5 EMA role

EMA forms a low-pass filter on weight trajectory:
- theta_ema <- decay * theta_ema + (1 - decay) * theta

This reduces validation variance and often improves checkpoint selection reliability.

## 23. Pretraining Transfer: What Is Actually Loaded

Transferred from ViT-B/32 when shapes match:
- patch embedding projection weights and bias
- attention qkv, projection, layer norms, and mlp weights in mapped encoder blocks

Important technical point:
- Shape-checked partial loading is safer than strict loading because architecture differs in heads and positional mechanism.

Defense line:
- Transfer keeps generic visual priors while preserving freedom for task-specific adaptation.

## 24. Complexity and Memory Budget Discussion

### 24.1 Attention complexity

Self-attention complexity per layer is O(T^2 * D), where:
- T = N + 1 tokens
- D = embedding dimension

With default N = 193, T = 194.
If extra scales are added, T increases quickly, and compute and memory rise quadratically with token count.

### 24.2 Why A6 can hit memory limits

Adding a 512 scale contributes many extra tokens.
Because attention uses quadratic token interactions, this can exceed 6GB VRAM even if model parameters are unchanged.

## 25. Evaluation Statistics: What Each Metric Really Tests

### 25.1 PLCC

Measures linear agreement between predictions and targets.
Sensitive to scaling and bias.

### 25.2 SROCC

Measures monotonic rank consistency.
Insensitive to affine scaling of predictions.

### 25.3 KROCC

Measures pairwise concordance probability.
More strict for local ranking consistency.

### 25.4 Aggregate metric rationale

Aggregate = |PLCC| + |SROCC| + |KROCC|

This combines absolute regression behavior and ranking behavior into one summary.

## 26. Failure Modes and Diagnostics (Technical)

### 26.1 Over-smoothing predictions

Symptom:
- high SROCC, lower PLCC

Cause hypothesis:
- model ranks correctly but is under-calibrated in scale

Action:
- try calibration, inspect prediction histogram and residuals

### 26.2 Rank errors on hard pairs

Symptom:
- weak KROCC relative to PLCC

Cause hypothesis:
- pairwise margin not strong enough or noisy near-tie labels

Action:
- tune ranking weight or margin, inspect pairwise disagreement regions

### 26.3 Scale inconsistency

Symptom:
- unstable validation, divergent per-scale behavior

Cause hypothesis:
- lambda_kl too low or poor scale pooling quality

Action:
- inspect per-scale score correlation with global score

## 27. Code-Level Defense Map (Where to Point Examiners)

Use these source files in oral defense when asked "where is this implemented?"

1. Dataset and patch pipeline:
- [dataset.py](dataset.py)

2. Core architecture and pretrained mapping:
- [model.py](model.py)

3. Composite loss and ranking/kl mechanics:
- [loss.py](loss.py)

4. Training stages, optimizer, scheduler, EMA:
- [train.py](train.py)

5. Evaluation metrics, TTA, calibration, output CSVs:
- [evaluate.py](evaluate.py)

6. Baseline ensemble implementation:
- [baseline_models.py](baseline_models.py)

7. Ablation orchestration:
- [ablation.py](ablation.py)

## 28. 15 Ultra-Technical Questions for Mock Viva

1. Why norm_first in transformer encoder, and what stability effect does it have?
2. Why are scale tokens concatenated contiguously, and what breaks if ordering changes?
3. What assumptions does score-to-distribution KL impose on score topology?
4. Why use isotonic regression for calibration instead of linear scaling?
5. Which metric is most sensitive to monotonic but non-linear prediction mappings?
6. How does ranking loss interact with MSE when target differences are near zero?
7. Why does attention complexity grow quadratically with token count?
8. How does freezing change optimization landscape dimensionality?
9. Why can EMA improve checkpoint selection even if training loss is unchanged?
10. What are risks of loading mismatched pretrained heads with strict state loading?
11. Why can high PLCC coexist with weaker KROCC?
12. How would you test whether KL consistency is genuinely causal and not correlated?
13. What would you modify first for cross-domain generalization?
14. How would you estimate uncertainty of predicted quality in this framework?
15. If labels are noisy, which loss component is most fragile and why?

If you can answer these 15 questions clearly, your technical defense will stand out as genuinely expert-level rather than script-level.

## 29. Technical Terms Glossary (Definition Section)

This section defines the key technical terms used across your thesis and code.

### 29.1 Data and Imaging Terms

- CT (Computed Tomography): A medical imaging modality that reconstructs cross-sectional anatomy from X-ray projections.
- Low-dose CT: CT acquired with reduced radiation dose, usually causing higher noise and possible quality degradation.
- Brain CT slice: A 2D axial image from a 3D CT volume.
- No-reference IQA (Blind IQA): Quality assessment without any pristine reference image.
- Quality score (0-4): Continuous target label representing perceived diagnostic quality.
- Label noise: Inconsistency in human annotations due to subjective judgment or inter-reader variation.
- Data leakage: Unintended information transfer from validation/test into training, causing over-optimistic results.
- Deterministic split: Fixed train/val/test partition that does not change across runs.
- Domain shift: Distribution change between training data and deployment data (scanner, protocol, hospital, population differences).

### 29.2 Representation and Tensor Terms

- Tensor: Multi-dimensional numeric array used by deep learning frameworks.
- Shape: Dimensional structure of a tensor, e.g., [B, N, C, H, W].
- Batch size (B): Number of samples processed in one forward/backward pass.
- Channel (C): Feature planes in an image tensor (for RGB, C=3).
- Token: A vector representation of one patch fed to the transformer.
- Embedding dimension (D or d_model): Length of each token feature vector (768 in this project).
- Sequence length (T): Number of tokens processed by attention (patch tokens plus cls token).
- Flatten: Reshape operation that collapses dimensions for efficient matrix operations.
- View/reshape: Reinterpret tensor dimensions without changing values.

### 29.3 Multi-Scale and Patch Terms

- Multi-scale pyramid: Multiple resized versions of the same image used simultaneously.
- Scale: One target resolution in the pyramid (e.g., 224 or 384).
- Patch: Non-overlapping local crop (32x32 here) converted to a token.
- Patch size: Side length of each square patch.
- Patch grid: 2D index layout of patches over one scale.
- Scale index: Integer ID indicating which scale a patch came from.
- Coordinate tuple (scale, row, col): Token location descriptor used by positional encoding.
- Token concatenation: Joining patch tokens from all scales into one sequence.

### 29.4 Transformer and Architecture Terms

- Transformer encoder: Stack of attention + feed-forward blocks for sequence modeling.
- Self-attention: Mechanism where each token attends to every other token in the sequence.
- Multi-head attention: Parallel attention subspaces that capture different relationships.
- Query/Key/Value (Q/K/V): Learned projections used to compute attention weights and context aggregation.
- Attention map: Matrix of token-to-token interaction weights.
- cls token: Learnable global token prepended to sequence; final global representation is read from it.
- Layer normalization (LayerNorm): Per-token feature normalization improving training stability.
- Norm-first (Pre-Norm): LayerNorm before sublayers; often stabilizes deep transformer optimization.
- FFN (Feed-Forward Network): Position-wise MLP inside each transformer block.
- GELU: Smooth non-linear activation used in transformer MLPs.
- Dropout: Random feature dropping during training for regularization.
- Head (prediction head): Small MLP mapping latent features to output score.
- Global head: Head using cls representation for final quality prediction.
- Per-scale head: Head applied to pooled tokens of one scale for consistency supervision.

### 29.5 Positional Encoding Terms

- Positional encoding: Added signal that informs model about token location.
- Hash-based positional encoding (in this project): Sum of learned embeddings for scale index, row index, and column index.
- Embedding table: Learnable lookup matrix mapping integer index to dense vector.
- Spatial identity: Relative/absolute location information in the image plane.
- Scale identity: Information about which resolution level a token belongs to.

### 29.6 Pretraining and Transfer Terms

- Pretraining: Training on a large source dataset (e.g., ImageNet) before task-specific fine-tuning.
- Fine-tuning: Adapting pretrained weights to target task data.
- ViT-B/32: Vision Transformer base model with patch size 32 used as weight source.
- Transfer learning: Reusing learned representations from one task/domain in another.
- Shape-matched loading: Loading only parameters whose dimensions match target module.
- Strict loading: Requiring exact key and shape match for all parameters.
- Frozen layer: Layer with requires_grad=False, so parameters are not updated.
- Unfreezing: Re-enabling gradient updates for previously frozen parameters.

### 29.7 Loss and Objective Terms

- Objective function: Total scalar target minimized during training.
- MSE (Mean Squared Error): Average squared difference between predicted and true scores.
- KL divergence: Asymmetric distance between two probability distributions.
- Score-to-distribution mapping: Converting scalar quality score into soft bin probabilities.
- Gaussian kernel smoothing: Weighting bins by Gaussian distance from score.
- Sigma: Gaussian width controlling how sharp or smooth soft distributions are.
- Epsilon: Small constant added for numerical stability.
- Pairwise ranking loss: Margin-based loss enforcing correct score ordering between sample pairs.
- Margin (m): Minimum desired separation between correctly ordered pair predictions.
- Lambda (loss weight): Coefficient controlling contribution of each loss term.
- Regularization: Any mechanism that reduces overfitting or enforces desired behavior.

### 29.8 Optimization and Training Terms

- Backpropagation: Gradient computation through computational graph.
- Gradient: Partial derivative of loss with respect to parameters.
- Optimizer: Rule for updating parameters using gradients.
- AdamW: Adaptive optimizer with decoupled weight decay.
- Weight decay: Parameter shrinkage term that helps regularization.
- Learning rate (LR): Step size of each optimization update.
- Warmup: Gradual LR increase in early phase for stable optimization.
- Cosine annealing: Smooth cosine-based LR decay schedule.
- Epoch: One full pass through training data.
- Iteration/step: One optimizer update.
- Early stopping: Halting training when validation metric stops improving.
- Patience: Number of non-improving epochs tolerated before stop.
- Gradient clipping: Limiting gradient norm to prevent exploding updates.
- Mixed precision (AMP): Using fp16/bfloat16 where safe for faster training and lower memory.
- GradScaler: Dynamic scaling of loss to avoid underflow in mixed precision.
- EMA (Exponential Moving Average of weights): Smoothed parameter copy for stabler validation.
- Checkpoint: Serialized training state (weights and optimizer/scheduler state).
- Resume training: Continue from saved checkpoint rather than retraining from scratch.

### 29.9 Evaluation and Statistics Terms

- Validation set: Data used for model selection and early stopping.
- Test set (blind split): Held-out data used only for final performance reporting.
- Inference: Forward pass without parameter updates.
- TTA (Test-Time Augmentation): Averaging predictions over transformed inputs at test time.
- Calibration: Post-hoc mapping from raw predictions to better-aligned target scale.
- Isotonic regression: Non-parametric monotonic calibration model.
- Residual: Prediction error (predicted - target).
- Prediction distribution: Histogram/statistical spread of model outputs.
- PLCC: Pearson linear correlation between predictions and targets.
- SROCC: Spearman rank correlation measuring monotonic ordering.
- KROCC: Kendall rank correlation measuring pairwise concordance.
- p-value: Probability of observing correlation at least this extreme under null hypothesis.
- Aggregate score: Sum of absolute PLCC, SROCC, and KROCC.

### 29.10 Complexity and Systems Terms

- Time complexity O(T^2 * D): Attention compute cost per layer with sequence length T and embedding dim D.
- Memory complexity: GPU/CPU memory required for activations, gradients, optimizer states, and parameters.
- VRAM: GPU memory used during training/inference.
- Throughput: Number of samples processed per unit time.
- Latency: Time to process one sample or one batch.
- Bottleneck: Component limiting overall speed or memory scalability.
- Quadratic scaling: Growth proportional to square of token count; key reason high-resolution scales are expensive.

### 29.11 Reproducibility Terms

- Random seed: Fixed initialization for repeatable pseudo-random operations.
- Determinism: Same inputs + same seed + same environment produce same outputs (within hardware limits).
- Configuration management: Centralizing experiment settings in one file (here, config.py).
- Ablation study: Controlled removal/change of components to isolate causal contribution.
- Baseline: Reference model used to contextualize performance claims.
- Fair comparison: Same data splits, metrics, and evaluation protocol across methods.

### 29.12 Practical Viva Vocabulary (Short Definitions)

- Underfitting: Model too simple or insufficiently trained; high bias.
- Overfitting: Model memorizes training data; weak generalization.
- Generalization: Performance transfer to unseen data.
- Robustness: Stability under noise, shift, or perturbations.
- Interpretability: Degree to which model behavior can be explained.
- Explainability: Post-hoc methods or narratives to justify predictions.
- Clinical relevance: Practical impact on real medical workflow and decision quality.
- Statistical significance: Evidence that observed effect is unlikely due to chance.
- Practical significance: Magnitude of effect meaningful in real use.

Use this glossary as your rapid reference before rehearsal. If an examiner uses a term unexpectedly, map it to one of these definitions and answer with confidence.

## 30. Learning Roadmap (What to Learn First, Next, and Last)

Follow this roadmap in order. Do not skip phases. Each phase has goals, exact files, and a completion checklist.

### Phase 0: Project Orientation (Day 1, 1-2 hours)

Goal:
- Understand the project purpose and where everything lives.

Read in this order:
1. [README.md](README.md)
2. [DefensePrep.md](DefensePrep.md)

Know before moving on:
- Problem statement (blind IQA for low-dose brain CT)
- Input/output definition
- Main models compared
- Main metrics used

Answer notes:
- Problem statement: In real clinical workflow, a clean reference CT is unavailable, so quality must be predicted from a single low-dose brain CT image (blind IQA setting).
- Input/output definition: Input is one CT slice; output is one continuous quality score in [0, 4] aligned with radiologist labels.
- Main models compared: CT-MUSIQ (proposed multi-scale transformer) and agaldran_combo (Swin-T + ResNet50 ensemble baseline).
- Main metrics used: PLCC, SROCC, KROCC, and Aggregate = |PLCC| + |SROCC| + |KROCC|.

Checkpoint:
- You can explain project in 60 seconds without reading notes.

Model answer:
- "This work solves blind quality assessment for low-dose brain CT by predicting radiologist-like quality scores from a single image. I use a multi-scale transformer with scale-aware positional encoding and a composite loss (MSE + ranking + KL consistency), then evaluate on a fixed train/val/test split with PLCC, SROCC, KROCC, and aggregate score against a strong ensemble baseline."

### Phase 1: Global Configuration and Experiment Rules (Day 1)

Goal:
- Learn all constants that control behavior across the codebase.

Read in this order:
1. [config.py](config.py)
2. [requirements.txt](requirements.txt)

Know before moving on:
- Data paths and naming conventions
- Split boundaries (train/val/test)
- Scales, patch size, model hyperparameters
- Training hyperparameters (lr, epochs, patience, stage settings)
- Loss weights

Answer notes:
- Data paths and naming: Images are in dataset/image with names like 0042.tif, labels in dataset/train.json with matching keys.
- Split boundaries: Train 0000-0699, Val 0700-0899, Test 0900-0999.
- Scale/patch/model core values: Scales [224, 384], patch size 32, embedding dim 768, 12 encoder layers, 8 heads.
- Training setup: Stage 1 frozen encoder for first epochs, Stage 2 full fine-tuning with warmup then cosine LR; early stopping uses patience.
- Loss weights: lambda_kl controls scale-consistency strength; ranking weight is defined in loss construction.

Checkpoint:
- You can answer: "Which constants define your experiment protocol?"

Model answer:
- "The experiment protocol is controlled by config.py: dataset paths and filename format, fixed split ranges, model scales/patch size/transformer dimensions, optimizer schedule parameters, loss weights, random seed, and checkpoint/result paths."

### Phase 2: Data Pipeline Internals (Day 2)

Goal:
- Understand exactly how one CT image becomes model-ready tensors.

Read in this order:
1. [dataset.py](dataset.py)

Know before moving on:
- How image loading works
- How augmentation is applied
- How multi-scale pyramid is built
- How patches and coordinates are generated
- Final dataloader output contract and tensor shapes

Answer notes:
- Image loading: Each TIFF is read from disk, cast to float32, and clipped to [0, 1].
- Augmentation: Only train split applies mild flips/rotation/crop/intensity/noise to preserve medical realism.
- Multi-scale pyramid: The same image is resized to each configured scale (default 224 and 384).
- Patch + coordinates: Non-overlapping 32x32 patches are extracted with coordinate tuple (scale_idx, row_idx, col_idx).
- Batch contract: patches [B, N, 3, 32, 32], coords [B, N, 3], score [B], image_id list.

Practice task:
- Draw tensor shapes at each step from image to patches/coords batch.

Checkpoint:
- You can explain why N=193 tokens for scales [224, 384] and patch 32.

Model answer:
- "At 224, grid is 224/32 = 7 so tokens are 7x7 = 49. At 384, grid is 384/32 = 12 so tokens are 12x12 = 144. Total N is 49 + 144 = 193."

### Phase 3: Architecture Core (Day 3)

Goal:
- Understand CT-MUSIQ modules and data flow through the network.

Read in this order:
1. [model.py](model.py)
2. [get_model.py](get_model.py)

Know before moving on:
- Patch embedding mechanics
- Hash positional encoding implementation
- cls token role
- Transformer block configuration
- Global and per-scale heads
- Forward-pass tensor shapes end-to-end
- Pretrained ViT-B/32 weight loading logic

Answer notes:
- Patch embedding: Conv2d with kernel=stride=32 converts each patch to a 768-d token.
- Hash positional encoding: Position vector = E_scale + E_row + E_col from coordinate indices.
- cls token role: Learnable token prepended to sequence; its final representation feeds global quality head.
- Transformer config: Encoder stack with multi-head self-attention + FFN (GELU, dropout, LayerNorm).
- Heads: One global head for final score, separate per-scale heads for consistency supervision.
- Shape flow: [B,N,3,32,32] -> [B,N,768] -> [B,N+1,768] -> heads -> global [B,1], scale list of [B,1].
- Pretraining: Shape-matched weights are loaded from ViT-B/32 (partial transfer with safe fallback).

Practice task:
- Reproduce the forward pass on paper with shape annotations.

Checkpoint:
- You can answer: "How are scale-specific tokens separated and scored?"

Model answer:
- "Scale tokens are contiguous in sequence because patches are concatenated scale-by-scale. For each scale, the model slices its token range, averages tokens, and sends the pooled vector to that scale head to produce one scale-specific score."

### Phase 4: Loss Engineering and Math (Day 4)

Goal:
- Defend each loss term mathematically and intuitively.

Read in this order:
1. [loss.py](loss.py)
2. [DefensePrep.md](DefensePrep.md) section 21

Know before moving on:
- MSE term behavior
- Score-to-distribution mapping for KL
- KL consistency meaning and effect
- Pairwise ranking loss and margin behavior
- Total objective and loss weights

Answer notes:
- MSE behavior: Penalizes absolute prediction error to match target scores numerically.
- Score-to-distribution: Scalar score is converted to soft Gaussian-weighted bin probabilities.
- KL consistency: Penalizes divergence between each scale distribution and global distribution.
- Ranking loss: Enforces correct pair ordering with margin, improving rank correlations.
- Total objective: L_total = L_mse + lambda_kl * L_kl + lambda_rank * L_rank.

Practice task:
- Explain one example where ranking loss helps even when MSE is reasonable.

Checkpoint:
- You can write and explain L_total from memory.

Model answer:
- "The objective is L_total = L_mse + lambda_kl * L_kl + lambda_rank * L_rank, where MSE optimizes absolute score regression, KL aligns scale and global predictions, and ranking optimizes pairwise order consistency."

### Phase 5: Training Dynamics and Optimization (Day 5)

Goal:
- Understand exactly how optimization is staged and stabilized.

Read in this order:
1. [train.py](train.py)

Know before moving on:
- Two-stage freezing/unfreezing logic
- Warmup and cosine scheduling flow
- AdamW setup and why used
- AMP and GradScaler role
- Gradient clipping and EMA usage
- Checkpointing and logging behavior

Answer notes:
- Two-stage logic: Stage 1 trains adaptation-sensitive parts while encoder is frozen; Stage 2 unfreezes all parameters.
- LR schedule: Stage-2 warmup avoids abrupt transitions; cosine decay stabilizes later convergence.
- AdamW: Adaptive optimization with decoupled weight decay, robust for transformer fine-tuning.
- AMP/GradScaler: Mixed precision speeds training and reduces memory while preserving gradient stability.
- Clipping/EMA: Clipping controls gradient spikes; EMA smooths parameter trajectory for stabler validation.
- Logging: Best checkpoint and per-epoch metrics are persisted for reproducibility.

Practice task:
- Explain why stage-1 freezing helps small medical datasets.

Checkpoint:
- You can answer: "What happens in epoch 1 vs epoch 6?"

Model answer:
- "Epoch 1 is Stage 1 with frozen encoder and higher adaptation-focused learning. Around epoch 6 (after Stage 1), the encoder is unfrozen, warmup begins for full fine-tuning, and the model transitions into end-to-end optimization."

### Phase 6: Evaluation and Statistical Meaning (Day 6)

Goal:
- Interpret model performance correctly and defend metric choices.

Read in this order:
1. [evaluate.py](evaluate.py)
2. [results/ct_musiq_results.csv](results/ct_musiq_results.csv)
3. [results/agaldran_combo/agaldran_combo_results.csv](results/agaldran_combo/agaldran_combo_results.csv)

Know before moving on:
- How predictions are generated on test/blind split
- PLCC/SROCC/KROCC definitions and differences
- Aggregate formula and meaning
- TTA and isotonic calibration paths
- Prediction CSV structure

Answer notes:
- Prediction generation: Model runs in eval mode over test split, storing predicted scores and targets per image.
- Metric differences: PLCC tests linear agreement, SROCC tests monotonic rank order, KROCC tests pairwise concordance.
- Aggregate meaning: Combined summary of linear fit + ranking quality.
- TTA/calibration: Optional flip averaging and isotonic regression can reduce variance and scale bias.
- CSV outputs: Per-image file includes prediction, target, error; summary file stores headline metrics.

Practice task:
- Compare two models and explain why one can have higher PLCC but lower KROCC.

Checkpoint:
- You can defend why multiple correlation metrics are needed.

Model answer:
- "One metric is incomplete: PLCC can be high even if local ordering is imperfect, while rank metrics may be high with calibration bias. Using PLCC, SROCC, and KROCC together captures complementary behavior, and aggregate summarizes all three."

### Phase 7: Baseline Fairness and Comparison Logic (Day 6)

Goal:
- Defend why your baseline comparison is fair and technically meaningful.

Read in this order:
1. [baseline_models.py](baseline_models.py)
2. [get_model.py](get_model.py)
3. [train.py](train.py) model selection path

Know before moving on:
- Swin-T + ResNet50 ensemble mechanism
- Shared training/evaluation protocol
- Why this baseline is strong

Answer notes:
- Ensemble mechanism: Baseline predicts one score from Swin-T and one from ResNet50, then averages them.
- Shared protocol: Same data split, same metrics, same evaluation script logic ensure fairness.
- Strength rationale: Two strong pretrained backbones reduce risk of weak-baseline bias.

Checkpoint:
- You can answer: "How did you ensure fair comparison across models?"

Model answer:
- "Both models use the same dataset split, target labels, metric suite, and evaluation pipeline. This controls confounders so differences are attributable to model design rather than protocol changes."

### Phase 8: Causal Evidence via Ablation (Day 7)

Goal:
- Use ablation results to justify design decisions causally.

Read in this order:
1. [ablation.py](ablation.py)
2. [config.py](config.py) ABLATION_CONFIGS
3. [results](results) ablation outputs if present

Know before moving on:
- What A1-A6 each isolates
- How to narrate A1->A2 and A2->A4 transitions
- How to discuss OOM/hardware constraints honestly

Answer notes:
- A1-A6 purpose: They isolate single-vs-multi-scale effect, KL effect, KL weight sensitivity, and scaling limits.
- A1->A2 narrative: Improvement indicates value of multi-scale representation.
- A2->A4 narrative: Improvement indicates added value of consistency regularization.
- OOM discussion: Memory failures in high-token settings are valid engineering findings, not experimental failures.

Checkpoint:
- You can answer: "Which component contributed most and how do you know?"

Model answer:
- "I infer contribution from controlled ablations: the largest aggregate gain between adjacent controlled settings indicates the strongest component effect, typically first from multi-scale addition and then from KL consistency depending on run results."

### Phase 9: Thesis Defense Synthesis (Final 2-3 Days)

Goal:
- Integrate technical details into clear defense answers.

Re-read in this order:
1. [DefensePrep.md](DefensePrep.md)
2. [README.md](README.md)
3. [writting/main.tex](writting/main.tex)
4. [writting/methodoloy.tex](writting/methodoloy.tex)

Know before defense day:
- 1-minute, 3-minute, and 10-minute narrative
- Key equations from memory
- Key numbers from memory
- Top limitations and future work

Answer notes:
- Narrative readiness: You should deliver short, medium, and full explanations with consistent technical core.
- Equations to memorize: Positional encoding sum, KL mapping equations, and total objective.
- Numbers to memorize: 1000 total images, 700/200/100 split, 193 tokens, metric definitions.
- Limitations: Dataset size, single-domain validation, and deployment generalization.

Checkpoint:
- You can handle interruption-style questions without losing structure.

Model answer:
- "When interrupted, I answer in five steps: goal, method choice, mechanism, evidence, limitation/next step. This keeps answers concise and technically complete under pressure."

## 31. Fast Weekly Schedule (If Time Is Limited)

If you only have 7 days, use this schedule:

1. Day 1: Phase 0 + Phase 1
2. Day 2: Phase 2
3. Day 3: Phase 3
4. Day 4: Phase 4
5. Day 5: Phase 5 + Phase 6
6. Day 6: Phase 7 + Phase 8
7. Day 7: Phase 9 mock viva

## 32. Progress Tracker (Mark as You Finish)

- [ ] Oriented with project overview and goals
- [ ] Memorized config and split logic
- [ ] Understood data pipeline and tensor contracts
- [ ] Understood architecture and positional encoding
- [ ] Understood and derived total loss
- [ ] Understood training stages and optimization internals
- [ ] Understood evaluation metrics and calibration options
- [ ] Understood baseline fairness argument
- [ ] Understood ablation causal narrative
- [ ] Rehearsed 1/3/10-minute scripts
- [ ] Practiced ultra-technical viva questions

When all boxes are checked, you are defense-ready at a strong technical level.