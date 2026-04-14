# CT-MUSIQ: Blind Quality Assessment for Brain CT

CT-MUSIQ is an undergraduate thesis project for no-reference image quality assessment (IQA) of low-dose brain CT images.

The repository includes:
- A custom multi-scale transformer model (`ct_musiq`)
- A strong baseline ensemble (`agaldran_combo`: Swin-T + ResNet50)
- End-to-end training and evaluation scripts
- Ablation tooling for thesis experiments

## 1. What This Project Does

Given a CT slice, the model predicts a quality score in the range `[0, 4]` that matches radiologist annotations.

Primary evaluation metrics:
- PLCC (Pearson Linear Correlation Coefficient)
- SROCC (Spearman Rank-Order Correlation Coefficient)
- KROCC (Kendall Rank-Order Correlation Coefficient)
- Aggregate = `|PLCC| + |SROCC| + |KROCC|`

## 2. Repository Structure

```text
ct-musiq/
  ablation.py               # Runs ablation experiments (A1-A6)
  baseline_models.py        # Swin-T + ResNet50 baseline models
  config.py                 # Centralized project configuration
  dataset.py                # Dataset pipeline, patch extraction, dataloaders
  evaluate.py               # Test/blind split evaluation and CSV export
  get_model.py              # Model factory (ct_musiq or agaldran_combo)
  loss.py                   # MSE + ranking + scale-consistency KL losses
  model.py                  # CT-MUSIQ architecture
  train.py                  # Main training loop and checkpointing
  requirements.txt          # Python dependencies

  dataset/
    train.json              # Labels (filename -> score)
    image/                  # CT slices (.tif)

  results/                  # Training logs, checkpoints, prediction/results CSVs
  writting/                 # Thesis LaTeX files
```

## 3. Data Layout and Splits

Expected data format:
- Images in `dataset/image/` named as `0000.tif` to `0999.tif`
- Labels in `dataset/train.json` using keys like `"0042.tif"`

Default split logic from `config.py`:
- Train: indices `0000-0699` (700 images)
- Validation: indices `0700-0899` (200 images)
- Test/Blind: indices `0900-0999` (100 images)

## 4. Model Summary

### 4.1 CT-MUSIQ (`ct_musiq`)

Core design:
- Multi-scale pyramid (`[224, 384]` by default)
- Patch size `32` -> `49 + 144 = 193` tokens per image
- Hash-based positional encoding using `(scale_idx, row_idx, col_idx)`
- Transformer encoder + global quality head
- Per-scale heads for consistency regularization

Training setup highlights:
- Two-stage training:
  - Stage 1: encoder frozen (`STAGE1_EPOCHS`)
  - Stage 2: full fine-tuning + warmup + cosine schedule
- Mixed precision automatically enabled on CUDA
- Gradient clipping and early stopping
- EMA support for more stable validation

### 4.2 Baseline (`agaldran_combo`)

- Swin-T and ResNet50 backbones with ImageNet pretraining
- Score-level ensemble: final score is the average of both
- Shares training/evaluation pipeline with CT-MUSIQ for fair comparison

## 5. Environment Setup

The project was prepared with a Python 3.12 virtual environment.

### 5.1 Create and activate venv (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 5.2 Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

## 6. How to Run

All commands below should be run from the repository root.

### 6.1 Train CT-MUSIQ (default)

```powershell
python train.py
```

Useful options:

```powershell
python train.py --model ct_musiq --epochs 100 --batch_size 10 --lambda_kl 0.10
python train.py --resume results/ct_musiq/ct_musiq_best.pth
```

### 6.2 Train baseline model

```powershell
python train.py --model agaldran_combo
```

### 6.3 Evaluate on test/blind split

Evaluate CT-MUSIQ:

```powershell
python evaluate.py --model ct_musiq
```

Evaluate baseline:

```powershell
python evaluate.py --model agaldran_combo
```

Optional evaluation extras:

```powershell
python evaluate.py --model ct_musiq --tta
python evaluate.py --model ct_musiq --tta --calibrate
python evaluate.py --model ct_musiq --checkpoint results/ct_musiq/ct_musiq_best.pth
```

### 6.4 Run ablation experiments

```powershell
python ablation.py
python ablation.py --configs A1 A2 A4
python ablation.py --skip A6
```

## 7. Output Files

### 7.1 Training outputs

Per model, training writes to:
- `results/<model>/<model>_best.pth` (best checkpoint)
- `results/<model>/<model>_training_log.csv` (epoch-by-epoch log)

### 7.2 Evaluation outputs

For `ct_musiq`:
- `results/ct_musiq_predictions.csv`
- `results/ct_musiq_results.csv`

For baseline models:
- `results/<model>/<model>_predictions.csv`
- `results/<model>/<model>_results.csv`

### 7.3 Ablation outputs

- `results/ablation_results.csv`
- `results/ablation_<Ai>_best.pth` for each ablation run

## 8. Current Known Blind/Test Results

Recent runs in this workspace reported:

- `ct_musiq`
  - PLCC: `0.9498`
  - SROCC: `0.9488`
  - KROCC: `0.8249`
  - Aggregate: `2.7235`

- `agaldran_combo`
  - PLCC: `0.9608`
  - SROCC: `0.9605`
  - KROCC: `0.8469`
  - Aggregate: `2.7682`

## 9. Troubleshooting

- If `timm` import fails:
  - Reinstall requirements: `pip install -r requirements.txt`
- If CUDA is unavailable:
  - Scripts automatically run on CPU (slower)
- If Windows DataLoader issues appear:
  - Keep `num_workers=0` (already default in scripts)
- If checkpoint not found during evaluation:
  - Train first, or pass `--checkpoint <path>`

## 10. Reproducibility Notes

- Seed is configured in `config.py` (`SEED = 42`)
- All key hyperparameters and paths are centralized in `config.py`
- Dataset split boundaries are deterministic and index-based

## 11. Author

M Samiul Hasnat  
Sichuan University  
Undergraduate Thesis: CT-MUSIQ
