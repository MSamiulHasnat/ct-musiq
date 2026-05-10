# LastMinute Prep

## CT-MUSIQ Thesis Defence — Speech (7–10 Min Edition)

> **Target pace:** ~130 words/min. Total script ≈ 950 words → ~7.5 min.
> Stay on each slide's idea, land the sentence, move on. No improvised tangents.

---

### Slide 1 — Title (15 sec)

"Good morning. I'm M Samiul Hasnat, student ID 2022521460113, supervised by Professor Wenchao Du. My thesis is CT-MUSIQ: a multi-scale Vision Transformer adapted to automatically judge the quality of Low-Dose CT images — no radiologist required."

---

### Slide 2 — Table of Contents (10 sec)

"Here's the roadmap: problem, background, objectives, architecture, loss design, results, and conclusions."

---

### Slide 3 — Introduction & Research Question (75 sec)

"Low-Dose CT protocols follow the ALARA principle — As Low As Reasonably Achievable. Reducing dose by 50 to 80 percent saves patients from radiation, but fewer photons create Poisson-driven streak artefacts, granular noise, and blurred soft-tissue boundaries.

Today's quality check is a radiologist manually reviewing every scan. That doesn't scale. Existing automated methods fail too: BRISQUE was built for natural-scene photos and collapses on calibrated Hounsfield-Unit data. CNNs can't capture streak artefacts that span the full cross-section — their receptive field is too small.

My research question: **Can an AI model judge Low-Dose CT quality the way a radiologist would — accurately, automatically, and on a single consumer GPU?**"

---

### Slide 4 — Background (40 sec)

"Two facts drive the problem.

First, the noise physics. Reducing photon flux creates a non-stationary Poisson plus Gaussian noise field. Streak artefacts travel along Radon directions across the whole image. Abdominal soft tissues — liver 50–65 HU, spleen 45–65 HU, kidney cortex 20–40 HU — share such narrow attenuation windows that even modest quantum noise erases the contrast between them.

Second, the cost. Repeat scans run $200–$1,000 each. A 1% reduction in repeat rate for a 50,000-scan facility prevents 500 unnecessary scans per year."

---

### Slide 5 — Research Objectives (45 sec)

"Four objectives, briefly.

One — adapt MUSIQ to single-channel CT with mandatory soft-tissue windowing at Level 40 HU, Width 350 HU, so the model sees what radiologists see.

Two — introduce scale-consistency KL regularisation at λ=0.10 plus a pairwise ranking loss to fight overfitting on a small 700-image dataset.

Three — design a two-stage fine-tuning protocol: 5 epochs frozen-encoder warm-up, then 95 epochs cosine-annealed full fine-tuning — all within 6 GB VRAM.

Four — independently reimplement the LDCTIQAC 2023 challenge winner, the Agaldran ensemble, under identical hardware and data for a fair within-paper baseline."

---

### Slide 6 — Significance (20 sec)

"A working NR-IQA system enables four things: real-time scan flagging at acquisition, retrospective archive auditing, objective benchmarking of reconstruction algorithms, and an automated PACS quality gate — all without consuming expert radiologist time."

---

### Slide 7 — Methodology (50 sec)

"The architecture: a ViT-B/32 backbone, 88 million parameters, processing two scales in parallel. The 224×224 image is split into a 7×7 grid — 49 patches. The 384×384 image into a 12×12 grid — 144 patches. Concatenated: 193 tokens total.

Grayscale is replicated to three identical channels so the ImageNet pretrained weights apply without surgery. Hash-based positional encoding injects scale, row, and column identity into each token. Three prediction heads: one global score, two per-scale auxiliary heads used only during training for the KL consistency loss.

Full training pipeline: AdamW, EMA at ρ=0.999, AMP mixed precision, gradient clip at norm 1. Fits in 5.6 GB VRAM on a consumer RTX 3060 Laptop."

---

### Slide 8 — CT-MUSIQ Architecture (30 sec)

"The key architectural advantage is self-attention. Every one of the 193 tokens can attend to every other token in a single forward pass. That means a patch from the top of the image can directly communicate with a patch at the bottom — capturing long-range Radon streak artefacts while simultaneously resolving local fine-grained noise. A CNN with a bounded receptive field cannot do both at once."

---

### Slide 9 — Loss Function (55 sec)

"The training objective has three components.

**MSE** is the base regression loss — drives absolute score accuracy.

**Scale-consistency KL loss at λ=0.10** is the core contribution. For each image, the model produces three score predictions: global, scale-1, and scale-2. I convert each to a soft Gaussian over 20 score bins and minimise KL divergence between the per-scale distributions and the global one. If scale-1 and scale-2 disagree, the model is overfitting to one scale's noise. Forcing agreement is a strong, essentially free regulariser. Ablation confirms: +0.034 aggregate — the single largest architectural gain in the study.

**Pairwise ranking loss at λ=0.5, margin m=0.1** operates on all B² ordered pairs per batch. It directly optimises PLCC, SROCC, and KROCC — the exact metrics used in evaluation."

---

### Slide 10 — Results (55 sec)

"On the held-out LDCTIQAC 2023 test set — 100 CT slices the model never saw during training — CT-MUSIQ achieves **Aggregate 2.7235**: PLCC 0.9498, SROCC 0.9488, KROCC 0.8249.

My reimplemented Agaldran ensemble — Swin-T plus ResNet-50 — reaches 2.7682. The gap is 0.045 points. With n=100 and Pearson r≈0.95, the 95% confidence interval is roughly ±0.03. **Statistically, these two models are tied.**

Both deep learning models crush the BRISQUE baseline at 2.1219 — a 0.60-point gap — unambiguous evidence that classical NSS methods do not work on calibrated Hounsfield-Unit data.

**Headline:** a single 88M-parameter Transformer on a 6 GB consumer GPU statistically ties the winner of an international medical imaging challenge."

---

### Slide 11 — Key Findings (25 sec)

"Two takeaways.

First: KL regularisation contributes +0.034 aggregate — twice the gain of simply adding the second scale (+0.022). The loss design matters more than the architecture scale.

Second: at 700 training images, the Agaldran ensemble beats a single deeper model. The lesson for data-scarce medical imaging: architectural diversity through ensembling can regularise more effectively than KL constraints alone."

---

### Slide 12 — Conclusion (30 sec)

"Three conclusions.

One — ImageNet-pretrained ViTs transfer to LDCT NR-IQA with clinical windowing and two-stage fine-tuning. No CT-specific pretraining needed.

Two — scale-consistency KL regularisation is the key insight: the most impactful single component in ablation.

Three — future directions: multi-scanner cross-institution validation; CT-MUSIQ × Agaldran ensembling, which I expect would cross Aggregate 2.80; and VLM-based explainable quality reports for PACS deployment."

---

### Slide 13 — Thank You (10 sec)

"Thank you. I'm grateful to Professor Wenchao Du for his guidance, and to the committee for their time. I welcome your questions."

---

## Quick Numbers to Know Cold

| Metric | CT-MUSIQ | Agaldran | BRISQUE |
|--------|----------|----------|---------|
| Aggregate | **2.7235** | 2.7682 | 2.1219 |
| PLCC | 0.9498 | 0.9608 | — |
| SROCC | 0.9488 | — | — |
| KROCC | 0.8249 | — | — |

- **KL loss:** λ=0.10, 20 score bins, +0.034 aggregate
- **Ranking loss:** λ=0.5, margin m=0.1
- **Scales:** 49 patches (224×224) + 144 patches (384×384) = **193 tokens**
- **Training:** 5 frozen + 95 full = 100 epochs, LR 3e-4 → 5e-5 → 1e-6
- **VRAM:** 5.6 GB on RTX 3060 Laptop (6 GB card)
- **Dataset:** 700 train / 100 test (LDCTIQAC 2023)
- **Windowing:** Level 40 HU, Width 350 HU (soft tissue)
- **Gap significance:** 0.045 gap, within 95% CI → **not statistically significant**
