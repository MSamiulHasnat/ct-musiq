# LastMinute Prep

## CT-MUSIQ Thesis Defence - Speech (13-Slide Edition)

### Slide 1 - Title (30 sec)

"Honoured chair, esteemed committee members, good morning. My name is M Samiul Hasnat, student ID 2022521460113. Under the supervision of Professor Wenchao Du, I am pleased to present my undergraduate thesis: No-Reference Perceptual Assessment of Low-Dose CT Images Using Adapted MUSIQ Transformer. The work I will defend today is called CT-MUSIQ - a multi-scale Vision Transformer adapted specifically for medical image quality assessment."

Pause briefly, advance to TOC.

### Slide 2 - Table of Contents (20 sec)

"Over the next twenty minutes I will walk through ten sections - beginning with the introduction and research question, the clinical background, my research objectives, and the significance of this work. I will then turn to the methodology, the loss function design, the CT-MUSIQ architecture in detail, the experimental results, and finish with key findings and conclusions."

### Slide 3 - Introduction and Research Question (2 min)

"The story begins with a clinical trade-off. Modern Low-Dose CT protocols, guided by the ALARA principle - As Low As Reasonably Achievable - reduce patient radiation exposure by 50 to 80 percent compared to standard-dose scans. This is a major win for patient safety. But fewer photons reaching the detector mean noisier images: Poisson-driven streak artefacts, granular texture, and blurred soft-tissue boundaries - all of which impair a radiologist's diagnostic confidence.
Today, the gold standard for assessing whether a low-dose CT scan is diagnostically usable is manual radiologist review. This is labour-intensive, suffers from measurable inter-observer variability, and simply cannot scale to the volume of CT scans modern hospitals produce.
Existing automated approaches also fall short. Classical natural-scene-statistics methods like BRISQUE were designed for everyday photographs and collapse on calibrated Hounsfield-Unit data - they cannot handle the way CT pixel values represent calibrated tissue density rather than scene reflectance. CNN-based models, meanwhile, struggle with the long-range streak artefacts that span the full anatomical cross-section, because their receptive fields are too local.
So my research question, in plain terms, is this: Can a modern AI model learn to judge the quality of low-dose CT scans the way a trained radiologist would - accurately, automatically, and using only a single ordinary computer? That last constraint is essential. A useful clinical tool must run on the hardware hospitals actually have, not on data-centre clusters."

### Slide 4 - Background (1.5 min)

"Two background pillars deepen the picture.
The first is the clinical trade-off itself. Reducing photon flux produces a non-stationary Poisson plus Gaussian noise field - the noise distribution depends on local anatomy, so it is not uniform across the image. Streak artefacts propagate along Radon directions - the angles at which the X-ray beams traverse the body. And critically, abdominal soft tissues - the liver at 50 to 65 Hounsfield Units, the spleen at 45 to 65, the kidney cortex at 20 to 40 - share narrow attenuation bands. Quantum noise can easily mask the contrast between these tissues. This is precisely the perceptually demanding regime where image quality assessment matters most.
The second is the manual audit bottleneck. Radiologist quality assessment is labour-intensive, varies between readers, and consumes expensive expert time. Repeat scans cost between two hundred and one thousand US dollars each. Even a one-percent reduction in repeat rate prevents roughly 500 unnecessary scans per year for a 50,000-scan facility - saving cost, saving expert time, and reducing aggregate population radiation dose."

### Slide 5 - Research Objectives (1.5 min)

"Four objectives drove this research.
First, adapt the multi-scale MUSIQ Transformer to single-channel abdominal CT, with mandatory soft-tissue windowing - Level 40 HU, Width 350 HU - aligning the model's input to the perceptual space radiologists use during annotation.
Second, introduce scale-consistency KL regularisation at lambda=0.10, together with a pairwise margin-ranking loss with margin m=0.1, to combat overfitting on the modest 700-image training set.
Third, design a two-stage fine-tuning protocol - five epochs of frozen-encoder warm-up followed by 95 epochs of cosine-annealed full fine-tuning - that remains stable on a single 6 GB consumer GPU.
Fourth, independently reimplement the LDCTIQAC 2023 challenge winner - the Agaldran ensemble of Swin-T and ResNet-50 - under matched hardware and data conditions. This last step gives me a strict, fair, within-paper baseline, rather than relying on numbers from the original leaderboard."

### Slide 6 - Significance and Hypotheses (1 min)

"The significance has four immediate use cases. An accurate no-reference IQA system enables, first, real-time scan flagging at the scanner; second, retrospective auditing of CT archives without consuming radiologist time; third, objective benchmarking of LDCT reconstruction and denoising algorithms; and fourth, PACS integration as an automated quality gate before image distribution. CT-MUSIQ targets all four.
My hypotheses are twofold. First, that multi-scale Transformer attention can capture both global streak artefacts and local high-frequency noise simultaneously - something convolutional networks struggle with because of their limited receptive field. Second, that scale-consistency regularisation prevents per-scale overfitting and improves generalisation on small medical datasets, where data augmentation alone is insufficient."

### Slide 7 - Methodology (1.5 min)

"Two methodological pillars: architecture and training.
The architecture uses a multi-scale ViT-B/32 backbone with approximately 88 million parameters. It ingests two scales simultaneously: 49 patches at 224x224 resolution, and 144 patches at 384x384 - 193 tokens in total flowing through the transformer encoder. Hash-based factored positional encoding injects three pieces of identity per token: which scale it came from, its row, and its column. There are three prediction heads - one global head that produces the final score, plus two per-scale auxiliary heads used only for the KL consistency loss during training.
For training, I use a two-stage protocol. Stage one: 5 epochs of frozen-encoder warm-up at learning rate 3e-4, training only the prediction heads. This stabilises the heads on the new task before touching the pretrained encoder. Stage two: 95 epochs of full fine-tuning with linear warm-up plus cosine annealing from 5e-5 down to 1e-6. I use AdamW, exponential moving average with rho=0.999, AMP mixed precision, and gradient clipping at norm 1. The entire pipeline fits in 5.6 GB of VRAM - an RTX 3060 Laptop, completely off-the-shelf consumer hardware."

### Slide 8 - Loss Function (2 min)

"The training objective is a composite of three components.
First, MSE regression loss - the primary objective. Mean squared error between the model's predicted quality score and the ground-truth radiologist score. This penalises large prediction errors quadratically, driving the model toward absolute accuracy.
Second, scale-consistency KL loss, with weight lambda=0.10. This is the most important contribution of this work. For each input, the model produces three score predictions: one global, and one from each per-scale head. I convert each prediction into a soft Gaussian distribution over 20 score bins, then minimise the KL divergence between the per-scale distributions and the global one. The intuition: if the 224-scale prediction wildly disagrees with the 384-scale prediction for the same image, the model is overfitting to one scale's idiosyncrasies. Forcing them to agree is a strong, essentially free regulariser. The ablation confirms this - KL contributes plus 0.034 aggregate, the largest single architectural gain in the entire study, roughly twice the gain of adding the second scale alone.
Third, pairwise ranking loss, with weight lambda=0.5 and margin m=0.1. For every pair of images in a batch, if image A has a higher ground-truth score than image B, the model must predict A higher than B by at least 0.1. This operates on all B^2 ordered pairs in each batch, directly optimising the rank-correlation metrics - PLCC, SROCC, and KROCC - that the LDCTIQAC challenge uses for evaluation. This loss closes the gap between training objective and evaluation objective."

### Slide 9 - CT-MUSIQ Architecture (2 min)

"Let me walk through the architecture pipeline in detail.
We start on the left, with a single grayscale CT slice. In the multi-scale patch tokenisation block, the image is resized to two resolutions in parallel: 224x224 and 384x384. The smaller scale is divided into a 7x7 grid of 32x32 patches - that is 49 patches. The larger scale becomes a 12x12 grid - 144 patches. The two are concatenated, giving 193 patches total.
Next is token preparation and encoding. Because ViT-B/32 was pretrained on three-channel ImageNet images, I replicate the grayscale channel three times - identical RGB channels, which lets the pretrained weights work without architectural surgery. A learnable CLS token is prepended for global representation. The hash-based positional encoding module then injects scale, row, and column identity into each token - this is how the transformer knows whether a patch came from the small scale or the large scale, and where it sat spatially within its grid.
These tokens feed into the ViT encoder: 12 transformer blocks, each with multi-head self-attention - 8 heads - followed by an MLP. This is where the key advantage comes in: every patch can attend to every other patch, regardless of distance. That is how we capture long-range streak artefacts that span the whole abdomen, while still resolving local fine-grained noise - something a CNN with limited receptive field cannot do simultaneously.
Finally, the prediction heads produce the quality score. During training, the two per-scale heads also fire - they feed the KL consistency loss I just described."

### Slide 10 - Results (2 min)

"Now to the results.
On the held-out LDCTIQAC 2023 test set - 100 abdominal CT slices the model has never seen - CT-MUSIQ achieves an aggregate score of 2.7235. That breaks down as PLCC 0.9498, SROCC 0.9488, KROCC 0.8249. This surpasses every published leaderboard method except the top ensemble.
For the fair within-paper comparison, my reimplemented Agaldran ensemble - Swin-T plus ResNet-50, the LDCTIQAC 2023 winner - reaches 2.7682. That is a 0.045-point gap. Crucially, this gap falls within the 95 percent confidence interval for correlation estimates on a 100-image test set. With a Pearson correlation around 0.95 and n=100, the confidence interval is roughly 0.93 to 0.97. So statistically, CT-MUSIQ and Agaldran are tied - the difference is not significant at the alpha=0.05 level.
Both deep learning models outperform the BRISQUE baseline at 2.1219 by approximately 0.60 aggregate points. That is unambiguous evidence that classical natural-scene-statistics methods are insufficient for calibrated medical CT.
The headline of this slide: a single 88-million-parameter Transformer model, trained on a 6 GB consumer GPU, statistically ties the ensemble winner of an international medical imaging challenge."

### Slide 11 - Key Findings (1.5 min)

"Two findings stand out.
First, single-model competitive. CT-MUSIQ surpasses every published leaderboard submission except the top ensemble - validating that multi-scale Transformer adaptation works for medical IQA. The ablation reveals that KL regularisation contributes plus 0.034 aggregate - the largest single architectural gain - roughly twice the plus 0.022 from adding the second scale alone. So the loss design matters more than simply scaling up the architecture.
Second, the ensemble edge. The Swin-T plus ResNet-50 ensemble - aggregate 2.7682, PLCC 0.9608 - reveals complementary strengths between window-based attention and pyramid convolution. With only 700 training images, architectural diversity through ensembling appears to regularise more effectively than KL constraints on a single deeper model. This is an instructive lesson for data-scarce medical imaging: when data is the bottleneck, sometimes two different architectures together beat one carefully regularised architecture."

### Slide 12 - Conclusion (1.5 min)

"Three conclusions to close.
One - multi-scale works. ImageNet-pretrained ViTs do transfer to low-dose CT no-reference IQA, when paired with clinical soft-tissue windowing and a two-stage fine-tuning protocol. No CT-specific pretraining is required. This significantly lowers the barrier to entry for medical IQA research.
Two - KL regularisation is the key insight. Scale-consistency KL loss at lambda=0.10 is the most impactful single component, contributing plus 0.034 aggregate in ablation - roughly twice the gain of the second scale alone. The general lesson: in data-scarce settings, regularising the relationship between auxiliary heads is more valuable than scaling the architecture.
Three - future work. Three concrete directions: multi-scanner cross-institution validation to test generalisation beyond LDCTIQAC; CT-MUSIQ x Agaldran ensembling, which I expect would cross 2.80 aggregate; and VLM-based explainable quality reports for clinical PACS deployment, so each score arrives with a human-readable justification radiologists can audit."

### Slide 13 - Thank You (30 sec)

"Thank you for your attention. I would like to express my deepest gratitude to my adviser, Professor Wenchao Du, for his guidance throughout this research, and to the committee for the opportunity to defend this work today. I welcome your questions and look forward to the discussion."

---

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

---

## Probable Defense Board Questions — Verified Against Thesis

> All answers verified against the thesis document as the ultimate ground truth, cross-referenced with the codebase.

---

### Architecture & Design Choices

**Q1. Why did you choose MUSIQ specifically over other ViT variants like Swin Transformer or DeiT?**

Standard ViT and DeiT require fixed-resolution inputs to maintain constant sequence lengths for learnable position embeddings — resizing a 512×512 CT slice to a single resolution destroys high-frequency quantum noise signatures critical for quality assessment. The thesis calls this the "input resolution paradox." Swin Transformer partially addresses it with shifted-window attention but still requires a fixed spatial resolution per scale, making two-resolution inputs awkward. MUSIQ was specifically designed for multi-scale IQA: it concatenates patches from multiple resolutions into one variable-length sequence and handles it with hash-based positional encoding — PE = E_scale(s) + E_row(r) + E_col(c) — requiring no padding, masking, or re-engineering. CNNs were rejected because stacked convolution progressively destroys high-frequency noise signatures while failing to model the long-range streak artefacts that span the full anatomical cross-section.

---

**Q2. Why scales 224 and 384 specifically? Did you experiment with three scales?**

224 matches the native input size of pretrained ViT-B/32. 384 was the largest second scale fitting within the 6 GB VRAM budget (batch size 10, AMP enabled): 7×7=49 patches at 224, 12×12=144 at 384 = 193 tokens total, peak VRAM 5.2–5.6 GB. A third scale (512) was tested as ablation A6, which ran successfully — 449 tokens, Aggregate 2.296, marginally below A4's 2.305. The thesis attributes this to slower convergence at the longer sequence within the 30-epoch ablation budget rather than a genuine signal deficit, and recommends full 100-epoch training of A6 to reach a definitive conclusion.

---

**Q3. Grayscale channel replication — does it introduce bias or redundant computation?**

Yes, mild redundancy: the patch embedding Conv2d processes the same signal three times. The thesis justifies this as the best available option given the constraints: the pretrained ViT-B/32 patch embedding is shaped [768, 3, 32, 32], and replicating the greyscale channel preserves these weights intact with no re-initialisation cost. The thesis explicitly states "No artificial colour information is introduced." The alternative — reinitialising the embedding for a single-channel input — would sacrifice pretrained feature detectors, which is impractical with only 700 training images.

---

**Q4. Why ViT-B/32 (patch size 32) and not ViT-B/16?**

Patch size 32 was chosen for direct compatibility with the pretrained ViT-B/32 from the timm library (`vit_base_patch32_224`). The thesis states this as the primary reason. The VRAM constraint is also implicit: patch size 16 would produce 772 tokens (vs 193), quadrupling attention cost and very likely exceeding 6 GB VRAM. No formal ablation comparing 32 vs 16 was run.

---

**Q5. CLS token versus mean pooling for the global head — did you try alternatives?**

The CLS token for global prediction and mean pooling for per-scale auxiliary heads follows "standard Vision Transformer design" — cited as such in the thesis with no separate ablation comparing the two. The CLS token at position 0 aggregates context from all 193 patch tokens across all 12 transformer layers. Per-scale auxiliary heads apply mean pooling over their respective scale's tokens. No ablation comparing CLS vs global mean pool was run.

---

### Loss Function

**Q6. How were λ=0.10 (KL) and λ=0.5 (ranking) determined?**

λ=0.10 was determined by ablation. Three values were tested: A3 (λ=0.05, Agg 2.289), A4 (λ=0.10, Agg 2.305), A5 (λ=0.20, Agg 2.281) — all two-scale, 30 epochs. λ=0.10 is robustly optimal: 0.05 under-regularises, 0.20 over-regularises by pushing per-scale heads too strongly at the expense of MSE. The performance spread across A3–A5 is only 0.024, suggesting mild sensitivity. λ_rank=0.5 was not ablated — it was set by judgment, not grid search.

---

**Q7. What if the global head is wrong — could KL propagate a bad estimate to both scale heads?**

The thesis directly addresses this. The formula is D_KL(p(q̂_s) ‖ p(q̂)) — per-scale approximates global, global is the target. The justification: the global head accesses the CLS token, which attends over all 193 tokens from both scales across all 12 transformer layers — it has strictly more context than any per-scale head, which only sees mean-pooled tokens within one scale. The asymmetry of KL divergence "severely penalises per-scale predictions that assign high probability to bins the global head considers very unlikely, preventing extreme scale-specific outlier predictions." Per-scale scores are discarded at inference — only the global head's output is used.

---

**Q8. Why 20 bins for the KL soft distribution? Is it sensitive?**

K=20 is specified in Table 4.1 with σ=0.5. The thesis provides no explicit justification for K=20 and no ablation over the bin count. If pressed: 20 bins over [0,4] gives centers spaced ~0.21 apart, and with σ=0.5 each score's mass spans ~2–3 bins — enough smoothness for meaningful KL gradients without excessive computation. This is a reasonable but unevidenced choice.

---

**Q9. Batch size and pairwise ranking — any stability issues?**

Batch size is 10 (Table 4.1). The thesis states: "for a batch of B=10, there are 10×10=100 pairs, computed via outer-product matrix operations." Ranking loss = (1/B²) ΣΣ max(0, m − sgn(qᵢ−qⱼ)·(q̂ᵢ−q̂ⱼ)), m=0.1. Batch size 10 was the VRAM maximum. No ablation over batch size was run.

---

### Dataset & Generalisation

**Q10. One dataset, two institutions — how confident are you in generalisation?**

Not confident yet. The thesis explicitly states the study "does not claim generalisation to other CT anatomical regions, multi-slice volumetric assessment, or other modalities." Section 6.3 acknowledges: scanners differ in detector geometry, reconstruction kernel, and noise power spectrum in ways directly relevant to the features the model learned. All acquisitions come from two institutions under a challenge-standardised protocol. A quality estimator calibrated on Mayo Clinic and NCC images "may produce systematically biased predictions in a different institutional environment." Multi-centre cross-scanner validation is listed as the most immediately pressing future work.

---

**Q11. Why not augment more aggressively with only 700 training images?**

CT images are calibrated physical measurements in HU, and their noise texture carries diagnostic meaning. The thesis states: "Aggressive colour distortions or large geometric deformations would alter perceived quality in ways that do not correspond to any real imaging artefact, introducing label noise." The pipeline applies: horizontal flip (p=0.5), vertical flip (p=0.5), rotation ±7° in float-mode PIL, random crop 85–100% area, brightness/contrast jitter ±0.05, Gaussian noise std 0.005–0.02. MixUp was specifically considered and rejected: "the perceptual quality of a pixel-blended CT image does not follow a convex combination of the component scores." CutMix and random erasing were noted as feasible but not applied.

---

**Q12. How were the ground-truth labels created, and how does label noise affect metrics?**

Multiple radiologists independently scored each image on a five-point Likert scale (0=completely unacceptable, 4=excellent quality). The released labels are averaged consensus scores, yielding continuous float targets. The thesis does not state the exact number of radiologists or an ICC value — these are defined by the LDCTIQAC 2023 challenge organisers. Label noise affects all methods equally and does not bias the comparison. CT-MUSIQ's PLCC of 0.9498 is at the empirical ceiling typically reported in well-designed IQA studies (0.95–0.97), suggesting the model has captured most of the explainable variance.

---

**Q13. Was non-abdominal CT tested — chest or head?**

No. The thesis explicitly scopes the work to "axial abdominal CT slices from LDCTIQAC 2023" and states it "does not claim generalisation to other CT anatomical regions (e.g., chest or orthopaedic CT)." The abdominal soft-tissue regime — liver 50–65 HU, spleen 45–65 HU, kidney cortex 20–40 HU — shapes what quality features matter; different anatomy would require retraining.

---

### Results & Statistical Validity

**Q14. n=100 is small — can you claim statistical significance?**

The thesis is clear: "The performance differences between CT-MUSIQ (2.7235), the Agaldran baseline (2.7682), and the top leaderboard submissions are not statistically significant at the 5% level." The 95% CI for Pearson r≈0.95 at n=100 is approximately [0.93, 0.97]. The PLCC gap between CT-MUSIQ (0.9498) and the local Agaldran (0.9608) is 0.011, which the thesis explicitly says "is not statistically significant at the 5% level." The thesis also states: "the 100-image test set provides insufficient statistical power to claim significant performance differences between methods separated by fewer than 0.02 aggregate points." Bootstrap CIs were not computed — acknowledged as a strengthening step.

---

**Q15. Any baselines beyond BRISQUE — NIQE, PIQE?**

NIQE is mentioned in Section 2.4 as an NSS method that fails on calibrated CT data but its numerical results are not in the comparison table. PIQE does not appear in the thesis. The comparison table includes eight published methods from the LDCTIQAC 2023 leaderboard, including an SNR baseline at Aggregate 2.4026 (PLCC 0.8226, SROCC 0.8748, KROCC 0.7052) — a physically motivated signal metric that still falls 0.60 points below CT-MUSIQ, confirming neither NSS statistics nor physical SNR captures radiologist-defined perceptual quality.

---

**Q16. What is the single-scale, no-KL true baseline score?**

Ablation A1: single scale [224], λ_KL=0.0, 30 epochs — Val Aggregate 2.249. The full ladder from Table 4.2: A1=2.249 → A2=2.271 (two scales, no KL, +0.022) → A4=2.305 (two scales, KL 0.10, +0.034). KL regularisation is the largest single gain — approximately twice the multi-scale contribution. A6 (three scales, KL 0.10) = 2.296, marginally below A4, attributed to slower convergence at 449 tokens within 30 epochs.

---

**Q17. Did PLCC and SROCC ever diverge significantly?**

CT-MUSIQ achieves PLCC 0.9498 — marginally exceeding the published agaldran (0.9491) — while the local Agaldran leads on SROCC (0.9605 vs 0.9488) and KROCC (0.8469 vs 0.8249). The thesis states: "The KROCC gap is the most pronounced among the three metrics (+0.022) and is the primary driver of the overall aggregate difference." CT-MUSIQ's PLCC advantage over the published leaderboard result suggests better absolute score calibration, which the thesis attributes to the KL loss constraining per-scale and global predictions to agree. The ensemble's SROCC/KROCC advantage reflects complementary feature representations giving a more informative relative ordering signal.

---

### Clinical Relevance

**Q18. A single scalar — isn't that an oversimplification of holistic radiologist judgment?**

Yes, and the thesis acknowledges it. The scalar inherits the limitations of the LDCTIQAC 2023 Likert annotation — a collapsed consensus judgment, not a breakdown by artefact type, anatomical region, or diagnostic task. The thesis frames the scalar as sufficient for the four targeted use cases: real-time scan flagging, retrospective archive auditing, reconstruction benchmarking, and PACS quality gating. The richer output direction — VLM-based explainable quality reports replacing the scalar head with a language decoder — is proposed as a longer-horizon future work.

---

**Q19. Inference latency on your RTX 3060 — fast enough for real-time flagging?**

Inference latency per slice was not benchmarked in this work — the thesis does not report a per-image prediction time. Training throughput is reported (one epoch over 700 images ≈ 31–40 seconds), giving a rough pipeline sense, but formal inference timing was not done. The thesis proposes prospective clinical PACS deployment as a future validation step where latency would need to be measured and verified.

---

**Q20. How do you handle edge cases — motion artefacts, metal implants, contrast agents?**

The thesis does not address these edge cases. The only distortion types characterised in LDCTIQAC 2023 are noise level (N0–N3) and streak artefacts (S0–S2). Motion artefacts, metal implant streaks, and contrast agent effects are qualitatively different degradation types not present in the training distribution. The thesis makes no claim of robustness to them. Handling pathological edge cases would require dedicated training data.

---

### Comparison to Agaldran

**Q21. Did you try a CT-MUSIQ + Agaldran ensemble?**

No. The thesis proposes it explicitly as future work: "A CT-MUSIQ + Agaldran ensemble would exploit the complementary strengths observed in Section 5.2 — CT-MUSIQ's PLCC advantage and the Agaldran's superiority in SROCC and KROCC — and averaging their predictions would likely outperform either alone." It was not attempted in this thesis.

---

**Q22. Your reimplemented Agaldran scored higher than the original published result (2.7682 vs 2.7427). How is that possible, and how faithful is the reimplementation?**

The thesis is explicitly upfront about this. Section 1.6 states: "The Agaldran baseline is not the original team's submission but an independent reimplementation motivated by their published architectural choices, trained under the same conditions as CT-MUSIQ." The reimplementation uses the same pretrained Swin-T and ResNet-50 backbones (from timm and torchvision), simple quality heads, 0.5/0.5 score averaging, and the same training loop as CT-MUSIQ. The higher score (2.7682 vs 2.7427) "may reflect independently optimised training conditions rather than architectural superiority and cannot be taken as evidence that the reimplementation surpasses the original in any absolute sense." The within-paper comparison is therefore approximate, and the published leaderboard figures are cited for reference.

---

**Q23. If you had six more months, what would you do?**

Four directions from Section 6.4:

One — CT-MUSIQ × Agaldran ensemble: score-level averaging to exploit CT-MUSIQ's PLCC advantage and Agaldran's SROCC/KROCC superiority. Predicted to outperform either alone.

Two — Multi-centre cross-scanner validation: the most pressing priority. Requires assembling a multi-institution, multi-scanner annotated dataset across at least four scanner platforms, repeating the MOS annotation protocol at each site, and evaluating domain adaptation strategies including scanner-conditioned normalisation layers, self-supervised feature alignment, or lightweight scanner-specific regression heads.

Three — VLM-based explainable quality reports: replace the scalar prediction head with a language decoder to produce radiologist-style natural-language justifications alongside numerical scores, directly enabling PACS deployment with the explainability layer that regulatory review would require.

Four — Self-supervised CT pretraining via masked image modelling in HU space: yield stronger CT-domain representations than ImageNet-pretrained ViT-B/32, directly addressing the chronic scarcity of annotated CT quality data constraining all current approaches.
