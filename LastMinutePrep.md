# LastMinute Prep

## CT-MUSIQ Thesis Defence - Speech

### Slide 1 - Title (30 sec)

"Honoured chair, esteemed committee members, good morning. My name is M Samiul Hasnat, student ID 2022521460113. Under the supervision of Professor Wenchao Du, I am pleased to present my undergraduate thesis: No-Reference Perceptual Assessment of Low-Dose CT Images Using Adapted MUSIQ Transformer. The work I will defend today is called CT-MUSIQ - a multi-scale Vision Transformer adapted specifically for medical image quality assessment."

Pause briefly, advance to TOC.

### Slide 2 - Table of Contents (20 sec)

"Over the next twenty minutes I will walk through ten sections. I will begin with the clinical motivation and research question, then cover the research highlights, background, objectives, and significance. I will then turn to the methodology, the CT-MUSIQ architecture in detail, the loss function design, the experimental results, and finish with the key findings and conclusions."

### Slide 3 - Introduction and Research Question (1.5 min)

"The story begins with a clinical trade-off. Modern Low-Dose CT protocols, guided by the ALARA principle - As Low As Reasonably Achievable - reduce patient radiation exposure by 50 to 80 percent compared to standard-dose scans. This is a major win for patient safety. But fewer photons reaching the detector mean noisier images: Poisson-driven streak artefacts, granular texture, and blurred soft-tissue boundaries - all of which impair a radiologist's diagnostic confidence.
Today, the gold standard for assessing whether a low-dose CT scan is diagnostically usable is manual radiologist review. This is labour-intensive, suffers from measurable inter-observer variability, and simply cannot scale to the volume of CT scans modern hospitals produce.
So my research question, in plain terms, is this: Can a modern AI model learn to judge the quality of low-dose CT scans the way a trained radiologist would - accurately, automatically, and using only a single ordinary computer? That last constraint is essential. A useful clinical tool must run on the hardware hospitals actually have, not on data-centre clusters."

### Slide 4 - Research Highlights (1 min)

"Four points define this work. First, the clinical need: low-dose CT introduces non-stationary noise and Radon-direction streaks that obscure subtle abdominal lesions. Second, the research gap: classical natural-scene-statistics methods like BRISQUE collapse on calibrated Hounsfield-Unit data - they were designed for natural photographs, not medical CT. Standard CNNs, meanwhile, struggle with the long-range streaks that span the full anatomical cross-section. Third, my proposed solution: CT-MUSIQ - a multi-scale Vision Transformer with mandatory soft-tissue windowing, scale-consistency KL regularisation, and a pairwise ranking objective, trained on a single 6 GB consumer GPU. And fourth, the key result: an aggregate score of 2.7235 on the LDCTIQAC 2023 100-image test set - second-best overall, and statistically tied with the top ensemble baseline."

### Slide 5 - Background (1.5 min)

"Two background pillars. The first is the clinical trade-off itself. Reducing photon flux produces a non-stationary Poisson plus Gaussian noise field - the noise distribution depends on local anatomy, so it is not uniform across the image. Streak artefacts propagate along Radon directions - the angles at which the X-ray beams traverse the body. And critically, abdominal soft tissues - the liver at 50 to 65 Hounsfield Units, the spleen at 45 to 65, the kidney cortex at 20 to 40 - share narrow attenuation bands. Quantum noise can easily mask the contrast between these tissues. This is precisely the perceptually demanding regime where image quality assessment matters most.
The second pillar is the manual audit bottleneck. Radiologist quality assessment is labour-intensive, varies between readers, and consumes expensive expert time. Repeat scans cost between two hundred and one thousand US dollars each. Even a one-percent reduction in repeat rate prevents roughly 500 unnecessary scans per year for a 50,000-scan facility - saving cost, saving expert time, and reducing aggregate population radiation dose."

### Slide 6 - Research Objectives (1.5 min)

"Four objectives drove this research.
First, adapt the multi-scale MUSIQ Transformer to single-channel abdominal CT, with mandatory soft-tissue windowing - Level 40 HU, Width 350 HU - aligning the model's input to the perceptual space radiologists use during annotation.
Second, introduce scale-consistency KL regularisation at lambda=0.10, together with a pairwise margin-ranking loss with margin m=0.1, to combat overfitting on the modest 700-image training set.
Third, design a two-stage fine-tuning protocol - five epochs of frozen-encoder warm-up followed by 95 epochs of cosine-annealed full fine-tuning - that remains stable on a single 6 GB consumer GPU.
Fourth, independently reimplement the LDCTIQAC 2023 challenge winner - the Agaldran ensemble of Swin-T and ResNet-50 - under matched hardware and data conditions. This last step gives me a strict, fair, within-paper baseline, rather than relying on numbers from the original leaderboard."

### Slide 7 - Significance and Hypotheses (1 min)

"The significance has four immediate use cases. An accurate no-reference IQA system enables, first, real-time scan flagging at the scanner; second, retrospective auditing of CT archives without consuming radiologist time; third, objective benchmarking of LDCT reconstruction and denoising algorithms; and fourth, PACS integration as an automated quality gate before image distribution. CT-MUSIQ targets all four.
My hypotheses are twofold. First, that multi-scale Transformer attention can capture both global streak artefacts and local high-frequency noise simultaneously - something convolutional networks struggle with because of their limited receptive field. Second, that scale-consistency regularisation prevents per-scale overfitting and improves generalisation on small medical datasets, where data augmentation alone is insufficient."

### Slide 8 - Methodology (1.5 min)

"Two methodological pillars: architecture and training.
The architecture uses a multi-scale ViT-B/32 backbone with approximately 88 million parameters. It ingests two scales simultaneously: 49 patches at 224x224 resolution, and 144 patches at 384x384 - 193 tokens in total flowing through the transformer encoder. Hash-based factored positional encoding injects three pieces of identity per token: which scale it came from, its row, and its column. There are three prediction heads - one global head that produces the final score, plus two per-scale auxiliary heads used only for the KL consistency loss during training.
For training, I use a two-stage protocol. Stage one: 5 epochs of frozen-encoder warm-up at learning rate 3e-4, training only the prediction heads. This stabilises the heads on the new task before touching the pretrained encoder. Stage two: 95 epochs of full fine-tuning with linear warm-up plus cosine annealing from 5e-5 down to 1e-6. I use AdamW, exponential moving average with rho=0.999, AMP mixed precision, and gradient clipping at norm 1. The entire pipeline fits in 5.6 GB of VRAM - an RTX 3060 Laptop, completely off-the-shelf consumer hardware."

### Slide 9 - CT-MUSIQ Architecture (2 min)

"Let me walk through the architecture pipeline in detail.
We start on the left, with a single grayscale CT slice. In the multi-scale patch tokenisation block, the image is resized to two resolutions in parallel: 224x224 and 384x384. The smaller scale is divided into a 7x7 grid of 32x32 patches - that is 49 patches. The larger scale becomes a 12x12 grid - 144 patches. The two are concatenated, giving 193 patches total.
Next is token preparation and encoding. Because ViT-B/32 was pretrained on three-channel ImageNet images, I replicate the grayscale channel three times - identical RGB channels, which lets the pretrained weights work without architectural surgery. A learnable CLS token is prepended for global representation. The hash-based positional encoding module then injects scale, row, and column identity into each token - this is how the transformer knows whether a patch came from the small scale or the large scale, and where it sat spatially within its grid.
These tokens feed into the ViT encoder: 12 transformer blocks, each with multi-head self-attention - 8 heads - followed by an MLP. This is where the key advantage comes in: every patch can attend to every other patch, regardless of distance. That is how we capture long-range streak artefacts that span the whole abdomen, while still resolving local fine-grained noise - something a CNN with limited receptive field cannot do simultaneously.
Finally, the prediction heads produce the quality score. During training, the two per-scale heads also fire - they are used only for the KL consistency loss, which I explain on the next slide."

### Slide 10 - Loss Function (2 min)

"The training objective is a composite of three components.
First, MSE regression loss - the primary objective. Mean squared error between the model's predicted quality score and the ground-truth radiologist score. This penalises large prediction errors quadratically, driving the model toward absolute accuracy.
Second, scale-consistency KL loss, with weight lambda=0.10. This is the most important contribution of this work. For each input, the model produces three score predictions: one global, and one from each per-scale head. I convert each prediction into a soft Gaussian distribution over 20 score bins, then minimise the KL divergence between the per-scale distributions and the global one. The intuition: if the 224-scale prediction wildly disagrees with the 384-scale prediction for the same image, the model is overfitting to one scale's idiosyncrasies. Forcing them to agree is a strong, essentially free regulariser. The ablation confirms this - KL contributes plus 0.034 aggregate, the largest single architectural gain in the entire study, roughly twice the gain of adding the second scale alone.
Third, pairwise ranking loss, with weight lambda=0.5 and margin m=0.1. For every pair of images in a batch, if image A has a higher ground-truth score than image B, the model must predict A higher than B by at least 0.1. This operates on all B^2 ordered pairs in each batch, directly optimising the rank-correlation metrics - PLCC, SROCC, and KROCC - that the LDCTIQAC challenge uses for evaluation. This loss closes the gap between training objective and evaluation objective."

### Slide 11 - Results (2 min)

"Now to the results.
On the held-out LDCTIQAC 2023 test set - 100 abdominal CT slices the model has never seen - CT-MUSIQ achieves an aggregate score of 2.7235. That breaks down as PLCC 0.9498, SROCC 0.9488, KROCC 0.8249. This surpasses every published leaderboard method except the top ensemble.
For the fair within-paper comparison, my reimplemented Agaldran ensemble - Swin-T plus ResNet-50, the LDCTIQAC 2023 winner - reaches 2.7682. That is a 0.045-point gap. Crucially, this gap falls within the 95 percent confidence interval for correlation estimates on a 100-image test set. With a Pearson correlation around 0.95 and n=100, the confidence interval is roughly 0.93 to 0.97. So statistically, CT-MUSIQ and Agaldran are tied - the difference is not significant at the alpha=0.05 level.
Both deep learning models outperform the BRISQUE baseline at 2.1219 by approximately 0.60 aggregate points. That is unambiguous evidence that classical natural-scene-statistics methods are insufficient for calibrated medical CT - they were designed for natural photographs, and they collapse when pixel values represent calibrated tissue density rather than scene reflectance.
The headline of this slide: a single 88-million-parameter Transformer model, trained on a 6 GB consumer GPU, statistically ties the ensemble winner of an international medical imaging challenge."

### Slide 12 - Key Findings (1.5 min)

"Two findings stand out.
First, single-model competitive. CT-MUSIQ surpasses every published leaderboard submission except the top ensemble - validating that multi-scale Transformer adaptation works for medical IQA. The ablation reveals that KL regularisation contributes plus 0.034 aggregate - the largest single architectural gain - roughly twice the plus 0.022 from adding the second scale alone. So the loss design matters more than simply scaling up the architecture.
Second, the ensemble edge. The Swin-T plus ResNet-50 ensemble - aggregate 2.7682, PLCC 0.9608 - reveals complementary strengths between window-based attention and pyramid convolution. With only 700 training images, architectural diversity through ensembling appears to regularise more effectively than KL constraints on a single deeper model. This is an instructive lesson for data-scarce medical imaging: when data is the bottleneck, sometimes two different architectures together beat one carefully regularised architecture."

### Slide 13 - Conclusion (1.5 min)

"Three conclusions to close.
One - multi-scale works. ImageNet-pretrained ViTs do transfer to low-dose CT no-reference IQA, when paired with clinical soft-tissue windowing and a two-stage fine-tuning protocol. No CT-specific pretraining is required. This significantly lowers the barrier to entry for medical IQA research.
Two - KL regularisation is the key insight. Scale-consistency KL loss at lambda=0.10 is the most impactful single component, contributing plus 0.034 aggregate in ablation - roughly twice the gain of the second scale alone. The general lesson: in data-scarce settings, regularising the relationship between auxiliary heads is more valuable than scaling the architecture.
Three - future work. Three concrete directions: multi-scanner cross-institution validation to test generalisation beyond LDCTIQAC; CT-MUSIQ x Agaldran ensembling, which I expect would cross 2.80 aggregate; and VLM-based explainable quality reports for clinical PACS deployment, so each score arrives with a human-readable justification radiologists can audit."

### Slide 14 - Thank You (30 sec)

"Thank you for your attention. I would like to express my deepest gratitude to my adviser, Professor Wenchao Du, for his guidance throughout this research, and to the committee for the opportunity to defend this work today. I welcome your questions and look forward to the discussion."

## Delivery Tips

- Pace: aim for about 130 words per minute. If you finish under 16 minutes, you are rushing; pause more after each slide transition.
- Pointer use: on Slide 9 (architecture pipeline), physically point at each block - Multi-Scale Tokenisation -> Token Encoding -> ViT Encoder -> Heads. This is your most visual slide; treat it like a guided tour.
- Numbers anchor confidence: when you say "2.7235", "0.045-point gap", "+0.034" - say them slowly. Numbers are your strongest evidence; let them land.
- Slide 11 is the climax: do not rush the statistical-tie argument. The fact that you used a 95 percent CI to demonstrate non-significance is sophisticated and shows methodological maturity - committee members will notice.
- Avoid filler words: "um", "like", "basically", "kind of". Pause silently instead.
- First minute matters most: practise the title-slide opening 10 times until it is automatic. If you start strong, the rest flows.

## Critical Q&A Preparation

### Q1: Why did you choose only 2 scales? Why not 3?

"I tested three scales in ablation A6, but adding the third scale at 512x512 increases sequence length to 449 tokens and pushes VRAM beyond 6 GB on my hardware. Within the consumer-GPU constraint that defines this work, 2 scales is the optimal choice. The 30-epoch ablation also showed diminishing returns at scale 3 once KL regularisation is in place."

### Q2: Why KL divergence specifically, not just MSE between the per-scale predictions?

"KL on soft Gaussian distributions is more informative than scalar MSE. It penalises shape mismatch between the predicted score distributions, not just mean differences. With 20 bins and sigma=0.5, neighbouring score bins share probability mass, so the loss is smooth and well-behaved. MSE between scalars would only constrain the mean - it discards the uncertainty information that the soft distribution carries."

### Q3: How does CT-MUSIQ generalise to other CT scanners or institutions?

"Honest answer: I have not tested cross-scanner generalisation, because the LDCTIQAC 2023 dataset is single-source. This is precisely why multi-scanner cross-institution validation is the first item in my future work. I expect a meaningful generalisation gap - that is exactly the limitation I want to address next."

### Q4: The 0.045-point gap to Agaldran - is that really not significant?

"Yes. For PLCC around 0.95 with n=100, the 95 percent confidence interval is approximately 0.93 to 0.97 - that is a +/-0.02 band on each correlation, which propagates to roughly +/-0.06 on aggregate. The 0.045 gap fits comfortably inside this. To detect a 0.045 difference as significant, I would need roughly n=300 test images. With 100, the result is statistically a tie."

### Q5: Why use abdominal soft-tissue windowing at Width 350 HU specifically?

"Width 350 HU centred at Level 40 HU is the clinical standard for abdominal soft-tissue evaluation. It maps the diagnostically relevant HU range - roughly -135 to +215 - to the full visual dynamic range, exactly as a radiologist would view the scan when scoring image quality. By windowing the model's input identically to what the human grader sees, I align the model's perceptual space with the ground-truth annotation process."

### Q6: What is the practical inference cost?

"On the same RTX 3060 Laptop, inference is approximately 80 milliseconds per slice in fp16 mode. For a 200-slice abdominal volume, that is roughly 16 seconds - well within real-time scan-flagging budget at the scanner."

### Q7: Why MUSIQ rather than a Swin or ConvNeXt baseline?

"MUSIQ's multi-scale tokenisation with hash-based positional encoding is purpose-built for variable-resolution input - exactly what we need to capture both global streaks and local noise simultaneously. Swin uses fixed window sizes; ConvNeXt has a fixed receptive field. MUSIQ's design specifically allows different scales to share one transformer encoder, which is the architectural premise the KL consistency loss builds on."
