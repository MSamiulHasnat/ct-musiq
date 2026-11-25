# Claims audit for a prospective paper

This is a working evidence table, not a manuscript. See [README.md](README.md)
for dataset provenance, scripts, hashes, and numerical details.

| Claim | Status | Evidence or action |
|---|---|---|
| The compact 448-pixel baseline is stronger than our earlier transformer pipeline on this released test collection | Supported for these exploratory runs | Three seeds give aggregate 2.7200 ± 0.0133 and MAE 0.3010 ± 0.0021, versus 2.4617 ± 0.0179 and 0.4652 ± 0.0210. Multiple pipeline settings differ, and the test has prior exploratory use. |
| The compact model is novel or state of the art | Not established | EfficientNet-B0 at 448 is already used in current author comparisons; ClinReadNet reports 2.7690. Novelty and matched literature comparisons remain unresolved. |
| Preserving floating-point input precision explains the compact model's stronger aggregate | Not supported by the controlled three-seed comparison | Float aggregate 2.7200 ± 0.0133 versus 8-bit control 2.7170 ± 0.0058; mean difference +0.0030. Quantized MAE is lower. This does not establish equivalence or explain the pipeline-level transformer gap. |
| Fewer parameters imply uniformly lower GPU memory | Contradicted by batch-32 benchmark | Compact has 20.4 times fewer parameters but higher measured peak allocation at batch 32. Its measured forward throughput is about 1.35 times higher on one RTX 4090. |
| The archived CT-MUSIQ checkpoint scores 2.4917 on the released 300-image test archive | Reproduced exploratory result | Predictions and metrics were saved; a repeat produced a byte-identical CSV. Test archive provenance should still be checked against the organizers' copy. |
| Related images cross the historical train/test split | Strong image-similarity evidence | All 100 historical test images share an inferred near-duplicate group with historical training images. Original patient/source mappings are not available. |
| A scalar high-frequency baseline has nontrivial predictive power | Supported as exploratory | Its aggregate is 2.4074 on the grouped development validation split and 2.2879 on the released test archive. It was fitted using 790 development training labels. |
| The historical transformer outperforms that scalar baseline on the released test archive | Supported for these saved predictions, with limitations | Aggregate difference is 0.2038. Training partitions differ; group bootstrap is not patient-level uncertainty. |
| Scale-consistency KL improves generalization | Not established | Historical ablations used the old split, A1 failed, and no A6 result was found. Perform matched, repeated ablations. |
| CT-MUSIQ is state of the art | Not supported | Its exploratory test score is below the challenge winner's reported score. Comparisons need matched evaluation and careful provenance. |
| Clinical deployment is ready | Not supported | No external clinical deployment study or independent patient/site validation has been established. |
| The KL term matches observed radiologist rating distributions or measures uncertainty | Contradicted by implementation | It compares fixed-width Gaussian encodings of scalar predictions, computing KL(global || scale). No individual reader-vote distribution is learned. |
| The stored data are brain CT windowed by this code | Contradicted | The challenge reports abdominal CT; the loader reads normalized TIFFs and applies no clinical HU window. |
| The Swin-T + ResNet50 baseline reproduces the challenge winner | Contradicted | The winning submission used Swin and BiT-ResNeXt50 with multiple heads. |

Do not infer that the split alone caused the difference between historical and
released-test results: the sets, model selection, and data distributions also
differ. A controlled comparison is proposed in the experiment protocol.
