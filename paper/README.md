# Working manuscript

`main.tex` is a research draft, not an approved journal submission. The
27-model benchmark and external GPU evaluation are complete. The draft
includes all nine arms, three seeds, anatomy-specific results and paired
source-slice resampling within two phantoms. Its target journal and
interpretation require author review. Author information, contributions,
funding/conflict statements and the applicable institutional ethics
determination require author confirmation.

The current question is whether quality-conditioned degradation ranking
changes smoothing responses while preserving useful benchmark accuracy and
agreement with independent expert ratings. EfficientNet, ordinal losses and
smoothing-bias analysis have prior art; the manuscript does not claim that
these individual components are new.

From the repository root, regenerate the tables and figure with:

These commands require the retained per-image prediction files. A fresh
clone must first reproduce the runs or obtain the documented local evidence
export; the generated tables and aggregate results are already versioned.

```text
python -m research_audit.bootstrap_external_models --results research_audit/results/external_gpu --require-complete
python -m research_audit.summarize_robustness --require-complete
```

Then compile from this directory using a LaTeX installation or Tectonic:

```text
tectonic --keep-logs main.tex
```

The source includes a manual bibliography so no separate BibTeX step is
required. The first Tectonic run downloads its standard TeX support files.
The Codex built-in compiler timed out during preparation; the portable
Tectonic build was used for the local preview.

`results_generated.tex` and `images/robustness_tradeoff.*` are generated from
saved experiment metrics. Source snapshots, model/data hashes and the
exploratory development history are recorded under `research_audit/`.
External labels are used only for evaluation, never calibration or model
selection. Initial external runs with a rotated NIfTI array presentation
are preserved separately and excluded from the final transfer table.

Potential journal fit remains contingent on the finished evidence. The
author requires a subscription or other route with no APC and a hard total
budget of US$120. No publication or review-time guarantee is implied.

Journal options checked on 25 November 2026:

- [Journal of Imaging Informatics in Medicine](https://link.springer.com/journal/10278/aims-and-scope)
  explicitly covers quantitative imaging, quality assurance and AI evaluation.
  Its [publishing options](https://link.springer.com/journal/10278/how-to-publish-with-us)
  include subscription publication; Springer says this route has no APC, while
  noting that some journals can impose separate page/colour charges. No such
  charge was found in this journal's current guidelines; total costs still need
  verification before submission. Its emphasis on validated applications makes
  the two-phantom limitation material. It is a candidate, not a guaranteed fit.
- [Computerized Medical Imaging and Graphics](https://shop.elsevier.com/journals/computerized-medical-imaging-and-graphics/0895-6111)
  is a higher-risk candidate because it emphasizes methodological advances.
  The generic degradation-ranking idea has prior art. Its specific current
  author guide could not be retrieved (403), so zero total publication charges
  have not been certified here.

The JIIM guidelines request a 150–250 word abstract, 4–6 keywords, separate
identified and blinded versions, and accept LaTeX for mathematical content.
LaTeX upload files must be flattened rather than placed in subfolders. These
formatting requirements do not remove the need to confirm authorship,
affiliation, correspondence and institutional declarations.


The completed evidence archive SHA-256 is
`b455720f959746baad8d1a42ec1ba8b60622b7cd6ad90c092c9d2a01bc9d3fd8`.
`research_audit/results/completed_study_manifest.json` records all 27
checkpoint hashes, configurations and the observed environment. Raw images,
weights and per-image prediction CSVs remain local or on the research server;
aggregate results and training traces are versioned. The final external
analysis is under `research_audit/results/external_gpu/`; CPU runs and initial
stored-array runs are retained separately as checks and experimental history.

Verification: all nine unit tests pass, all 27 models have 300-image LDCT
and 1,666-image external outputs, checkpoint hashes agree between endpoints,
and the generic evaluation engine gives exactly matching scores on all
1,666 reference-checkpoint images. The PDF was compiled with portable
Tectonic and inspected as rendered pages. This remains a research draft;
no journal submission has been made.
