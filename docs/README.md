# GC-EndoGaussian — documentation

Three documents, one role each. All are current and mutually consistent (decontaminated
controllability results; the per-Gaussian residual identified as the key ingredient). Numbers are
verified against `results_archive/`.

| Document | Role | Read it for |
|---|---|---|
| **[paper.md](paper.md)** | The **paper** (submission draft) | The polished write-up: abstract, method, experiments, honest findings, references. |
| **[IMPLEMENTATION.md](IMPLEMENTATION.md)** | The **implementation** & reproduction guide | Exact configs, hyperparameters, data pipeline, train/eval commands, and how to reproduce every number and figure. |
| **[RESEARCH_OVERVIEW.md](RESEARCH_OVERVIEW.md)** | The **overview / key techniques** ("the tricks") | A single readable account: what we built, the innovation, the key techniques (motion-weighted seeding, the *match* recipe, the residual, the decontamination), and an honest publication assessment. |

Figures and demo videos live in [figures/](figures/) and are embedded by `paper.md`.

### The two honest headlines (shared by all three docs)
1. **Cost-free editability.** The control layer adds editing at reconstruction/tracking parity (within
   ~0.15 dB PSNR, 205 FPS, +0.07% params); a residual-matched ablation shows the **per-Gaussian residual**
   is the key ingredient (not the GNN or the specific recipe), so this is a practical recipe, not a
   superiority over SC-GS.
2. **A decontaminated controllability evaluation.** An uncorrected control-from-tracks metric is confounded by the
   model's own reconstruction; once decontaminated, learned sparse control (ours and a retrained SC-GS)
   does **not** beat classical interpolation — reported openly as a methodological caution.
