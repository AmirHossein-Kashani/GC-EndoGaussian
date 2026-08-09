# GC-EndoGaussian — Response to Reviewers (camera-ready)

**Venue:** AE-CAI | CARE | OR 2.0 | PRiSM · **Reviews:** 1× Accept, 2× Probably Accept (one best-paper-award-candidate mention) · **Revision:** text-only — no experimental results changed.

We thank all three reviewers for their careful reading and constructive feedback. Every raised weakness is addressed in the revised manuscript. Locations refer to the camera-ready source; section/figure numbers match the submitted layout.

---

## Reviewer 1

### W1 — Blurred results in the videos: is it a failure? How are they used?

> "Some rendered results in the videos still appear unrealistic (like overly blurred). I am wondering if it is considered a failure, or how users perceive or utilize these types of results."

Not a failure, but the expected — and informative — behavior in regions the camera never observed. Two changes explain it:

- **Figure 2 caption** now separates geometry from appearance: the revealed region is under-observed from the effectively single-direction endoscopic viewpoint, so its **appearance** renders soft (we deliberately do not hallucinate unseen detail), while the exposed **geometry** is a coherent continuation of the depth-anchored reconstruction rather than a view-dependent shell — displacing tissue exposes structure, not holes.
- **New Discussion paragraph** (§6, "How should an edited view be read?") tells the user how to utilize such views: the soft region marks where observation ends; the geometry underneath stays constrained (stereo-depth supervision fused across the deforming sequence, occlusion-holdout result of §5.6, zero out-of-support displacement of Fig. 3), so a reveal doubles as a probe of geometric completeness. Generative inpainting of unobserved appearance is named as future work.
### W2 — Clinical relevance / workflow integration unclear

> "The clinical relevance and potential applications are not sufficiently clear. The paper would benefit from more discussion of how the proposed framework would be integrated into clinical workflows."

- **Introduction ¶2** now gives concrete named scenarios: an instructor retracting a flap in a recorded procedure to show trainees the structures behind it (authoring several teaching configurations from one recording); post-operative case review probing what would have been visible under a different retraction; AR-guidance research adjusting a reconstruction toward a planned exposure.
- **Discussion (§6)** adds a workflow-integration sentence: the system consumes recorded stereo endoscopy and tool masks as already produced by existing pipelines, and is interactive after training (205 FPS, sub-5 ms edits), so it can be embedded in surgical-video review and training-content tools.

---

## Reviewer 4

### W1 — Validation breadth

> "Validation is limited to two EndoNeRF reconstruction sequences and four SuPer tracking trials, reducing confidence in generalization across different anatomies, procedures, and imaging conditions."

Acknowledged — stated explicitly as Limitation (iv). Broader validation across anatomies and acquisition settings is planned; the released implementation already supports additional dataset formats, which we hope facilitates community replication.

### W2 — No user study / clinician assessment

> "The evaluation focuses on geometric editing metrics but lacks any user study or clinician assessment regarding the usefulness, realism, or usability of the editing interface."

Agreed and flagged as Limitation (ii); the **Conclusion** now names it concretely: "Future work should quantify perceptual edit plausibility **with clinician user studies**…"

### W3 — Not biomechanically accurate

> "While the authors explicitly state that the method is not biomechanically accurate, this limits its applicability for simulation or predictive surgical planning."

We agree; the scoping is intentional and stated throughout (Abstract, §2.1, Fig. 2 caption, Limitation (i), Conclusion): GC-EndoGaussian evaluates the editing **interface** on a learned dynamic reconstruction; physics-based simulation (e.g., SimEndoGS-style) can be layered on our control nodes where predictive behavior is required.

---

## Reviewer 5

### W1 — Motivation needs clinical examples

> "The need for editing the gaussian reconstructions is not super well motivated to me. … it would be more specific to give some clinical examples of why / what these things are."

Addressed identically to Reviewer 1 W2: **Introduction ¶2** replaces the abstract application list with three concrete clinical scenarios, and §6 adds the workflow-integration sentence.

### W2 — Justify the evaluation metrics

> "justifying the choice of evaluation metrics would help strengthen the results and discussion"

**§4 (Metrics and implementation)** now states the rationale explicitly: the two metric families are deliberately disjoint. Reconstruction metrics (PSNR/SSIM/LPIPS/depth-RMSE) follow the standard EndoNeRF/EndoGaussian protocol for comparability and are computed **only on unedited held-out frames**. The interface metrics measure what is verifiable without ground truth: command-following (handle fidelity), spatial confinement (3D and pixel leakage), geometric sanity (foldover), and interactivity (latency).

### W3 — Figure 2 blur unclear

> "In figure 2, I see a lot of blurring behind the pulled back region. it's unclear to me why this blurring is helpful. I think the caption should make it more clear why this is interesting."

**Figure 2's caption** was rewritten (see Reviewer 1 W1): the blur is under-observed appearance (deliberately not hallucinated); the interesting part is the exposed geometry — a coherent, depth-anchored continuation where a view-overfit reconstruction would show holes.

### W4 — "Why this matters" missing

> "I'm still missing the 'why this matters' a bit. Perhaps addressing my previous comments about the motivation would help."

Three additions combine to answer this: the concrete clinical uses (Intro ¶2), the reveal-as-completeness-probe framing (§6), and the workflow sentence (§6). The residual finding also matters as a design rule: any group attaching sparse editing handles to a high-fidelity surgical reconstruction should retain a dense per-Gaussian residual, or fidelity, stability, and edit locality all degrade (§5.5).

### W5 — Ground truth for edits / how was RMSE computed?

> "The ground truth for the reconstruction manipulation is also unclear to me. In the submission video, I'm not sure what I should be expecting to see in these blurred regions, how was RMSE computed?"

**§4** removes the ambiguity: a counterfactual edit has no ground-truth image, so **edited renders are never scored with PSNR, SSIM, LPIPS, or depth RMSE** — those numbers come exclusively from unedited held-out frames. Edits are evaluated only by the self-referential interface metrics. What to expect in revealed regions is now specified in the Fig. 2 caption and §6 (soft appearance where observation ends; coherent geometry underneath).

---

## Summary of manuscript changes

| # | Location | Change |
|---|----------|--------|
| 1 | Introduction ¶2 | Abstract application list → three concrete clinical scenarios |
| 2 | Figure 2 caption | Geometry-vs-appearance explanation of the revealed region |
| 3 | §4 Metrics | Metric-family rationale; explicit "no reconstruction metric on edited renders" |
| 4 | §6 Discussion | New paragraph "How should an edited view be read?" + workflow integration; §5.6 labeled for cross-reference |
| 5 | Conclusion | Future work names clinician user studies |

**Supplementary video:** unchanged from the reviewed version. The revised Figure 2 caption and the new Discussion paragraph (§6) now tell the viewer how to interpret the soft revealed regions it shows.
