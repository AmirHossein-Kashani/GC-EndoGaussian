# Response to Reviewers

**Paper #13 — GC-EndoGaussian: Residual-Centered Sparse Control for Editable Endoscopic 4D Gaussian Reconstruction**

**AE-CAI | CARE | OR 2.0 | PRiSM — Camera-Ready Revision**

We sincerely thank all three reviewers for their careful reading and constructive feedback, and we are grateful for the positive assessments of the method's novelty, the rigor of the ablation studies, and the practical value of an editable reconstruction interface. Every weakness raised has been addressed in the revised manuscript, as detailed below. All references to locations name the exact section, paragraph, figure, or caption in the camera-ready layout.

**Nature of this revision.** No previously reported result has changed, and all textual changes are printed in blue in the revised manuscript. One new supporting experiment has been added: an object-removal edit on the public STIR dataset whose revealed content is verified against held-out real frames, presented as a new Results subsection (Section 5.4) with the new Figure 4 and Table 6. We added it because it converts the interpretation of Figure 2 that Reviewers 1 and 5 asked about into a measured claim.

---

## Reviewer 1

We thank the reviewer for the positive assessment, for nominating the paper as an award candidate, and for highlighting the thoroughness of the evaluation and the value of retraining-free editing for real-time surgical applications.

### Comment 1.1 — Blurred results in the videos: failure mode, and how such results are used

> "Some rendered results in the videos still appear unrealistic (like overly blurred). I am wondering if it is considered a failure, or how users perceive or utilize these types of results."

**Response.** This is not a failure but the expected — and, we argue, informative — behavior in regions the camera never observed. Three changes address it, and the third demonstrates it experimentally rather than asserting it.

**(a) The caption of Figure 2 now states what that figure does and does not claim.** We agree the revealed appearance there is not sharp enough to identify sub-surface structure, and the caption says so plainly: that region is under-observed from the effectively single-direction endoscopic viewpoint, and we deliberately do not hallucinate unseen detail. What Figure 2 establishes is *selection and localization* — the edit-magnitude map on the right shows the edit acting on precisely the intended tissue section and nowhere else, so the interface identifies and displaces the structure the user selected rather than disturbing the scene globally. The exposed geometry is likewise a coherent continuation of the depth-anchored reconstruction rather than a view-dependent shell: displacing tissue exposes structure, not holes.

**(b) A new paragraph, "How should an edited view be read?"** (second paragraph of Section 6) tells the user how to utilize such views. The soft region marks where observation ends, while the geometry underneath remains constrained — through stereo-depth supervision fused across the deforming sequence, through the occlusion-holdout result in Section 5.7, and through the zero out-of-support displacement of Figure 3. A reveal therefore doubles as a probe of geometric completeness, and generative inpainting of unobserved appearance is named as future work.

**(c) A new Results subsection completes the claim with ground truth.** Section 5.4, "Revealing occluded structure by object removal," reports the case where the occluded region *is* observed elsewhere in the same recording. On two STIR clips, an object is selected by its own depth extent, its Gaussians are removed at inference, and the exposed region is rendered at a timestamp when it was observed; the result matches the held-out real frame at **25.66 dB** (collection 06/seq23) and **25.12 dB** (06/seq33), against **12.6 dB** and **10.6 dB** for removal without that step (Figure 4, Table 6).

The distinction that removal alone exposes is worth stating precisely, and the subsection does: while the object occludes the region, no pixel of that timestamp depends on the Gaussians behind it, so they receive no photometric gradient there and the optimizer leaves them in an arbitrary configuration. Removal at the edit timestamp therefore exposes an *unsupervised state*, not absent geometry — which is why it scores 10–13 dB. The same Gaussians are well constrained at the timestamps where that surface was filmed (58.9% and 67.6% of the occluded Gaussians are directly observed at some other timestamp), and the reference timestamp is now detected automatically from the depth inside the object's footprint, with no frame nominated by hand. Automatic selection is not a compromise: hand-picked frames give 25.9 and 24.5 dB, so it matches on one clip and improves on the other.

The blur the reviewer observed is therefore precisely an observation limit, not a failure of the edit mechanism: where occluded tissue was observed at any timestamp the edit re-shows it accurately and verifiably; where it never was, the render stays soft rather than hallucinated.

### Comment 1.2 — Clinical relevance and workflow integration

> "The clinical relevance and potential applications are not sufficiently clear. The paper would benefit from more discussion of how the proposed framework would be integrated into clinical workflows."

**Response.** Two additions address this.

**(a)** The second paragraph of Section 1 now gives concrete named scenarios: an instructor replaying a recorded procedure can retract a tissue flap to show trainees the structures behind it, authoring several teaching configurations from a single recording; in post-operative case review, a surgeon can probe what would have been visible had a region been retracted differently; and AR-guidance research needs reconstructions whose configuration can be adjusted toward a planned exposure rather than replayed verbatim.

**(b)** The closing sentences of the second paragraph of Section 6 add the workflow statement: the system consumes recorded stereo endoscopy and tool masks as already produced by existing pipelines, and is interactive after training — 205 frames per second, with edits re-rendering in under 5 ms — so it can be embedded in surgical-video review and training-content tools.

---

## Reviewer 4

We thank the reviewer for the Accept recommendation and for the detailed appraisal of the ablation methodology, the editability metrics, and the practicality of the approach.

### Comment 4.1 — Validation breadth

> "Validation is limited to two EndoNeRF reconstruction sequences and four SuPer tracking trials, reducing confidence in generalization across different anatomies, procedures, and imaging conditions."

**Response.** Partially addressed in the revision and stated explicitly as a limitation. The manuscript adds a third public dataset from a different acquisition setting: two clips from STIR (Schmidt et al., *IEEE Transactions on Medical Imaging*, 2024, cited in the revised bibliography), recorded with a da Vinci Xi stereo endoscope and introduced in the Datasets paragraph of Section 4. The reconstruction and editing pipeline runs on them unchanged and yields the verified object-removal result of Section 5.4. Limitation (iv) in the final paragraph of Section 6 now states that evaluation covers two EndoNeRF scenes, four SuPer trials, and two STIR clips from two acquisition settings. Broader validation across anatomies and procedures remains planned, and the released implementation already supports these dataset formats, which we hope facilitates community replication.

### Comment 4.2 — No user study or clinician assessment

> "The evaluation focuses on geometric editing metrics but lacks any user study or clinician assessment regarding the usefulness, realism, or usability of the editing interface."

**Response.** Agreed, and flagged as limitation (ii) in the final paragraph of Section 6. The Conclusion (Section 7) now names it concretely: future work should quantify perceptual edit plausibility with clinician user studies.

### Comment 4.3 — Not biomechanically accurate

> "While the authors explicitly state that the method is not biomechanically accurate, this limits its applicability for simulation or predictive surgical planning."

**Response.** We agree, and the scoping is intentional and stated throughout — in the Abstract, in Section 2.1, in the caption of Figure 2, in the closing paragraph of Section 5.4, in limitation (i) of Section 6, and in the Conclusion. GC-EndoGaussian evaluates the editing interface on a learned dynamic reconstruction; physics-based simulation, for example in the style of SimEndoGS, can be layered on our control nodes where predictive behavior is required.

---

## Reviewer 5

We thank the reviewer for the constructive comments, which prompted the most substantive improvements in this revision: the concrete clinical motivation, the explicit metric rationale, and the new ground-truth-verified experiment.

### Comment 5.1 — Motivation needs clinical examples

> "The need for editing the gaussian reconstructions is not super well motivated to me. … I think it would be more specific to give some clinical examples of why / what these things are, it would really help understand the point of the paper."

**Response.** Addressed identically to Comment 1.2. The second paragraph of Section 1 replaces the abstract application list the reviewer quotes with three concrete clinical scenarios — authoring teaching configurations from a recorded procedure, counterfactual inspection in post-operative case review, and adjusting a reconstruction toward a planned exposure for AR-guidance research — and the second paragraph of Section 6 adds the workflow-integration sentences.

### Comment 5.2 — Justify the evaluation metrics

> "Justifying the choice of evaluation metrics would help strengthen the results and discussion."

**Response.** The paragraph "Metrics and implementation" in Section 4 now states the rationale explicitly. The two metric families are deliberately disjoint. Reconstruction metrics — PSNR, SSIM, LPIPS, and depth RMSE — follow the standard EndoNeRF/EndoGaussian protocol for comparability and are computed only on unedited held-out frames. The interface metrics measure what is verifiable without ground truth: command following through handle fidelity, spatial confinement through 3D and pixel leakage, geometric sanity through the foldover rate, and interactivity through edit latency. The same paragraph now also names the one edit that *does* admit ground truth — removal of an occluding structure whose hidden region is exposed later in the same recording — and points to Section 5.4, where it is scored directly.

### Comment 5.3 — Figure 2 blur unclear

> "In figure 2, I see a lot of blurring behind the pulled back region. It's unclear to me why this blurring is helpful. I think the caption should make it more clear why this is interesting."

**Response.** We took this comment as a reason to change what Figure 2 claims, not merely to explain the blur. Its caption now states openly that the revealed appearance is not sharp enough to identify sub-surface structure, and redirects the figure's claim to what it genuinely demonstrates: the edit-magnitude panel shows the algorithm acting on exactly the selected tissue section and nowhere else, i.e. correct detection, selection, and localization of the structure the user grabbed.

The appearance claim is then carried by the new Figure 4 and Section 5.4, where the revealed content is verified against held-out real frames at 25.66 and 25.12 dB. The two figures now divide the work: Figure 2 proves the edit lands where the user selected; Figure 4 proves that what appears underneath is genuine wherever the camera observed it. Together they establish that blur appears exactly where, and only where, observation is missing.

### Comment 5.4 — Why this matters

> "I'm still missing the 'why this matters' a bit. Perhaps addressing my previous comments about the motivation would help."

**Response.** Four additions combine to answer this: the concrete clinical uses in the second paragraph of Section 1; the new capability in Section 5.4, which turns "see what is under this structure" into a measured, automatically operated edit rather than an illustration; the framing of a reveal as a probe of geometric completeness in the second paragraph of Section 6; and the workflow sentences that close that paragraph. The residual finding also matters as a design rule, stated in Section 5.6: any group attaching sparse editing handles to a high-fidelity surgical reconstruction should retain a dense per-Gaussian residual, or fidelity, stability, and edit locality all degrade.

### Comment 5.5 — Ground truth for edits, and how RMSE was computed

> "The ground truth for the reconstruction manipulation is also unclear to me. In the submission video, I'm not sure what I should be expecting to see in these blurred regions, how was RMSE computed?"

**Response.** The "Metrics and implementation" paragraph in Section 4 removes the ambiguity. A counterfactual edit has no ground-truth image, so edited renders are never scored with PSNR, SSIM, LPIPS, or depth RMSE; those numbers come exclusively from unedited held-out frames, and edits are evaluated by the self-referential interface metrics of Section 5.3.

Section 5.4 provides the one case in which an edit does admit ground truth. In the STIR clips the occluding structure genuinely moves away later in the recording, so the frame the edit reveals exists in the recorded video; the revealed region is scored against that held-out frame inside the removed object's silhouette, giving 25.66 and 25.12 dB (Table 6), versus 12.6 and 10.6 dB for removal without temporal completion. What to expect in revealed regions is now stated in the caption of Figure 2, in Section 5.4, and in Section 6: accurate re-shown content where the region was observed at some timestamp, and soft, non-hallucinated appearance where it never was.

---

## Summary of Manuscript Changes

1. **Section 1, second paragraph** — the abstract application list is replaced by three concrete clinical scenarios.
2. **Caption of Figure 2** — reframed: the figure now claims selection and localization, states openly that the revealed appearance is not sharp enough to identify sub-surface structure, and forwards the appearance claim to Figure 4.
3. **Section 3.3, final paragraph after Equation 3** — removal is defined as a second edit type, in which the opacity of the selected Gaussians is set to zero at render time so the rasterizer composites whatever the reconstruction holds behind them, with a forward reference to Section 5.4.
4. **Section 4, Datasets** — the two STIR clips are introduced as a third public dataset from a different acquisition setting.
5. **Section 4, "Metrics and implementation"** — metric-family rationale, the explicit statement that no reconstruction metric is computed on edited renders, and the identification of removal as the one edit admitting ground truth.
6. **New Section 5.4, "Revealing occluded structure by object removal," with Figure 4 and Table 6** — object-level selection by depth extent; the explanation that removal alone exposes an unsupervised rather than an absent state; automatic detection of the reference timestamp from the depth inside the object's footprint; revealed regions matching held-out ground truth at 25.66 dB (06/seq23) and 25.12 dB (06/seq33) versus 12.6 and 10.6 dB without temporal completion; automatic timestamps matching or beating hand-picked ones (25.9 and 24.5 dB); nine further edit timestamps spanning 24.6–25.9 dB; and the assumptions of a static endoscope and static occluded surface.
7. **Section 6, second paragraph** — new paragraph "How should an edited view be read?", including the workflow-integration sentences and a closing pointer to Section 5.4; Sections 5.6 and 5.7 carry labels so this paragraph can cross-reference them.
8. **Section 6, final paragraph, limitation (iv)** — dataset coverage updated to two EndoNeRF scenes, four SuPer trials, and two STIR clips from two acquisition settings.
9. **Section 7, Conclusion** — the verified removal result is stated in one sentence, and future work names clinician user studies.
10. **Bibliography** — citations added for the STIR dataset paper and the STIR Challenge 2024 dataset release.

**Supplementary video.** The previously reviewed content is unchanged; a new final chapter presents the object-removal result on both STIR clips — the scene before the edit, the removal, the revealed region shown at the automatically detected timestamp, and a side-by-side comparison against the real frame — with on-screen captions and a closing summary of the measurements.

We thank the reviewers again for their time and for feedback that has clearly strengthened the paper.
