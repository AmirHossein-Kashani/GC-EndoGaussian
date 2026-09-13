# GC-EndoGaussian: Response to Reviewers (camera-ready)

Venue: AE-CAI, CARE, OR 2.0, PRiSM. Reviews: one Accept and two Probably Accept, with one best-paper-award-candidate mention.

Nature of this revision: no previously reported result has changed, and all textual changes are printed in blue in the revised manuscript. One new supporting experiment has been added, a ground-truth-verified reveal on the public STIR dataset, shown in the new Figure 6 and discussed in Section 6. We added it because it directly substantiates the interpretation of Figure 2 that Reviewers 1 and 5 asked about.

We thank all three reviewers for their careful reading and constructive feedback. Every raised weakness is addressed in the revised manuscript. All locations below name the exact section, paragraph, figure, or caption in the camera-ready layout.

## Reviewer 1

### Weakness 1: blurred results in the videos, is it a failure, and how are such results used

Reviewer comment: "Some rendered results in the videos still appear unrealistic (like overly blurred). I am wondering if it is considered a failure, or how users perceive or utilize these types of results."

Response: it is not a failure, but the expected and informative behavior in regions the camera never observed. Three changes in the revision explain this, and the third one demonstrates it experimentally.

First, the caption of Figure 2 now separates geometry from appearance. The revealed region is under-observed from the effectively single-direction endoscopic viewpoint, so its appearance renders soft, because we deliberately do not hallucinate unseen detail. The exposed geometry, in contrast, is a coherent continuation of the depth-anchored reconstruction rather than a view-dependent shell. Displacing tissue exposes structure, not holes.

Second, the new paragraph titled "How should an edited view be read?", which is the second paragraph of Section 6 (Discussion and Limitations), tells the user how to utilize such views. The soft region marks where observation ends. The geometry underneath stays constrained, through stereo-depth supervision fused across the deforming sequence, through the occlusion-holdout result in Section 5.6, and through the zero out-of-support displacement shown in Figure 3. A reveal therefore doubles as a probe of geometric completeness. Generative inpainting of unobserved appearance is named in the same paragraph as future work.

Third, a new ground-truth-verified experiment now tests this interpretation rather than merely asserting it. It is presented in the new Figure 6 and in the third paragraph of Section 6. On a public STIR clip in which an organ initially covers the tissue bed and is later dragged aside, removing the organ's Gaussians at an early timestamp re-exposes content the model learned from later frames. In this clip, 67.6 percent of the occluded Gaussians are directly observed at another timestamp, and the revealed region matches the held-out ground-truth frame at 24.8 dB. A second clip from the same dataset behaves identically at 24.1 dB. This is the positive counterpart of Figure 2: where occluded tissue was observed at any timestamp, the edit re-shows it accurately, and where it never was observed, the render stays soft rather than hallucinated. The blur the reviewer observed is therefore precisely an observation limit, not a failure of the edit mechanism.

### Weakness 2: clinical relevance and workflow integration unclear

Reviewer comment: "The clinical relevance and potential applications are not sufficiently clear. The paper would benefit from more discussion of how the proposed framework would be integrated into clinical workflows."

Response: two additions address this.

First, the second paragraph of Section 1 (Introduction) now gives concrete named scenarios: an instructor replaying a recorded procedure can retract a tissue flap to show trainees the structures behind it, authoring several teaching configurations from a single recording; in post-operative case review, a surgeon can probe what would have been visible had a region been retracted differently; and AR-guidance research needs reconstructions whose configuration can be adjusted toward a planned exposure rather than replayed verbatim.

Second, the closing sentences of the second paragraph of Section 6 add the workflow-integration statement: the system consumes recorded stereo endoscopy and tool masks as already produced by existing pipelines, and it is interactive after training, at 205 frames per second with edits re-rendering in under 5 milliseconds, so it can be embedded in surgical-video review and training-content tools.

## Reviewer 4

### Weakness 1: validation breadth

Reviewer comment: "Validation is limited to two EndoNeRF reconstruction sequences and four SuPer tracking trials, reducing confidence in generalization across different anatomies, procedures, and imaging conditions."

Response: this is partially addressed in the revision and stated explicitly as a limitation. The revised manuscript adds a third public dataset from a different acquisition setting: two clips from STIR (Schmidt and colleagues, IEEE Transactions on Medical Imaging 2024, cited in the revised bibliography), recorded with a da Vinci Xi stereo endoscope. The reconstruction and editing pipeline runs on them unchanged and yields the ground-truth-verified reveal shown in the new Figure 6. Limitation four, in the final paragraph of Section 6, now reads that evaluation covers two EndoNeRF scenes, four SuPer trials, and two STIR clips from two acquisition settings. Broader validation across anatomies and procedures remains planned, and the released implementation already supports these dataset formats, which we hope facilitates community replication.

### Weakness 2: no user study or clinician assessment

Reviewer comment: "The evaluation focuses on geometric editing metrics but lacks any user study or clinician assessment regarding the usefulness, realism, or usability of the editing interface."

Response: agreed, and flagged as limitation two in the final paragraph of Section 6. The Conclusion, Section 7, now names it concretely: future work should quantify perceptual edit plausibility with clinician user studies.

### Weakness 3: not biomechanically accurate

Reviewer comment: "While the authors explicitly state that the method is not biomechanically accurate, this limits its applicability for simulation or predictive surgical planning."

Response: we agree, and the scoping is intentional and stated throughout the paper, in the Abstract, in Section 2.1, in the caption of Figure 2, in limitation one of Section 6, and in the Conclusion. GC-EndoGaussian evaluates the editing interface on a learned dynamic reconstruction. Physics-based simulation, for example in the style of SimEndoGS, can be layered on our control nodes where predictive behavior is required.

## Reviewer 5

### Weakness 1: motivation needs clinical examples

Reviewer comment: "The need for editing the gaussian reconstructions is not super well motivated to me. It would be more specific to give some clinical examples of why and what these things are."

Response: addressed identically to Reviewer 1, Weakness 2. The second paragraph of Section 1 replaces the abstract application list with three concrete clinical scenarios, and the second paragraph of Section 6 adds the workflow-integration sentences.

### Weakness 2: justify the evaluation metrics

Reviewer comment: "Justifying the choice of evaluation metrics would help strengthen the results and discussion."

Response: the paragraph titled "Metrics and implementation" in Section 4 now states the rationale explicitly. The two metric families are deliberately disjoint. Reconstruction metrics, namely PSNR, SSIM, LPIPS, and depth RMSE, follow the standard EndoNeRF and EndoGaussian protocol for comparability and are computed only on unedited held-out frames. The interface metrics measure what is verifiable without ground truth: command following through handle fidelity, spatial confinement through 3D and pixel leakage, geometric sanity through the foldover rate, and interactivity through edit latency.

### Weakness 3: Figure 2 blur unclear

Reviewer comment: "In figure 2, I see a lot of blurring behind the pulled back region. It is unclear to me why this blurring is helpful. I think the caption should make it more clear why this is interesting."

Response: the caption of Figure 2 was rewritten, as described under Reviewer 1, Weakness 1. The blur is under-observed appearance, deliberately not hallucinated, and the interesting part is the exposed geometry, a coherent depth-anchored continuation where a view-overfit reconstruction would show holes. The new Figure 6 completes the argument with a measured case: on a clip where the hidden region is observed at other timestamps, the same edit machinery reveals it and ground truth confirms it at 24.8 dB. This establishes that blur appears exactly where, and only where, observation is missing.

### Weakness 4: why this matters

Reviewer comment: "I am still missing the why this matters a bit. Perhaps addressing my previous comments about the motivation would help."

Response: three additions combine to answer this: the concrete clinical uses in the second paragraph of Section 1, the framing of a reveal as a probe of geometric completeness in the second and third paragraphs of Section 6, and the workflow sentences at the end of the second paragraph of Section 6. The residual finding also matters as a design rule, stated in Section 5.5: any group attaching sparse editing handles to a high-fidelity surgical reconstruction should retain a dense per-Gaussian residual, or fidelity, stability, and edit locality all degrade.

### Weakness 5: ground truth for edits, and how RMSE was computed

Reviewer comment: "The ground truth for the reconstruction manipulation is also unclear to me. In the submission video, I am not sure what I should be expecting to see in these blurred regions, how was RMSE computed?"

Response: the paragraph titled "Metrics and implementation" in Section 4 removes the ambiguity. A counterfactual edit has no ground-truth image, so edited renders are never scored with PSNR, SSIM, LPIPS, or depth RMSE. Those numbers come exclusively from unedited held-out frames, and edits are evaluated by the self-referential interface metrics of Section 5.3.

The revision additionally provides the one case in which an edit does admit ground truth, shown in the new Figure 6 and discussed in the third paragraph of Section 6. In that STIR clip the occluder genuinely moves away later in the recording, so the frame that the edit reveals exists in the recorded video. There, the revealed region scores 24.8 dB against that held-out frame. What to expect in revealed regions is now specified in the caption of Figure 2 and in Section 6: accurate re-shown content where the region was observed at some timestamp, and soft, non-hallucinated appearance where it never was.

## Summary of manuscript changes

1. Section 1, Introduction, second paragraph: the abstract application list is replaced by three concrete clinical scenarios.
2. Caption of Figure 2: new geometry-versus-appearance explanation of the revealed region.
3. Section 3.3, final paragraph after Equation 4: removal is defined as a second edit type, in which the opacity of the selected Gaussians is set to zero at render time, with a forward reference to Figure 6.
4. Section 4, paragraph "Metrics and implementation": metric-family rationale, with the explicit statement that no reconstruction metric is computed on edited renders.
5. Section 6, second paragraph: new paragraph titled "How should an edited view be read?", including the workflow-integration sentences; Section 5.6 carries a label so this paragraph can cross-reference it.
6. Section 6, third paragraph, and the new Figure 6: new supporting experiment, a ground-truth-verified reveal on two STIR clips, in which 67.6 percent of the occluded Gaussians are observed at another timestamp and the revealed region matches held-out ground truth at 24.8 dB on the first clip and 24.1 dB on the second.
7. Section 6, final paragraph, limitation four: dataset coverage updated to two EndoNeRF scenes, four SuPer trials, and two STIR clips from two acquisition settings.
8. Section 7, Conclusion: future work names clinician user studies.
9. Bibliography: citations added for the STIR dataset paper and the STIR Challenge 2024 dataset release.

Supplementary video: extended with a final chapter presenting the STIR verified reveal, showing the scene before the edit, the removal, the revealed content, and the ground-truth comparison, with on-screen captions guiding the viewer. The previously reviewed content of the video is unchanged. The revised caption of Figure 2 and Section 6 tell the viewer how to interpret the soft revealed regions.
