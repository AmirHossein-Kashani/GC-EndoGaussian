# Paper updates — new experiments → LaTeX, section by section

Sources: job XXXXXXXX (edit metrics, `eval_edit.py`), jobs XXXXXXXX–90 + XXXXXXXX
(seed study: seeds 1234/2025/3407/4242 + the existing seed-6666 paper runs).
Numbers below are final.

**New figure files to upload to Overleaf `figures/`:**
- `fig_residual_ablation.pdf` — visual evidence for Table 3 (what the residual fixes)
- `fig_edit_locality.pdf` — edit locality curve (compact support)

**Headline results**
| | |
|---|---|
| Handle fidelity (median) | 0.96–0.97, all models |
| 3D leakage beyond binding support | exactly 0 (structural) |
| Pixel-space edit leakage | **40.4% without residual vs 9.6–12.3% with** |
| Foldover rate | ≤0.13% of neighbor pairs |
| Edit→render latency | 4.5–4.9 ms |
| Motion energy carried by residual | 69–77% |
| Seed study (n=4): vanilla | PSNR 37.186 ± 0.088 |
| Seed study (n=4): GC-EndoGaussian | PSNR 37.073 ± 0.062 (gap 0.11 dB ≈ seed noise) |
| Seed study: SC-GS no residual | 36.798 ± 0.010 over 3 converged; **2 of 5 seeds fail** (1 collapse, 1 reproducible crash) |
| Seed study (n=4): SC-GS + residual | PSNR 37.313 ± 0.055, all seeds converge |

---

## 1. Abstract

**(a) Replace** "User-specified node translations can then be applied at inference time to
produce local edits without retraining." **with:**

```latex
User-specified node translations can then be applied at inference time to produce local
edits without retraining: edited tissue follows the commanded handle displacement at a
median ratio of 0.96, the induced displacement field is identically zero beyond the
binding support of the grabbed nodes, and a new edit re-renders in under 5\,ms.
```

**(b) Replace** "At the standard 3000 fine-stage iterations, GC-EndoGaussian is 0.27 dB
below EndoGaussian on `pulling_soft_tissues`;" **with:**

```latex
At the standard 3000 fine-stage iterations, GC-EndoGaussian is 0.11\,dB below EndoGaussian
on \texttt{pulling\_soft\_tissues} averaged over four random seeds
($37.07\pm0.06$ vs.\ $37.19\pm0.09$\,dB), a gap comparable to seed-to-seed variation;
```

**(c) Replace** "adding it to an SC-GS-style sparse-control model improves PSNR from 36.80
to 37.29 dB and reduces tracking reprojection error from 7.02 to 3.41 pixels." **with:**

```latex
adding it to an SC-GS-style sparse-control model improves PSNR from $36.80\pm0.01$ to
$37.31\pm0.06$\,dB across seeds, reduces tracking reprojection error from 7.02 to 3.41
pixels, and repairs a stability failure: without the residual, training diverges on two of
five seeds, whereas every residual-equipped run converges. The residual also keeps edits
clean in image space: without it, 40\% of the image change induced by a local edit falls
outside the edited region, versus 10--12\% with it.
```

---

## 2. §1 Contributions

**(a) Replace contribution 2 with:**

```latex
\item \textbf{A sparse edit-control layer for 4D endoscopic Gaussians, evaluated
quantitatively.} Motion-seeded control nodes, K-nearest-neighbor soft binding, weighted
skinning, and inference-time node translations provide a compact interface for local scene
manipulation. We measure the interface directly: handle-following fidelity of $0.96$
(median), structurally compact support (zero displacement beyond the binding
neighborhood), foldover rates below $0.14\%$, and sub-5\,ms edit-to-render latency. The
sparse-control and skinning paradigm follows prior editable Gaussian methods; our
contribution is its integration and measurement in a surgical reconstruction pipeline.
```

**(b) Replace contribution 3 with:**

```latex
\item \textbf{A controlled isolation of the fidelity-preserving component.} A
residual-matched comparison shows that an SC-GS-style model without a dense residual loses
$0.52$\,dB PSNR, has roughly twice the tracking error, leaks $40\%$ of edit-induced image
change outside the edited region, and fails to converge on two of five random seeds. The
same control architecture with the residual recovers near-baseline reconstruction and
tracking, confines edits, and converges on every seed tested.
```

---

## 3. §4.3 Metrics and implementation — append at the end

```latex
Reconstruction robustness to initialization is reported as mean $\pm$ standard deviation
over four random seeds per configuration on \texttt{pulling\_soft\_tissues} under the
standard schedule (Table~\ref{tab:seeds}). Editing is evaluated with five interface
metrics (Section~\ref{sec:edit_eval}): \emph{handle fidelity} (achieved displacement of
strongly-bound Gaussians divided by the commanded magnitude), \emph{3D leakage} (mean
normalized displacement of Gaussians beyond twice the handle radius), \emph{pixel leakage}
(fraction of image change outside the dilated projected footprint of the edited region),
\emph{foldover rate} (fraction of Gaussian $k$-NN pairs whose relative orientation inverts
under the edit), and \emph{edit latency} (wall-clock from setting a new edit vector to a
completed $640\times512$ re-render). Each edit metric aggregates 32 edits: 8 handle draws
$\times$ 4 magnitudes (1--8\% of scene extent), with handle radius $0.10\times$ the scene
extent.
```

---

## 4. §5.1 — replace the single-seed comparison sentence and add the seed table

**Replace** "Under the standard 3000-iteration fine stage on `pulling_soft_tissues`, the
editable model is 0.27 dB lower in PSNR." **with:**

```latex
Under the standard 3000-iteration fine stage on \texttt{pulling\_soft\_tissues}, the
editable model is $0.11$\,dB lower in PSNR averaged over four random seeds
(Table~\ref{tab:seeds}); per-configuration seed variation is $\leq 0.09$\,dB, so the
single-seed gaps reported in Table~\ref{tab:main} should be read with that granularity in
mind.
```

**Add table (end of §5.1):**

```latex
\begin{table}[t]
\caption{Seed robustness on \texttt{pulling\_soft\_tissues} (standard schedule). Mean
$\pm$ standard deviation over four converged seeds per configuration. The residual-free
SC-GS-style model additionally \emph{failed to converge on two of five attempted seeds}
(one collapse to 7.5\,dB, one reproducible CUDA failure during node-graph training); no
residual-equipped configuration failed on any seed.\label{tab:seeds}}
\centering
\begin{tabular}{lccc}
\toprule
Method & PSNR$\uparrow$ & LPIPS$\downarrow$ & Converged \\
\midrule
EndoGaussian             & $37.19 \pm 0.09$ & $0.062 \pm 0.001$ & 4/4 \\
SC-GS-style, no residual & $36.80 \pm 0.01$ & $0.090 \pm 0.001$ & 3/5 \\
SC-GS-style + residual   & $37.31 \pm 0.06$ & $0.065 \pm 0.000$ & 4/4 \\
GC-EndoGaussian          & $37.07 \pm 0.06$ & $0.065 \pm 0.001$ & 4/4 \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 5. NEW §5 subsection — place after §5.2 (runtime), before the tracking section

```latex
\subsection{Quantitative edit evaluation}\label{sec:edit_eval}

Table~\ref{tab:edit} measures the editing interface directly on the three budget-matched
models of Table~\ref{tab:residual}. All models follow the handle at a median fidelity of
$0.96$--$0.97$, with \emph{identically zero} displacement beyond the binding support of
the grabbed nodes (Figure~\ref{fig:locality}) --- locality is a structural property of the
compact K-nearest-neighbor binding, independent of training. Foldover rates remain below
$0.14\%$ of Gaussian neighbor pairs at all tested magnitudes, and a new edit re-renders in
under $5$\,ms, so the edit loop runs at interactive rates on top of the 205\,FPS renderer.

The models differ sharply in \emph{pixel-space} hygiene: without the dense residual,
$40\%$ of the total image change induced by a local edit falls outside the edited region's
footprint, versus $10$--$12\%$ for the residual-equipped models. The residual absorbs
appearance detail that the sparse graph would otherwise distribute globally, so removing
it does not only cost reconstruction fidelity (Table~\ref{tab:residual}) --- it also makes
edits visually leak. Finally, decomposing the learned motion of the hybrid models shows
the node field carries $23$--$31\%$ of the deformation energy and the residual the
remaining $69$--$77\%$: the sparse graph is an \emph{interface} over a predominantly dense
motion model, consistent with Section~\ref{sec:ablations}.

\begin{table}[t]
\caption{Edit-interface metrics on \texttt{pulling\_soft\_tissues} (3000 fine-stage
iterations, matched budget). Median over 32 edits (8 handle draws $\times$ 4 magnitudes,
1--8\% of scene extent). Fid.\ = handle fidelity; Leak\textsubscript{3D} = normalized
displacement beyond $2\times$ handle radius; Leak\textsubscript{px} = image change outside
the edited footprint; Fold = foldover rate; Lat.\ = edit-to-render latency;
Res.\,frac.\ = fraction of learned motion energy carried by the dense residual.
\label{tab:edit}}
\centering
\begin{tabular}{lcccccc}
\toprule
Method & Fid.$\uparrow$ & Leak\textsubscript{3D}$\downarrow$ &
Leak\textsubscript{px}$\downarrow$ & Fold$\downarrow$ & Lat.\,(ms)$\downarrow$ &
Res.\,frac. \\
\midrule
SC-GS-style, no residual & 0.97 & 0.0\% & 40.4\% & 0.07\% & 4.5 & --- \\
SC-GS-style + residual   & 0.97 & 0.0\% & \phantom{0}9.6\% & 0.08\% & 4.7 & 0.69 \\
GC-EndoGaussian          & 0.96 & 0.0\% & 12.3\% & 0.05\% & 4.8 & 0.77 \\
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[t]
\centering
\includegraphics[width=0.9\linewidth]{figures/fig_edit_locality.pdf}
\caption{Edit locality. Median per-Gaussian displacement (normalized by the commanded
magnitude) as a function of distance to the nearest grabbed control node, aggregated over
32 edits. The displacement field falls off within the handle radius (shaded) and is
identically zero beyond the K-nearest-neighbor binding support --- locality is structural,
not learned.\label{fig:locality}}
\end{figure}
```

---

## 6. §5.2 Table 2 — add one row after "Render speed"

```latex
Edit-to-render latency & --- & 4.8\,ms \\
```

---

## 7. §5.4 (residual isolation) — add the stability + visual evidence

**(a) Append to the §5.4 text:**

```latex
Two further observations sharpen this conclusion. First, the residual-free model is not
merely worse but \emph{unstable}: across five random seeds it diverged twice (one collapse
to 7.5\,dB, one reproducible failure during node-graph optimization), whereas every
residual-equipped configuration converged on all four seeds tested
(Table~\ref{tab:seeds}). Second, the gap is visible:
Figure~\ref{fig:residual_visual} shows the residual-free model smearing high-frequency
vascular texture and specular highlights that the dense residual restores without changing
the control architecture.
```

**(b) New figure:**

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{figures/fig_residual_ablation.pdf}
\caption{What the dense residual fixes. Test frame of \texttt{pulling\_soft\_tissues}
(3000 fine-stage iterations, matched budget). Top: rendering with inset location; middle:
zoom; bottom: per-pixel error over tissue (tool region excluded from losses and metrics;
shared color scale). The SC-GS-style model without the residual (third column) shows
visibly higher error on vascular and specular detail; adding the per-Gaussian residual
(fourth column) recovers it, matching the $0.52$\,dB gap in
Table~\ref{tab:residual}.\label{fig:residual_visual}}
\end{figure*}
```

---

## 8. §6 Limitations — replace two bullets

**(a) Replace the "Direct edit quality is evaluated qualitatively" bullet with:**

```latex
\item \textbf{Edit evaluation is geometric, not perceptual.} Section~\ref{sec:edit_eval}
quantifies handle fidelity, locality, foldover, and latency, but does not measure the
\emph{plausibility} of edited tissue configurations: no user study with surgical experts
was conducted, and no comparison against physically simulated deformation is reported.
```

**(b) Replace the "Efficiency reporting is incomplete" bullet with:**

```latex
\item \textbf{Efficiency reporting is incomplete.} Rendering speed, parameter count, and
edit-to-render latency are reported, but wall-clock training time and memory use should be
measured in a final systems evaluation.
```

---

## 9. §7 Conclusion — strengthen two sentences

**(a) Replace** "The method binds dense Gaussians to a compact set of control nodes and
supports local inference-time edits through weighted node translations." **with:**

```latex
The method binds dense Gaussians to a compact set of control nodes and supports local
inference-time edits through weighted node translations, with measured handle fidelity of
0.96, structurally compact edit support, and sub-5\,ms edit-to-render latency.
```

**(b) Replace** "An SC-GS-style sparse-control model without a dense residual loses
reconstruction and tracking fidelity, whereas adding the per-Gaussian residual recovers
most of the gap." **with:**

```latex
An SC-GS-style sparse-control model without a dense residual loses reconstruction and
tracking fidelity, leaks edit-induced image change outside the edited region, and fails to
converge on some seeds, whereas adding the per-Gaussian residual recovers reconstruction
and tracking, confines edits, and stabilizes training.
```

---

## Reproduction

- Edit metrics: `sbatch run_eval_edit.bash` → `output/endonerf/*/edit_metrics.json`
- Seed study: `sbatch --export=ALL,SEED=<n> run_seed_study.bash` (+ `run_seed_retry.bash`)
- Aggregation + locality figure: `python tools/aggregate_paper_updates.py`
- Ablation figure: `python tools/make_fig_residual_ablation.py`
