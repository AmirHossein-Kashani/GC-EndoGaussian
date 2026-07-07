# Cutting figure — editable tissue retraction ("see behind the tissue")

Final choice for the edit figure: **cutting scene only**, 3 clean panels —
**before**, **lateral drag (retraction)**, and the **heatmap of that lateral drag**.
Overlays: green dashed circle kept, **no arrow**, **no text titles** on the images.

Use-case framing: dragging the circled tissue sideways **retracts it to reveal the
region behind**, improving the surgeon's visibility — done at inference time, no
retraining.

**Preview:**

![cutting reveal](figures/edit_reveal_cutting_preview.png)

*(left: before · middle: after lateral retraction · right: per-pixel edit magnitude)*

---

## Files to upload to Overleaf `figures/`

- `edit_reveal_cutting_before.png` — before, green circle only
- `edit_reveal_cutting_lateral.png` — after lateral retraction, green circle only
- `edit_reveal_cutting_diff.png` — edit-magnitude heatmap of that same lateral drag

(All new files. Your existing `edit_before/after/diff.png` are untouched.)

---

## Partial LaTeX update (keeps your 3-panel `figure`, no titles)

Replace only the three `\includegraphics` lines (and, since you want the titles
removed, drop the `\\[2pt]{\footnotesize ...}` sub-labels). Then replace the caption.

```latex
\begin{figure}[t]
\centering
\begin{minipage}[t]{0.32\columnwidth}\centering
\includegraphics[width=\linewidth,height=85pt]{figures/edit_reveal_cutting_before.png}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\columnwidth}\centering
\includegraphics[width=\linewidth,height=85pt]{figures/edit_reveal_cutting_lateral.png}
\end{minipage}\hfill
\begin{minipage}[t]{0.32\columnwidth}\centering
\includegraphics[width=\linewidth,height=85pt]{figures/edit_reveal_cutting_diff.png}
\end{minipage}
\caption{Editable tissue retraction on \texttt{cutting\_tissues\_twice}. A local group
of control nodes (dashed outline) is dragged laterally at inference time to retract
the overlying tissue and expose the region behind it, without retraining. Left:
before. Middle: after lateral retraction. Right: per-pixel edit magnitude
$\lVert I_{\mathrm{after}}-I_{\mathrm{before}}\rVert$, localized to the retracted
region. The edit illustrates controllable visibility; it does not model tissue
mechanics or establish biomechanical validity.\label{fig:edit}}
\end{figure}
```

Notes:
- The `height=85pt` keeps the same footprint as your current figure (it slightly
  vertical-stretches the 640×512 panels, exactly like your existing block). For the
  true aspect ratio, drop `,height=85pt`.
- The green dashed circle marks the grabbed control-node region; there is no arrow and
  no text baked into the images.

---

## Regenerate (if you want to tweak)

```
.venv/bin/python tools/compose_panels.py \
  --dir output/endonerf/cutting_match/edit_fig_r06 --mag 0.16 \
  --dirs="-x:lateral:0.16" --diff_dir="-x:0.16" \
  --no_arrow --circle_on_diff \
  --prefix docs/figures/edit_reveal_cutting --dmax 0.35
```

- `--no_arrow` → green circle only. Drop `--circle_on_diff` to remove the circle from
  the heatmap. Change `-x`→`+x` for the opposite lateral direction, or the `0.16`
  magnitude for a bigger/smaller retraction (renders `0.08/0.12/0.16` exist).
