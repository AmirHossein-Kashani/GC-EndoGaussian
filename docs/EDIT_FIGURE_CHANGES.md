# Figure 2 (edit / editability) — what changed

This note documents the new **generality-oriented edit figure** added for
GC-EndoGaussian, replacing the old drag-to-edit Figure 2. Nothing existing was
overwritten — all outputs are new files.

---

## 1. Why the old figure was weak

The original Figure 2 is the paper's *only* visual proof of the headline
"editability" claim, but it undersold it:

| Panel | What it actually was | Problem |
|---|---|---|
| `edit_before.png` | `pulling_match/edit_gentle/before.png` | fine |
| `edit_after.png` | `edit_gentle/after_0.png` — the **gentlest** sweep step (mag `0.06`) | before/after look nearly identical |
| `edit_diff.png` | a custom fire-on-grey composite | noisy blob, no colorbar, reads as "error" not "edit" |

The edit was real but rendered almost invisible, and the diff panel looked like
noise. (The stronger existing sweep at mag `0.30` over-drags and smears the whole
surface into artifacts, so it wasn't usable either.)

---

## 2. What the new figure shows

A **generality** figure: **2 scenes × 2 drag directions each**, demonstrating that
the control-node handle produces *local, directionally-controllable* edits — not a
one-off.

```
              before            drag ↓            lateral drag       edit magnitude
 pulling  [handle outline]  [↓ arrow, pulled]  [→ arrow, shifted]   [heatmap]      ┐
 cutting  [handle outline]  [↓ arrow, pulled]  [← arrow, shifted]   [heatmap]      ┘ + colorbar
```

Each edit panel bakes in:
- a **dashed green outline** = the grabbed control-node region (the "handle"),
- a **cyan arrow** = the applied inference-time node translation,
- the last column = per-pixel edit magnitude `‖I_after − I_before‖` on a
  perceptually-uniform `inferno` colormap, which stays localized to the manipulated
  region. A shared colorbar covers both scenes.

All edits are applied at inference time on the already-trained `match` models and
re-rendered from the same view **without retraining**.

---

## 3. New image files (in [docs/figures/](figures/))

Used by the LaTeX (each is a self-contained 640×512 panel with overlays baked in):

| pulling | cutting | shared |
|---|---|---|
| `edit_gen_pulling_before.png` | `edit_gen_cutting_before.png` | `edit_gen_colorbar.png` |
| `edit_gen_pulling_down.png` | `edit_gen_cutting_down.png` | |
| `edit_gen_pulling_right.png` | `edit_gen_cutting_left.png` | |
| `edit_gen_pulling_diff.png` | `edit_gen_cutting_diff.png` | |

Alternate direction panels (ready if you want to swap one in):
`edit_gen_pulling_up.png`, `edit_gen_cutting_right.png`.

The original `edit_before.png` / `edit_after.png` / `edit_diff.png` are **untouched**.

---

## 4. Design decisions

- **Sweet-spot magnitude.** The old mag `0.06` was invisible and `0.30` smeared. A
  magnitude sweep (`0.08 → 0.20` of scene extent) was rendered; the clean-but-clearly
  visible point is **~0.08 for pulling** and **~0.12–0.16 for cutting**.
- **Local region.** Selection radius (`radius_frac`) is tuned so the edit is *local*:
  pulling grabs 406/2048 nodes (`radius_frac 0.14`); cutting needed a tighter
  `radius_frac 0.06` (188 nodes) because its node cloud is denser — the first
  attempt at `0.14` grabbed ~80% of nodes and read as a whole-scene pan.
- **Directions.** Each scene shows one vertical (↓) and one lateral (→ / ←) drag to
  cover the 2-D control space. Directions/magnitudes are per-panel configurable.
- **Honest framing.** The caption still states the edits illustrate locality and
  controllability only and **do not establish biomechanical validity**.

---

## 5. Updated LaTeX

Drop-in replacement for the old `\begin{figure}...\end{figure}` edit block is in
[docs/figure2_generality.tex](figure2_generality.tex). It uses only `graphicx`
(already loaded by the Wiley class) and switches the figure from single-column
`figure` to full-width `figure*` (needed for the 2×4 grid).

---

## 6. How to regenerate

GPU work runs on a SLURM cluster (login node has no GPU) — rendering-only jobs,
a couple of minutes each once scheduled.

1. **Render the drag sweep** (already-trained checkpoints, exports node projections):
   ```
   sbatch run_edit_figure.bash     # pulling + cutting, 4 dirs × mag sweep
   sbatch run_edit_cutting.bash    # cutting with tighter local region
   ```
   Scripts: [tools/edit_figure.py](../tools/edit_figure.py).

2. **Compose the panels** (CPU-only, uses the repo `.venv`):
   ```
   .venv/bin/python tools/compose_panels.py \
     --dir output/endonerf/pulling_match/edit_fig --mag 0.08 \
     --dirs "+y:down,-y:up,+x:right" --diff_dir "+y:0.08" \
     --prefix docs/figures/edit_gen_pulling --dmax 0.35 \
     --colorbar_out docs/figures/edit_gen_colorbar.png

   .venv/bin/python tools/compose_panels.py \
     --dir output/endonerf/cutting_match/edit_fig_r06 --mag 0.12 \
     --dirs "+y:down,-x:left:0.16,+x:right" --diff_dir "+y:0.12" \
     --prefix docs/figures/edit_gen_cutting --dmax 0.35
   ```
   Script: [tools/compose_panels.py](../tools/compose_panels.py). `--dirs` accepts
   `dir:name[:mag]` so magnitude can be set per panel; `--dmax` fixes the shared
   colorbar scale.
