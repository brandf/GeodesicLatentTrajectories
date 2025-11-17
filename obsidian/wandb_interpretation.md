# W&B Metric Interpretation Guide

This page explains what each logged metric means, why it matters, and what trends we hope to see during training.

## Core Training Metrics
- **`train/loss` (CE loss)**  
  *Definition:* Cross-entropy (CE) between extrapolated logits and the next token.  
  *Why:* Primary signal of how well the model predicts the data.  
  *Goal:* Decrease smoothly; GLT runs should match or beat the baseline curve.

- **`train/grad_norm`**  
  *Definition:* L2 norm of all gradients after accumulation (with clipping applied if enabled).  
  *Why:* Spikes suggest instability or a dominant auxiliary loss.  
  *Goal:* Stay in the same ballpark as baseline; rising trends indicate GLT penalties are overpowering CE.

- **`train/lrm`, `train/dt`, `train/tok_per_sec`, `train/mfu`**  
  *Definition:* Learning-rate multiplier, per-step wall time, throughput, and model FLOP utilization.  
  *Why:* Sanity check for performance regressions.  
  *Goal:* Flat/steady; sudden drops imply hardware or code path changes.

- **`train/loss` vs `glt/loss`**  
  *Definition:* CE-only loss vs. total GLT objective (CE + penalties).  
  *Why:* Highlights when auxiliary losses dominate training.  
  *Goal:* `glt/loss` should be close to CE; large gaps mean lambda scaling needs adjustment.

## Loss Components
- **`loss_components/ce`**  
  *Definition:* CE contribution per step.  
  *Goal:* Mirrors `train/loss`.

- **`loss_components/glt/local`**  
  *Definition:* Midpoint straightness error—distance between yₜ and the SLERP midpoint of (yₜ₋₁, yₜ₊₁).  
  *Goal:* Trend toward zero as local curvature flattens.
- **`loss_components/glt/global`**  
  *Definition:* Span-level straightness (great-circle deviation over long spans).  
  *Goal:* Decrease steadily; high plateaus mean long-range wiggles remain.
- **`loss_components/glt/angle`**  
  *Definition:* Variance of angular step sizes between successive latents.  
  *Goal:* Shrink toward zero—constant step sizes make extrapolation reliable.
- **`loss_components/glt/bi`**  
  *Definition:* Bidirectional midpoint loss; same as `local` but reversed (predict yₜ from the SLERP between yₜ₊₁ and yₜ₋₁) to check symmetry.  
  *Goal:* Track `glt/local`; divergence means the trajectory behaves differently forward vs. backward.
  *Why:* Use these to diagnose which constraint dominates CE; keep them roughly O(1) unless you intentionally crank the lambdas.

## GLT Scalar Diagnostics
- **`glt/curvature_mean` & `glt/curvature_std`**  
  *Definition:* Mean/stdev of \|y_t − midpoint(y_{t−1}, y_{t+1})\|.  
  *Why:* Directly measures how “geodesic” the latent path is.  
  *Goal:* Trend downward as training progresses; GLT runs should sit below the baseline trace.

- **`glt/angle_mean`, `glt/angle_std`, `glt/angle_var`**  
  *Definition:* Statistics of the angular distance between successive latents.  
  *Why:* Constant step sizes simplify extrapolation.  
  *Goal:* Mean stays roughly constant while variance drops; spikes mean angle penalty is too weak.

- **`glt/pre_norm_radius_mean/std`**  
  *Definition:* Norm of hidden states before normalization.  
  *Why:* Detects pathological scaling in the trunk when GLT is active.  
  *Goal:* ~√d (≈27.7 for d=768) with small std; large drifts imply projection bugs.

- **`glt/loss`**  
  *Definition:* Full GLT objective (scaled CE + penalties).  
  *Why:* Gives a single number to compare GLT configs.  
  *Goal:* Track CE closely; divergence indicates penalties dominate.

## Visualization Outputs
- **`viz/pca_directions`**  
  *Definition:* PCA scatter of local direction vectors y_{t+1}−y_t for a sample sequence.  
  *Why:* Shows whether local variations collapse into a low-dimensional manifold.  
  *Goal:* Tight, centered clusters; diffuse clouds mean the trajectory still twists.

- **`viz/curvature_trace`**  
  *Definition:* Token-by-token curvature plot for a representative sequence.  
  *Why:* Reveals where curvature spikes occur (semantic boundaries, artifacts).  
  *Goal:* Lower amplitude and smoother traces as GLT strengthens.

- **`viz/cosine_heatmap`**  
  *Definition:* Pairwise cosine similarity matrix for latents in one sequence.  
  *Why:* Diagnoses long-range structure; diagonal banding indicates consistent geometry.  
  *Goal:* Cleaner band structure vs. baseline; random noise suggests GLT isn’t shaping long-range relationships.

## Histograms
- **`glt/curvature_hist`, `glt/angle_hist`, `glt/pre_norm_radius_hist`**  
  *Definition:* Distribution snapshots for curvature, angles, and norms.  
  *Why:* Displays full tails rather than relying on mean/var.  
  *Goal:* Curvature histograms shift toward zero; angle histograms narrow; radius histograms stay tight around √d.

Use this guide when comparing runs: first confirm the foundational metrics (CE, grad norm, throughput) stay healthy, then move down to the GLT-specific scalars and visuals to check whether the geometry actually improved.
