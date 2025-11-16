# GLT Visualization PRD

## 1. Purpose
This document defines the visualization requirements for the **Geodesic Latent Trajectories (GLT)** project.  
The purpose of these visualizations is to provide **diagnostic insight** into whether the model is learning smooth, near-geodesic trajectories on the hypersphere.

All visualization outputs will be logged to **Weights & Biases (W&B)**.  
No custom UI or local visualization framework is required.

---

## 2. Principles
1. **Do not attempt to visualize the latent space directly in 2D/3D.**  
   High-dimensional projections are misleading.
2. **Visualize scalar invariants** (angles, curvature, norms), which remain meaningful in high-D.
3. **Use images only for structured matrices** (similarity heatmaps, PCA of local differences).
4. **Consistency and interpretability over aesthetics.**

---

## 3. Visualization Categories

### 3.1 Scalar Metrics (Line Plots)
Logged every step or every N steps.

#### (A) Curvature Metrics
Curvature for token t:
```
kappa_t = || y_t - slerp(y_{t-1}, y_{t+1}, 0.5) ||
```

Log:
- `glt/curvature_mean`
- `glt/curvature_std`

#### (B) Angular Step Metrics
Angle between adjacent latents:
```
theta_t = arccos(y_t · y_{t+1})
```

Log:
- `glt/angle_mean`
- `glt/angle_std`
- `glt/angle_var`

#### (C) Pre-Normalization Radius
Norm before L2-normalization:
```
r_t = || h_t ||
```

Log:
- `glt/pre_norm_radius_mean`
- `glt/pre_norm_radius_std`

---

## 4. Histograms
Logged periodically (e.g., every 200–1000 steps).

### 4.1 Distributions to Log
- `glt/curvature_hist`  
  Histogram of κ_t values.

- `glt/angle_hist`  
  Histogram of θ_t values.

- `glt/pre_norm_radius_hist`  
  Histogram of r_t values.

Histogram format:
```python
wandb.Histogram(array)
```

---

## 5. Image-Based Visualizations (Matplotlib → W&B)
Logged less frequently (e.g., every 500–2000 steps).  
Designed to visually inspect local structure.

### 5.1 Cosine Similarity Heatmap
Compute:
```
C_ij = y_i · y_j
```
for a single example sequence.

Requirements:
- Use `imshow` with colorbar.
- Log as `viz/cosine_heatmap`.

### 5.2 Curvature Trace Plot
Plot κ_t over t for a representative sequence.

Requirements:
- `viz/curvature_trace`
- Helps identify where curvature spikes (semantic boundaries, structural transitions).

### 5.3 PCA of Local Direction Vectors (Diagnostic Only)
Compute:
```
v_t = y_{t+1} - y_t
```
Perform PCA → 2D scatter.

Requirements:
- Logged as `viz/pca_directions`
- Not interpreted semantically, used only to see whether local variations collapse/improve over training.

---

## 6. Frequency Guidelines

| Visualization Type | Frequency | Reason |
|--------------------|-----------|--------|
| Scalar metrics     | Every step or N steps (≤50) | Cheap, primary diagnostics |
| Histograms         | ~200–1000 steps             | Captures distribution changes |
| Image visualizers  | ~500–2000 steps             | Expensive; useful infrequent snapshots |

The exact intervals should be controlled by config flags.

---

## 7. Configuration Flags
Add a visualization config section:

```
viz:
  scalar_every: 10
  hist_every: 500
  image_every: 1000
  sequence_index: 0    # which example from batch to visualize
```

---

## 8. Required Deliverables (Coding Agent)

### 8.1 Compute Functions
Implement reusable functions to compute:
- curvature
- angles
- pre-normalization radius
- cosine similarity matrix
- PCA of local directions

### 8.2 Logging Utilities
A module:
```
glt/logging.py
```
with functions:
- `log_scalars(...)`
- `log_histograms(...)`
- `log_images(...)`
- `maybe_log_visualizations(step, data, config)`
- All W&B-compatible.

### 8.3 Matplotlib Plotters
Collocated in a file:
```
glt/plotting.py
```
Functions:
- `plot_cosine_heatmap`
- `plot_curvature_trace`
- `plot_pca_directions`

### 8.4 Integration Into Training Loop
At minimum:
```
if step % viz.scalar_every == 0:
    log_scalars(...)
if step % viz.hist_every == 0:
    log_histograms(...)
if step % viz.image_every == 0:
    log_images(...)
```

---

## 9. Acceptance Criteria
A visualization pass is considered complete when:

1. All scalar metrics appear in W&B line charts.
2. Histograms are visible and update over time.
3. Cosine similarity heatmaps render correctly for a sequence.
4. Curvature trace plots show token-level curvature.
5. PCA direction plots render without errors.
6. Logging does not significantly slow training.
7. Configurable logging frequencies work.

---

## 10. Summary
This PRD defines a **practical, interpretable, high-dimensional visualization suite** for GLT.  
Rather than visualizing the latent geometry directly, which is misleading in high dimensions, we visualize **scalar invariants and structured matrices**.

These diagnostics allow the team to:
- validate geodesic behavior,
- debug training,
- compare runs,
- and understand GLT model dynamics using only W&B.

