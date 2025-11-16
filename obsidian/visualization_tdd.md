# GLT Visualization Technical Design Document (TDD)

This TDD specifies how to implement the visualization and logging pipeline for **Geodesic Latent Trajectories (GLT)** using **Weights & Biases (W&B)**.

It corresponds to the requirements in `visualization_prd.md` and is aimed at a coding agent integrating with a nanoGPT/nanochat-style training loop.

---

## 1. Scope

- All visual diagnostics are computed from:
  - final hidden states `h` (before L2 normalization), and
  - normalized latents `y` on the unit sphere.
- All outputs are logged to **W&B** as:
  - Scalars
  - Histograms
  - Images (matplotlib figures)

There is no requirement to build a custom web UI. Training is assumed to run over SSH on a remote GPU.

---

## 2. Module Layout

Suggested directory structure (under the project root):

- `glt/metrics.py`
  - Compute curvature, angles, radii, cosine similarity matrices, local differences.
- `glt/plotting.py`
  - Matplotlib-based figure generators (heatmaps, traces, PCA scatter).
- `glt/logging.py`
  - W&B-oriented logging helpers.
- `glt/viz_config.py`
  - Simple dataclass/config for visualization frequency and options.

These modules should be independent of the core model’s implementation details except for requiring tensors `h` and `y`.

---

## 3. Data Inputs and Shapes

Assume standard PyTorch tensor shapes:

- `h`: `(B, T, D)`
  - Final hidden states from the transformer *before* L2 normalization.
- `y`: `(B, T, D)`
  - L2-normalized latents: `y = h / ||h||`.
- `mask`: `(B, T)` or `None`
  - 1 for valid tokens, 0 for padding.

All metrics operate in batch mode and then reduce over batch and sequence as needed.

---

## 4. Metrics Implementation (`glt/metrics.py`)

### 4.1 Normalization Utility

Even if normalization exists elsewhere, define a utility here for safety:

```python
import torch

def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)
```

### 4.2 Safe Dot and Angle

```python
def safe_dot(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    dot = (a * b).sum(dim=-1)
    return dot.clamp(min=-1.0 + eps, max=1.0 - eps)

def angle(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.arccos(safe_dot(a, b, eps=eps))
```

### 4.3 SLERP (for Curvature)

```python
def slerp(u: torch.Tensor,
          v: torch.Tensor,
          tau: torch.Tensor,
          eps: float = 1e-6) -> torch.Tensor:
    '''
    u, v: (..., D) normalized
    tau: (...,) or scalar in [0,1]
    returns: (..., D)
    '''
    theta = angle(u, v, eps=eps)           # (...)
    sin_theta = torch.sin(theta)
    small = sin_theta.abs() < eps

    # General case
    coeff_u = torch.sin((1.0 - tau) * theta) / (sin_theta + eps)
    coeff_v = torch.sin(tau * theta) / (sin_theta + eps)

    out = coeff_u.unsqueeze(-1) * u + coeff_v.unsqueeze(-1) * v

    # Linear fallback
    lerp = (1.0 - tau).unsqueeze(-1) * u + tau.unsqueeze(-1) * v
    lerp = l2_normalize(lerp)

    out = torch.where(small.unsqueeze(-1), lerp, out)
    out = l2_normalize(out)
    return out
```

### 4.4 Curvature (κ_t)

For each batch and sequence position (excluding ends):

```python
def curvature(y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    '''
    y: (B, T, D) normalized
    mask: (B, T) or None
    returns: kappa: (B, T-2) with curvature per interior token
    '''
    B, T, D = y.shape
    if T < 3:
        return y.new_zeros((B, 0))

    y_prev = y[:, :-2, :]   # (B, T-2, D)
    y_mid  = y[:, 1:-1, :]  # (B, T-2, D)
    y_next = y[:, 2:, :]    # (B, T-2, D)

    tau = y_prev.new_full((B, T-2), 0.5)
    y_hat = slerp(y_prev, y_next, tau)  # (B, T-2, D)

    diff = (y_mid - y_hat).pow(2).sum(dim=-1).sqrt()  # (B, T-2)

    if mask is not None:
        m_prev = mask[:, :-2]
        m_mid  = mask[:, 1:-1]
        m_next = mask[:, 2:]
        m = m_prev * m_mid * m_next
        diff = diff * m
    return diff
```

### 4.5 Angular Steps (θ_t)

```python
def angular_steps(y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    '''
    y: (B, T, D)
    returns: theta: (B, T-1)
    '''
    B, T, D = y.shape
    if T < 2:
        return y.new_zeros((B, 0))

    y_curr = y[:, :-1, :]
    y_next = y[:, 1:, :]
    theta = angle(y_curr, y_next)  # (B, T-1)
    if mask is not None:
        m_curr = mask[:, :-1]
        m_next = mask[:, 1:]
        theta = theta * (m_curr * m_next)
    return theta
```

### 4.6 Pre-Normalization Radius

```python
def pre_norm_radius(h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    '''
    h: (B, T, D)
    returns: r: (B, T)
    '''
    r = h.norm(dim=-1)  # (B, T)
    if mask is not None:
        r = r * mask
    return r
```

### 4.7 Cosine Similarity Matrix (Single Sequence)

```python
def cosine_similarity_matrix(y_seq: torch.Tensor) -> torch.Tensor:
    '''
    y_seq: (T, D)
    returns: (T, T) matrix C_ij = y_i · y_j
    '''
    return y_seq @ y_seq.T
```

### 4.8 Local Direction Vectors for PCA

```python
def local_directions(y_seq: torch.Tensor) -> torch.Tensor:
    '''
    y_seq: (T, D)
    returns: v: (T-1, D) where v_t = y_{t+1} - y_t
    '''
    return y_seq[1:, :] - y_seq[:-1, :]
```

---

## 5. Plotting Implementation (`glt/plotting.py`)

Use **matplotlib** for all plotting.

### 5.1 Cosine Similarity Heatmap

```python
import matplotlib.pyplot as plt

def plot_cosine_heatmap(C, title: str = "Cosine similarity heatmap"):
    '''
    C: (T, T) numpy array or torch.Tensor
    returns: matplotlib Figure
    '''
    if hasattr(C, "detach"):
        C = C.detach().cpu().numpy()

    fig, ax = plt.subplots()
    im = ax.imshow(C, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("token index")
    ax.set_ylabel("token index")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig
```

### 5.2 Curvature Trace

```python
def plot_curvature_trace(kappa_seq, title: str = "Curvature vs token index"):
    '''
    kappa_seq: (T-2,) 1D array or tensor
    '''
    if hasattr(kappa_seq, "detach"):
        kappa_seq = kappa_seq.detach().cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(kappa_seq)
    ax.set_title(title)
    ax.set_xlabel("token index (interior)")
    ax.set_ylabel("kappa_t")
    fig.tight_layout()
    return fig
```

### 5.3 PCA of Local Directions

Use scikit-learn’s PCA (assumed available in the environment).

```python
from sklearn.decomposition import PCA

def plot_pca_directions(v, title: str = "PCA of local directions"):
    '''
    v: (T-1, D) direction vectors
    '''
    if hasattr(v, "detach"):
        v = v.detach().cpu().numpy()

    if v.shape[0] < 2:
        # Not enough points
        return None

    pca = PCA(n_components=2)
    v_2d = pca.fit_transform(v)

    fig, ax = plt.subplots()
    ax.scatter(v_2d[:, 0], v_2d[:, 1], s=5)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig
```

---

## 6. Visualization Config (`glt/viz_config.py`)

Simple dataclass:

```python
from dataclasses import dataclass

@dataclass
class VizConfig:
    scalar_every: int = 10
    hist_every: int = 500
    image_every: int = 1000
    sequence_index: int = 0  # which example in batch to visualize
    enabled: bool = True
```

This can be nested into the main experiment config.

---

## 7. Logging Utilities (`glt/logging.py`)

Assumes `wandb` is already initialized in the training script.

### 7.1 Scalar Logging

```python
import wandb
from .metrics import curvature, angular_steps, pre_norm_radius

def log_scalar_metrics(step: int,
                       y: torch.Tensor,
                       h: torch.Tensor,
                       mask: torch.Tensor | None,
                       prefix: str = "glt/"):
    kappa = curvature(y, mask=mask)       # (B, T-2)
    theta = angular_steps(y, mask=mask)   # (B, T-1)
    radius = pre_norm_radius(h, mask=mask)

    # Flatten non-zero entries if mask exists
    def valid_flat(x):
        if x.numel() == 0:
            return x.new_zeros((0,))
        return x[x != 0] if mask is not None else x.reshape(-1)

    kappa_flat = valid_flat(kappa)
    theta_flat = valid_flat(theta)
    radius_flat = valid_flat(radius)

    log_data = {}

    if kappa_flat.numel() > 0:
        log_data[prefix + "curvature_mean"] = kappa_flat.mean().item()
        log_data[prefix + "curvature_std"] = kappa_flat.std().item()

    if theta_flat.numel() > 0:
        log_data[prefix + "angle_mean"] = theta_flat.mean().item()
        log_data[prefix + "angle_std"] = theta_flat.std().item()
        log_data[prefix + "angle_var"] = theta_flat.var().item()

    if radius_flat.numel() > 0:
        log_data[prefix + "pre_norm_radius_mean"] = radius_flat.mean().item()
        log_data[prefix + "pre_norm_radius_std"] = radius_flat.std().item()

    if log_data:
        wandb.log(log_data, step=step)
```

### 7.2 Histograms

```python
def log_histograms(step: int,
                   y: torch.Tensor,
                   h: torch.Tensor,
                   mask: torch.Tensor | None,
                   prefix: str = "glt/"):
    kappa = curvature(y, mask=mask)
    theta = angular_steps(y, mask=mask)
    radius = pre_norm_radius(h, mask=mask)

    def valid_flat(x):
        if x.numel() == 0:
            return x.new_zeros((0,))
        return x[x != 0] if mask is not None else x.reshape(-1)

    kappa_flat = valid_flat(kappa).detach().cpu().numpy()
    theta_flat = valid_flat(theta).detach().cpu().numpy()
    radius_flat = valid_flat(radius).detach().cpu().numpy()

    log_data = {}
    if kappa_flat.size > 0:
        log_data[prefix + "curvature_hist"] = wandb.Histogram(kappa_flat)
    if theta_flat.size > 0:
        log_data[prefix + "angle_hist"] = wandb.Histogram(theta_flat)
    if radius_flat.size > 0:
        log_data[prefix + "pre_norm_radius_hist"] = wandb.Histogram(radius_flat)

    if log_data:
        wandb.log(log_data, step=step)
```

### 7.3 Image Visualizations

```python
from .plotting import plot_cosine_heatmap, plot_curvature_trace, plot_pca_directions
from .metrics import cosine_similarity_matrix, local_directions

def log_images(step: int,
               y: torch.Tensor,
               mask: torch.Tensor | None,
               seq_index: int = 0,
               prefix: str = "viz/"):
    '''
    y: (B, T, D)
    Pick one sequence to visualize.
    '''
    if y.size(0) == 0:
        return

    b = min(seq_index, y.size(0) - 1)
    y_seq = y[b]  # (T, D)

    # If mask provided, optionally truncate at last valid token
    if mask is not None:
        m_seq = mask[b]  # (T,)
        valid_len = int(m_seq.sum().item())
        y_seq = y_seq[:valid_len]

    # Cosine similarity heatmap
    C = cosine_similarity_matrix(y_seq)
    fig_C = plot_cosine_heatmap(C, title=f"Cosine similarity (seq {b})")

    # Curvature trace
    from .metrics import curvature as curvature_single
    kappa_seq = curvature_single(y_seq.unsqueeze(0), mask=None).squeeze(0)  # (T-2,)
    fig_k = plot_curvature_trace(kappa_seq, title=f"Curvature trace (seq {b})")

    # PCA of local directions
    v = local_directions(y_seq)
    fig_pca = plot_pca_directions(v, title=f"PCA of directions (seq {b})")

    log_data = {}
    if fig_C is not None:
        log_data[prefix + "cosine_heatmap"] = wandb.Image(fig_C)
    if fig_k is not None:
        log_data[prefix + "curvature_trace"] = wandb.Image(fig_k)
    if fig_pca is not None:
        log_data[prefix + "pca_directions"] = wandb.Image(fig_pca)

    if log_data:
        wandb.log(log_data, step=step)

    # Close figures to free memory
    import matplotlib.pyplot as plt
    plt.close("all")
```

### 7.4 Orchestrator Helper

```python
from .viz_config import VizConfig

def maybe_log_visualizations(step: int,
                             viz_cfg: VizConfig,
                             y: torch.Tensor,
                             h: torch.Tensor,
                             mask: torch.Tensor | None):
    if not viz_cfg.enabled:
        return

    if step % viz_cfg.scalar_every == 0:
        log_scalar_metrics(step, y, h, mask)

    if step % viz_cfg.hist_every == 0:
        log_histograms(step, y, h, mask)

    if step % viz_cfg.image_every == 0:
        log_images(step, y, mask, seq_index=viz_cfg.sequence_index)
```

---

## 8. Integration with Training Loop

In the main training script:

1. Initialize W&B at the start of the run.
2. Create a `VizConfig` from the experiment config.
3. After each forward/backward step, call `maybe_log_visualizations`.

Example:

```python
import wandb
from glt.viz_config import VizConfig
from glt.logging import maybe_log_visualizations

wandb.init(project="glt-experiments", config=cfg)

viz_cfg = VizConfig(
    scalar_every=cfg.viz.scalar_every,
    hist_every=cfg.viz.hist_every,
    image_every=cfg.viz.image_every,
    sequence_index=cfg.viz.sequence_index,
    enabled=cfg.viz.enabled,
)

for step, batch in enumerate(train_loader):
    # forward pass
    logits, loss, losses, h, y = model(batch)  # ensure model returns h and y
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # base loss logging
    wandb.log({
        "loss/total": loss.item(),
        "loss/ce": losses["ce"].item(),
        # ... other loss components ...
    }, step=step)

    # visualization logging
    maybe_log_visualizations(step, viz_cfg, y, h, batch.get("mask", None))
```

Note: The model’s forward should optionally return both `h` and `y` so the visualization module can access them.

---

## 9. Testing

Basic tests (can be light-weight):

- **Unit-level:**
  - `metrics.curvature` returns near-zero on a perfect SLERP-generated geodesic.
  - `metrics.angular_steps` returns near-constant angles for synthetic data.
  - `plotting` functions return non-`None` `Figure` objects for valid inputs.
- **Smoke test:**
  - Run a tiny training loop (few steps, synthetic data) with W&B disabled or mocked, ensure:
    - no exceptions raised,
    - metric shapes are as expected.

---

## 10. Summary

This TDD defines:

- The functions to compute GLT-relevant metrics (curvature, angles, radii).
- The plotting logic to render interpretable diagnostics.
- The W&B logging pipeline to visualize everything remotely.
- The integration points in the training loop, with configurable logging frequencies.

The end result is a robust, SSH-friendly visualization suite that makes high-dimensional GLT behavior inspectable without ever trying to visualize the latent space directly in 2D.