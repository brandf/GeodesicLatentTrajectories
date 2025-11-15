# Geodesic Latent Trajectories (GLT)
## Technical Design Document (TDD)

This document specifies the implementation details for the **Geodesic Latent Trajectories (GLT)** project, to be integrated into a nanoGPT/nanochat-style LLM stack.

It assumes the reader has the **PRD** (`glt_prd.md`) and **Math Appendix** (`glt_math_appendix.md`) for conceptual and mathematical background.

---

## 1. Objectives (Implementation-Level)

1. Add a **GLT latent head** on top of the transformer to map hidden states into a hyperspherical latent space.
2. Implement the **geometric utilities** (SLERP, log/exp maps, angle computation).
3. Implement **GLT loss components**:
   - token prediction (cross-entropy)
   - local straightness loss
   - global straightness loss (optional)
   - angular spacing loss
   - bi-directional midpoint loss
4. Wire these losses into the training loop with configurable weights.
5. Provide hooks for **extrapolative decoding** (geodesic continuation) for generation experiments.
6. Provide minimal visualization/debug tools for latent trajectories.

The first milestone only requires training and evaluating a model with GLT loss while still doing standard next-token prediction from latent points.

---

## 2. High-Level Architecture

### 2.1 Modules / Files

Proposed structure inside the nanochat fork:

- `glt/geometry.py`
  - Core geometric utilities on the hypersphere.
- `glt/head.py`
  - `GLTHead` module: mapping from transformer hidden states to latent points on the sphere.
- `glt/losses.py`
  - Implementation of all GLT losses.
- `glt/config.py`
  - Dataclasses / dict configs for GLT hyperparameters.
- `glt/visualization.py` (optional, first pass can be simple)
  - Helpers for logging angle distributions and curvature metrics.
- Integration points:
  - `model.py` (or equivalent) — instantiate `GLTHead` and call losses.
  - `train.py` — add GLT-related CLI flags / config entries and loss logging.

### 2.2 Data Flow

At a high level:

```text
input_ids
  → token_embedding
  → transformer blocks
  → hidden states h_t ∈ ℝ^H
  → GLTHead: h_t → y_t ∈ S^{D-1}
  → vocab projection: y_t → logits_t ∈ ℝ^{V}
  → CE loss with target x_{t+1}

Also:
  → from y_t sequence compute:
       L_local, L_global, L_angle, L_bi
  → total loss = weighted sum
```

---

## 3. GLT Geometry Module (`glt/geometry.py`)

All tensor ops should be implemented in PyTorch, batched over:

- batch dimension: `B`
- sequence dimension: `T`
- latent dimension: `D`

Shape conventions:

- `y`: `(B, T, D)` for sequences of latent points on the sphere.
- `u`, `v`: `(B, T, D)` or `(B, D)` for endpoints.

### 3.1 Normalization Utility

```python
def normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # x: (..., D)
    return x / (x.norm(dim=-1, keepdim=True) + eps)
```

### 3.2 Safe Dot and Angle

```python
def safe_dot(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # a, b: (..., D)
    dot = (a * b).sum(dim=-1)
    return dot.clamp(min=-1.0 + eps, max=1.0 - eps)

def angle(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.arccos(safe_dot(a, b, eps=eps))
```

### 3.3 SLERP

```python
def slerp(u: torch.Tensor,
          v: torch.Tensor,
          tau: torch.Tensor,
          eps: float = 1e-6) -> torch.Tensor:
    '''
    u, v: (..., D), assumed normalized
    tau: (...,) or broadcastable scalar in [0,1]
    returns: (..., D)
    '''
    theta = angle(u, v, eps=eps)  # (...,)

    sin_theta = torch.sin(theta)
    # Handle small angles with linear fallback
    small = sin_theta.abs() < eps

    # General case
    coeff_u = torch.sin((1.0 - tau) * theta) / (sin_theta + eps)
    coeff_v = torch.sin(tau * theta) / (sin_theta + eps)
    out = coeff_u.unsqueeze(-1) * u + coeff_v.unsqueeze(-1) * v

    # Linear interpolation fallback for small angles
    lerp = (1.0 - tau).unsqueeze(-1) * u + tau.unsqueeze(-1) * v
    lerp = normalize(lerp)

    out = torch.where(small.unsqueeze(-1), lerp, out)
    out = normalize(out)
    return out
```

### 3.4 Log Map

```python
def log_map(y: torch.Tensor, z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    '''
    Log map on S^{D-1}: T_y S^{D-1} vector pointing from y to z.
    y, z: (..., D), normalized.
    returns: (..., D)
    '''
    theta = angle(y, z, eps=eps)  # (...,)
    sin_theta = torch.sin(theta)

    # Project z onto tangent space at y
    dot = safe_dot(y, z, eps=eps)  # (...,)
    proj = z - dot.unsqueeze(-1) * y  # (..., D)

    # Avoid division by zero / extremely small angles
    scale = theta / (sin_theta + eps)  # (...,)
    v = scale.unsqueeze(-1) * proj
    # For extremely small theta, log_map(y,z) ≈ z - y (but projected) → proj
    small = theta.abs() < eps
    v = torch.where(small.unsqueeze(-1), proj, v)
    return v
```

### 3.5 Exp Map

```python
def exp_map(y: torch.Tensor, v: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    '''
    Exponential map on S^{D-1}: from tangent vector at y to point on the sphere.
    y: (..., D), normalized
    v: (..., D), tangent (approximately y·v ≈ 0)
    '''
    norm_v = v.norm(dim=-1)  # (...,)
    # Handle zero norm (no movement)
    zero = norm_v < eps

    # Unit direction in tangent
    u = torch.zeros_like(v)
    u[~zero] = v[~zero] / norm_v[~zero].unsqueeze(-1)

    cos_term = torch.cos(norm_v).unsqueeze(-1) * y
    sin_term = torch.sin(norm_v).unsqueeze(-1) * u
    out = cos_term + sin_term

    # For zero norm, just return y
    out = torch.where(zero.unsqueeze(-1), y, out)
    out = normalize(out)
    return out
```

---

## 4. GLT Head Module (`glt/head.py`)

### 4.1 Design

`GLTHead` takes in transformer hidden states `h` and outputs normalized latent vectors `y`:

- Input: `h` with shape `(B, T, H)`
- Output: `y` with shape `(B, T, D)` on \(S^{D-1}\)

### 4.2 Implementation Sketch

```python
import torch
import torch.nn as nn
from .geometry import normalize

class GLTHead(nn.Module):
    def __init__(self, hidden_size: int, latent_size: int, use_mlp: bool = True):
        super().__init__()
        if use_mlp:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, latent_size),
                nn.GELU(),
                nn.Linear(latent_size, latent_size),
            )
        else:
            self.proj = nn.Linear(hidden_size, latent_size)

        self.out_norm_eps = 1e-8

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        '''
        h: (B, T, H)
        returns: y: (B, T, D) normalized
        '''
        y = self.proj(h)  # (B, T, D)
        y = normalize(y, eps=self.out_norm_eps)
        return y
```

---

## 5. GLT Losses (`glt/losses.py`)

All losses operate on `y: (B, T, D)`.

We assume standard CE token loss is computed elsewhere using logits from a vocab projection of `y`.

### 5.1 Local Straightness / Bi-Directional Loss

```python
from .geometry import slerp

def local_midpoint_loss(y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    '''
    y: (B, T, D)
    mask: (B, T) or None, 1 for valid tokens, 0 for padding.
          If provided, only positions with t-1, t, t+1 all valid are included.
    Returns scalar loss (mean over valid midpoints).
    '''
    B, T, D = y.shape
    if T < 3:
        return y.new_tensor(0.0)

    y_prev = y[:, :-2, :]  # (B, T-2, D)
    y_mid  = y[:, 1:-1, :] # (B, T-2, D)
    y_next = y[:, 2:  , :] # (B, T-2, D)

    # geodesic midpoint between y_prev and y_next
    tau = y_prev.new_full((B, T-2), 0.5)
    y_hat = slerp(y_prev, y_next, tau)  # (B, T-2, D)

    diff = (y_mid - y_hat).pow(2).sum(dim=-1)  # (B, T-2)

    if mask is not None:
        m_prev = mask[:, :-2]
        m_mid  = mask[:, 1:-1]
        m_next = mask[:, 2:]
        m = m_prev * m_mid * m_next
        diff = diff * m
        denom = m.sum().clamp(min=1.0)
    else:
        denom = diff.numel()

    return diff.sum() / denom
```

### 5.2 Global Straightness Loss

```python
import torch
from .geometry import slerp

def global_straightness_loss(y: torch.Tensor,
                             mask: torch.Tensor | None = None,
                             num_spans: int = 1) -> torch.Tensor:
    '''
    y: (B, T, D)
    mask: (B, T) or None
    num_spans: number of random segments per batch to sample
    '''
    B, T, D = y.shape
    if T < 3 or num_spans <= 0:
        return y.new_tensor(0.0)

    device = y.device
    losses = []

    for _ in range(num_spans):
        # random start and end indices with at least distance 2
        s = torch.randint(0, T - 2, (1,), device=device).item()
        t = torch.randint(s + 2, T, (1,), device=device).item()
        seg_len = t - s + 1

        y_s = y[:, s, :]  # (B, D)
        y_t = y[:, t, :]  # (B, D)

        # tau: (seg_len,) equally spaced between 0 and 1
        tau = torch.linspace(0.0, 1.0, seg_len, device=device)  # (seg_len,)
        # Broadcast to (B, seg_len)
        tau = tau.unsqueeze(0).expand(B, seg_len)

        # y_hat: (B, seg_len, D)
        y_hat = slerp(y_s.unsqueeze(1).expand(B, seg_len, D),
                      y_t.unsqueeze(1).expand(B, seg_len, D),
                      tau)

        y_seg = y[:, s:t+1, :]  # (B, seg_len, D)
        diff = (y_seg - y_hat).pow(2).sum(dim=-1)  # (B, seg_len)

        if mask is not None:
            m_seg = mask[:, s:t+1]
            diff = diff * m_seg
            denom = m_seg.sum().clamp(min=1.0)
        else:
            denom = diff.numel()

        losses.append(diff.sum() / denom)

    if not losses:
        return y.new_tensor(0.0)

    return torch.stack(losses).mean()
```

### 5.3 Angular Spacing Loss

```python
from .geometry import angle

def angular_spacing_loss(y: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    '''
    Encourage constant angular step size between consecutive y_t.
    y: (B, T, D)
    mask: (B, T) or None
    '''
    B, T, D = y.shape
    if T < 2:
        return y.new_tensor(0.0)

    y_curr = y[:, :-1, :]  # (B, T-1, D)
    y_next = y[:, 1: , :]  # (B, T-1, D)
    theta = angle(y_curr, y_next)  # (B, T-1)

    if mask is not None:
        m_curr = mask[:, :-1]
        m_next = mask[:, 1:]
        m = m_curr * m_next
        theta = theta * m
        denom = m.sum().clamp(min=1.0)
        # Compute mean over valid
        mean_theta = theta.sum() / denom
        var = ((theta - mean_theta) ** 2 * m).sum() / denom
    else:
        mean_theta = theta.mean()
        var = ((theta - mean_theta) ** 2).mean()

    return var
```

---

## 6. Config (`glt/config.py`)

```python
from dataclasses import dataclass

@dataclass
class GLTConfig:
    latent_size: int = 512
    use_mlp_head: bool = True

    lambda_ce: float = 1.0
    lambda_local: float = 0.3
    lambda_global: float = 0.05
    lambda_angle: float = 0.1
    lambda_bi: float = 0.1

    global_num_spans: int = 1  # segments per batch for global loss
```

---

## 7. Model Integration

### 7.1 In the Model Class

Assume there is a `GPT`-like model with:

- input embeddings
- transformer blocks
- final hidden states `h` of shape `(B, T, H)`
- vocab projection `lm_head`

We modify:

1. Add a `GLTHead` instance.
2. Replace the input to `lm_head` with `y` instead of `h`.

#### Example

```python
class GPTWithGLT(nn.Module):
    def __init__(self, base_cfg, glt_cfg: GLTConfig, vocab_size: int):
        super().__init__()
        # ... build transformer, embeddings, etc.
        self.transformer = Transformer(base_cfg)
        self.glt_head = GLTHead(hidden_size=base_cfg.n_embd,
                                latent_size=glt_cfg.latent_size,
                                use_mlp=glt_cfg.use_mlp_head)
        self.lm_head = nn.Linear(glt_cfg.latent_size, vocab_size, bias=False)

        self.glt_cfg = glt_cfg

    def forward(self, idx, targets=None, mask=None):
        # idx: (B, T)
        # mask: (B, T) or None
        h = self.transformer(idx)  # (B, T, H)
        y = self.glt_head(h)       # (B, T, D)

        logits = self.lm_head(y)   # (B, T, V)

        loss = None
        losses = {}

        if targets is not None:
            # Standard CE next-token prediction
            ce_loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                targets[:, 1:].reshape(-1),
                reduction='mean',
            )
            losses['ce'] = ce_loss

            # GLT losses
            l_local = local_midpoint_loss(y, mask=mask)
            l_global = global_straightness_loss(
                y, mask=mask, num_spans=self.glt_cfg.global_num_spans
            )
            l_angle = angular_spacing_loss(y, mask=mask)
            # bi-loss can be same as local; kept separate for weighting
            l_bi = l_local

            losses['local'] = l_local
            losses['global'] = l_global
            losses['angle'] = l_angle
            losses['bi'] = l_bi

            loss = (
                self.glt_cfg.lambda_ce * ce_loss
                + self.glt_cfg.lambda_local * l_local
                + self.glt_cfg.lambda_global * l_global
                + self.glt_cfg.lambda_angle * l_angle
                + self.glt_cfg.lambda_bi * l_bi
            )

        return logits, loss, losses
```

---

## 8. Training Integration

Key changes:

1. Extend config/CLI to include a `GLTConfig`.
2. Construct `GPTWithGLT` instead of the vanilla model if GLT is enabled.
3. During training:
   - Log each component of `losses` (CE, local, global, angle, bi).
   - Optionally log derived metrics from `y` (mean angle, angle variance, midpoint error).

---

## 9. Inference & Extrapolative Decoding

### 9.1 Function to Extrapolate `y` Forward

```python
from .geometry import log_map, exp_map, angle

def extrapolate_next_latent(y_seq: torch.Tensor,
                            use_mean_step: bool = True) -> torch.Tensor:
    '''
    y_seq: (T, D) or (1, T, D) — last few latents for a single example.
    returns: y_next: (D,)
    '''
    if y_seq.dim() == 3:
        y_seq = y_seq[0]  # assume batch=1
    T, D = y_seq.shape
    assert T >= 2, 'Need at least two points to extrapolate.'

    y_prev = y_seq[-2]
    y_curr = y_seq[-1]

    # tangent at y_curr pointing forward
    v_back = log_map(y_curr.unsqueeze(0), y_prev.unsqueeze(0))[0]
    v_forward = -v_back  # flip direction

    if use_mean_step and T > 2:
        # estimate mean step angle over the sequence
        angles = []
        for i in range(T - 1):
            a = angle(y_seq[i].unsqueeze(0), y_seq[i+1].unsqueeze(0))[0]
            angles.append(a)
        mean_theta = torch.stack(angles).mean()
        curr_theta = v_forward.norm()
        if curr_theta > 1e-6:
            v_forward = v_forward * (mean_theta / curr_theta)

    y_next = exp_map(y_curr.unsqueeze(0), v_forward.unsqueeze(0))[0]
    return y_next
```

### 9.2 Extrapolative Generation Mode (Experimental)

1. Maintain the current latent sequence `y_1, …, y_T`.
2. Extrapolate `y_{T+1}` using `extrapolate_next_latent`.
3. Decode logits from `y_{T+1}` via the same vocab projection.
4. Sample a token.
5. Append this token to the input sequence; optionally re-run the transformer or treat this as a latent-only research mode.

---

## 10. Edge Cases & Numerical Concerns

1. Very small angles:
   - Use clamping and `eps` in `angle`, `slerp`, `log_map`, `exp_map`.
2. Antipodal points (`y ≈ -z`):
   - Rare if trajectories are smooth; can jitter, skip, or fallback to Euclidean interpolation.
3. Masking:
   - All losses should accept an optional `(B, T)` mask to exclude padding and invalid contexts.

---

## 11. Deliverables Checklist

1. `glt/geometry.py`
   - `normalize`, `safe_dot`, `angle`, `slerp`, `log_map`, `exp_map`.
2. `glt/head.py`
   - `GLTHead` with configurable latent dimension and MLP flag.
3. `glt/losses.py`
   - `local_midpoint_loss`
   - `global_straightness_loss`
   - `angular_spacing_loss`
4. `glt/config.py`
   - `GLTConfig`.
5. Model integration:
   - `GPTWithGLT` (or equivalent modification).
6. Training integration:
   - Config/CLI, logging for all GLT loss terms.
7. Optional:
   - Visualization utilities for angle histograms and midpoint errors.
   - Extrapolative decoding helper.

---

## 12. Summary

This TDD defines a concrete implementation plan to realize:

> **Learn a mapping from token embeddings to a hyperspherical latent space where the sequence follows a near-geodesic curve.**

and

> **Learn a latent space where token trajectories become linear in log-map coordinates.**

The core components are:

- a GLT head that projects transformer states onto a hypersphere,
- Riemannian-inspired losses that encourage locally straight, globally coherent trajectories,
- and latent-based decoding (with an optional extrapolative mode) built on those trajectories.
