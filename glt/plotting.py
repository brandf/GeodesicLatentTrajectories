from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor


def _to_numpy(t: Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def plot_cosine_heatmap(matrix: Tensor, title: str = "Cosine similarity") -> Optional[plt.Figure]:
    if matrix.numel() == 0:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    cax = ax.imshow(_to_numpy(matrix), aspect="auto", cmap="viridis", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Token")
    ax.set_ylabel("Token")
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def plot_curvature_trace(curvature: Tensor, title: str = "Curvature trace") -> Optional[plt.Figure]:
    if curvature.numel() == 0:
        return None
    fig, ax = plt.subplots(figsize=(6, 3))
    xs = np.arange(curvature.numel())
    ax.plot(xs, _to_numpy(curvature))
    ax.set_title(title)
    ax.set_xlabel("Token index (t)")
    ax.set_ylabel("κ_t")
    fig.tight_layout()
    return fig


def plot_pca_directions(points: Tensor, title: str = "PCA of local directions") -> Optional[plt.Figure]:
    if points.size(0) < 2:
        return None
    centered = points - points.mean(dim=0, keepdim=True)
    if centered.abs().max() == 0:
        return None
    q = min(2, centered.size(0), centered.size(1))
    if q < 2:
        return None
    U, S, V = torch.pca_lowrank(centered, q=q)
    components = V[:, :2]
    coords = centered @ components
    fig, ax = plt.subplots(figsize=(4, 4))
    arr = _to_numpy(coords)
    ax.scatter(arr[:, 0], arr[:, 1], s=10, alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    fig.tight_layout()
    return fig
