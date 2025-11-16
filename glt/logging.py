from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import wandb

from .metrics import (
    angular_steps,
    curvature_values,
    ensure_normalized,
    flatten_valid,
    local_directions,
    pre_norm_radius,
    cosine_similarity_matrix,
)
from .plotting import plot_cosine_heatmap, plot_curvature_trace, plot_pca_directions
from .viz_config import VizConfig


@dataclass
class VizBatch:
    latents: torch.Tensor
    pre_norm: torch.Tensor
    mask: Optional[torch.Tensor] = None

    def to(self, device: str) -> "VizBatch":
        return VizBatch(
            latents=self.latents.to(device),
            pre_norm=self.pre_norm.to(device),
            mask=None if self.mask is None else self.mask.to(device),
        )


def _scalar_dict(latents: torch.Tensor, pre_norm: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, float]:
    latents = ensure_normalized(latents)
    scalars: Dict[str, float] = {}
    curv, curv_mask = curvature_values(latents, mask=mask)
    ang, ang_mask = angular_steps(latents, mask=mask)
    radius, radius_mask = pre_norm_radius(pre_norm, mask=mask)

    flat_curv = flatten_valid(curv, curv_mask)
    if flat_curv.numel() > 0:
        scalars["glt/curvature_mean"] = flat_curv.mean().item()
        scalars["glt/curvature_std"] = flat_curv.std(unbiased=False).item()

    flat_ang = flatten_valid(ang, ang_mask)
    if flat_ang.numel() > 0:
        scalars["glt/angle_mean"] = flat_ang.mean().item()
        scalars["glt/angle_std"] = flat_ang.std(unbiased=False).item()
        scalars["glt/angle_var"] = flat_ang.var(unbiased=False).item()

    flat_radius = flatten_valid(radius, radius_mask)
    if flat_radius.numel() > 0:
        scalars["glt/pre_norm_radius_mean"] = flat_radius.mean().item()
        scalars["glt/pre_norm_radius_std"] = flat_radius.std(unbiased=False).item()
    return scalars


def _hist_dict(latents: torch.Tensor, pre_norm: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, wandb.Histogram]:
    latents = ensure_normalized(latents)
    hist_data: Dict[str, wandb.Histogram] = {}
    curv, curv_mask = curvature_values(latents, mask=mask)
    ang, ang_mask = angular_steps(latents, mask=mask)
    radius, radius_mask = pre_norm_radius(pre_norm, mask=mask)

    flat_curv = flatten_valid(curv, curv_mask)
    if flat_curv.numel() > 0:
        hist_data["glt/curvature_hist"] = wandb.Histogram(flat_curv.cpu().numpy())
    flat_ang = flatten_valid(ang, ang_mask)
    if flat_ang.numel() > 0:
        hist_data["glt/angle_hist"] = wandb.Histogram(flat_ang.cpu().numpy())
    flat_radius = flatten_valid(radius, radius_mask)
    if flat_radius.numel() > 0:
        hist_data["glt/pre_norm_radius_hist"] = wandb.Histogram(flat_radius.cpu().numpy())
    return hist_data


def log_scalar_metrics(step: int, viz_batch: VizBatch, wandb_run):
    scalars = _scalar_dict(viz_batch.latents, viz_batch.pre_norm, viz_batch.mask)
    if scalars:
        wandb_run.log(scalars, step=step)


def log_histograms(step: int, viz_batch: VizBatch, wandb_run):
    hist = _hist_dict(viz_batch.latents, viz_batch.pre_norm, viz_batch.mask)
    if hist:
        wandb_run.log(hist, step=step)


def log_images(step: int, viz_batch: VizBatch, sequence_index: int, wandb_run):
    latents = ensure_normalized(viz_batch.latents)
    b = min(max(sequence_index, 0), latents.size(0) - 1)
    seq = latents[b]
    mask = viz_batch.mask[b] if viz_batch.mask is not None else None
    valid_len = seq.size(0)
    if mask is not None:
        valid_len = int(mask.sum().item())
    seq = seq[:valid_len]
    if seq.size(0) == 0:
        return
    cos = cosine_similarity_matrix(seq)
    curvature, _ = curvature_values(seq.unsqueeze(0), mask=None)
    curvature = curvature.squeeze(0)
    dirs = local_directions(seq)

    fig_heat = plot_cosine_heatmap(cos, title=f"Cosine similarity (seq {b})")
    fig_curv = plot_curvature_trace(curvature, title=f"Curvature trace (seq {b})")
    fig_pca = plot_pca_directions(dirs, title=f"PCA directions (seq {b})")

    log_data: Dict[str, Any] = {}
    if fig_heat is not None:
        log_data["viz/cosine_heatmap"] = wandb.Image(fig_heat)
    if fig_curv is not None:
        log_data["viz/curvature_trace"] = wandb.Image(fig_curv)
    if fig_pca is not None:
        log_data["viz/pca_directions"] = wandb.Image(fig_pca)
    if log_data:
        wandb_run.log(log_data, step=step)
    import matplotlib.pyplot as plt
    plt.close("all")


def maybe_log_visualizations(step: int, viz_cfg: VizConfig, viz_batch: Optional[VizBatch], wandb_run):
    if not viz_cfg.enabled or viz_batch is None:
        return
    if viz_cfg.wants_scalar(step):
        log_scalar_metrics(step, viz_batch, wandb_run)
    if viz_cfg.wants_hist(step):
        log_histograms(step, viz_batch, wandb_run)
    if viz_cfg.wants_image(step):
        log_images(step, viz_batch, viz_cfg.sequence_index, wandb_run)
