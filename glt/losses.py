from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .geometry import angle, slerp


def _masked_mean(values: Tensor, mask: Optional[Tensor]) -> Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(values.dtype)
    denom = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denom


def _mask_triplet(mask: Tensor) -> Tensor:
    return mask[:, :-2] * mask[:, 1:-1] * mask[:, 2:]


def local_midpoint_loss(y: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-6) -> Tensor:
    """
    Encourage the point y_t to lie near the geodesic midpoint of y_{t-1} and y_{t+1}.
    """
    if y.size(1) < 3:
        return y.new_tensor(0.0)
    y_prev, y_mid, y_next = y[:, :-2, :], y[:, 1:-1, :], y[:, 2:, :]
    tau = y_prev.new_full(y_prev.shape[:-1], 0.5)
    midpoint = slerp(y_prev, y_next, tau, eps=eps)
    diff = (y_mid - midpoint).pow(2).sum(dim=-1)
    if mask is not None:
        weights = _mask_triplet(mask).to(diff.dtype)
        diff = diff * weights
        denom = weights.sum().clamp_min(1.0)
    else:
        denom = diff.numel()
    return diff.sum() / denom


def bidirectional_midpoint_loss(y: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-6) -> Tensor:
    """Same as local straightness but swapping endpoints to emphasize symmetry."""
    if y.size(1) < 3:
        return y.new_tensor(0.0)
    y_prev, y_mid, y_next = y[:, :-2, :], y[:, 1:-1, :], y[:, 2:, :]
    tau = y_prev.new_full(y_prev.shape[:-1], 0.5)
    midpoint = slerp(y_next, y_prev, tau, eps=eps)
    diff = (y_mid - midpoint).pow(2).sum(dim=-1)
    if mask is not None:
        weights = _mask_triplet(mask).to(diff.dtype)
        diff = diff * weights
        denom = weights.sum().clamp_min(1.0)
    else:
        denom = diff.numel()
    return diff.sum() / denom


def _iter_spans(total_steps: int, num_spans: int, span_len: int) -> List[Tuple[int, int]]:
    if total_steps < 3 or num_spans <= 0:
        return []
    span_len = max(span_len, 3)
    max_start = total_steps - 3
    if max_start < 0:
        return []
    stride = max(1, (max_start + num_spans) // num_spans)
    indices: List[Tuple[int, int]] = []
    start = 0
    for _ in range(num_spans):
        end = min(total_steps - 1, start + span_len - 1)
        if end - start >= 2:
            indices.append((start, end))
        start = min(start + stride, max_start)
    return indices


def global_straightness_loss(
    y: Tensor,
    mask: Optional[Tensor] = None,
    num_spans: int = 1,
    span_len: Optional[int] = None,
    eps: float = 1e-6,
) -> Tensor:
    """
    Penalize deviations from great-circle arcs between distant endpoints.
    """
    T = y.size(1)
    if T < 3 or num_spans <= 0:
        return y.new_tensor(0.0)
    if span_len is None:
        span_len = max(3, T // max(1, num_spans))
    losses = []
    for start, end in _iter_spans(T, num_spans, span_len):
        seg_len = end - start + 1
        y_start = y[:, start, :].unsqueeze(1).expand(-1, seg_len, -1)
        y_end = y[:, end, :].unsqueeze(1).expand(-1, seg_len, -1)
        tau = torch.linspace(0.0, 1.0, seg_len, device=y.device, dtype=y.dtype)
        tau = tau.unsqueeze(0).expand(y.size(0), seg_len)
        ref = slerp(y_start, y_end, tau, eps=eps)
        segment = y[:, start : end + 1, :]
        diff = (segment - ref).pow(2).sum(dim=-1)
        if mask is not None:
            weights = mask[:, start : end + 1].to(diff.dtype)
            diff = diff * weights
            denom = weights.sum().clamp_min(1.0)
        else:
            denom = diff.numel()
        losses.append(diff.sum() / denom)
    if not losses:
        return y.new_tensor(0.0)
    return torch.stack(losses).mean()


def angular_spacing_loss(y: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-6) -> Tensor:
    """Encourage constant angular displacement between successive latent points."""
    if y.size(1) < 2:
        return y.new_tensor(0.0)
    y_curr, y_next = y[:, :-1, :], y[:, 1:, :]
    theta = angle(y_curr, y_next, eps=eps)
    if mask is not None:
        weights = (mask[:, :-1] * mask[:, 1:]).to(theta.dtype)
        denom = weights.sum().clamp_min(1.0)
        mean_theta = (theta * weights).sum() / denom
        var = ((theta - mean_theta) ** 2 * weights).sum() / denom
    else:
        mean_theta = theta.mean()
        var = ((theta - mean_theta) ** 2).mean()
    return var


__all__ = [
    "angular_spacing_loss",
    "bidirectional_midpoint_loss",
    "global_straightness_loss",
    "local_midpoint_loss",
]
