from __future__ import annotations

import torch
from torch import Tensor

from .geometry import angle, exp_map, log_map


def extrapolate_next_latent(y_seq: Tensor, use_mean_step: bool = True) -> Tensor:
    """
    Given a single latent trajectory ``y_seq`` with shape ``(T, D)`` or ``(1, T, D)``,
    extrapolate the next latent on the hypersphere via geodesic continuation.
    """
    if y_seq.dim() == 3:
        y_seq = y_seq[0]
    if y_seq.dim() != 2 or y_seq.size(0) < 2:
        raise ValueError("Need at least two latent points to extrapolate")
    dtype = torch.float32 if y_seq.dtype in (torch.float16, torch.bfloat16) else y_seq.dtype
    seq = y_seq.to(dtype)
    y_prev = seq[-2].unsqueeze(0)
    y_curr = seq[-1].unsqueeze(0)
    v_back = log_map(y_curr, y_prev)
    v_forward = -v_back
    if use_mean_step and seq.size(0) > 2:
        thetas = []
        for i in range(seq.size(0) - 1):
            thetas.append(angle(seq[i : i + 1], seq[i + 1 : i + 2]))
        mean_theta = torch.stack(thetas).mean()
        curr_theta = torch.linalg.norm(v_forward, dim=-1, keepdim=True).clamp_min(1e-6)
        v_forward = v_forward * (mean_theta / curr_theta)
    y_next = exp_map(y_curr, v_forward)[0]
    return y_next.to(dtype=y_seq.dtype)


__all__ = ["extrapolate_next_latent"]
