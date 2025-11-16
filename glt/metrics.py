from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor

from .geometry import normalize as glt_normalize, slerp, angle as geometry_angle


def curvature_values(y: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-6) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Return curvature (B, T-2) computed from normalized latents.
    """
    if y.size(1) < 3:
        return y.new_zeros((y.size(0), 0))
    y_prev, y_mid, y_next = y[:, :-2, :], y[:, 1:-1, :], y[:, 2:, :]
    tau = y_prev.new_full(y_prev.shape[:-1], 0.5)
    midpoint = slerp(y_prev, y_next, tau, eps=eps)
    curv = torch.linalg.norm(y_mid - midpoint, dim=-1)
    valid = None
    if mask is not None:
        valid = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
    return curv, valid


def angular_steps(y: Tensor, mask: Optional[Tensor] = None, eps: float = 1e-6) -> Tuple[Tensor, Optional[Tensor]]:
    if y.size(1) < 2:
        return y.new_zeros((y.size(0), 0))
    ang = geometry_angle(y[:, :-1, :], y[:, 1:, :], eps=eps)
    valid = None
    if mask is not None:
        valid = mask[:, :-1] & mask[:, 1:]
    return ang, valid


def pre_norm_radius(h: Tensor, mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
    return torch.linalg.norm(h, dim=-1), mask


def cosine_similarity_matrix(seq: Tensor) -> Tensor:
    seq = glt_normalize(seq)
    return torch.matmul(seq, seq.transpose(0, 1))


def local_directions(seq: Tensor) -> Tensor:
    if seq.size(0) < 2:
        return seq.new_zeros((0, seq.size(1)))
    return seq[1:] - seq[:-1]


def flatten_valid(values: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    flat = values.reshape(-1)
    if mask is None:
        return flat
    mask_flat = mask.reshape(-1)
    if mask_flat.dtype != torch.bool:
        mask_flat = mask_flat > 0.5
    return flat[mask_flat]


def prepare_mask(mask: Optional[Tensor], target_shape: Tuple[int, ...]) -> Optional[Tensor]:
    if mask is None:
        return None
    return mask


def ensure_normalized(y: Tensor) -> Tensor:
    return glt_normalize(y)
