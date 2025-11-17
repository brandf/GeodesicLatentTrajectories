from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def _promote_dtype(t1: Tensor, t2: Tensor | None = None) -> torch.dtype:
    dtype = t1.dtype if t2 is None else torch.promote_types(t1.dtype, t2.dtype)
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def normalize(x: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize the last dimension of ``x`` to unit length."""
    dtype = x.dtype
    norm = torch.linalg.norm(x.to(torch.float32), dim=-1, keepdim=True)
    norm = norm.clamp_min(eps).to(dtype)
    return x / norm


def safe_dot(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    """Dot product between hyperspherical points with clamping for stability."""
    dtype = _promote_dtype(a, b)
    dot = (a.to(dtype) * b.to(dtype)).sum(dim=-1)
    return dot.clamp(min=-1.0 + eps, max=1.0 - eps)


def angle(a: Tensor, b: Tensor, eps: float = 1e-6) -> Tensor:
    """Return the geodesic distance (angle) between points on the sphere."""
    dots = safe_dot(a, b, eps=eps)
    return torch.arccos(dots)


def _prepare_tau(theta: Tensor, tau: Tensor | float) -> Tensor:
    if isinstance(tau, Tensor):
        return tau.to(dtype=theta.dtype, device=theta.device)
    return torch.as_tensor(tau, device=theta.device, dtype=theta.dtype)


def slerp(u: Tensor, v: Tensor, tau: Tensor | float, eps: float = 1e-6) -> Tensor:
    """
    Batched spherical linear interpolation between normalized vectors ``u`` and ``v``.
    ``tau`` can be broadcast to the shape of ``u`` without the last dimension.
    """
    dtype = _promote_dtype(u, v)
    u = normalize(u.to(dtype))
    v = normalize(v.to(dtype))
    theta = angle(u, v, eps=eps)
    sin_theta = torch.sin(theta)
    tau_tensor = _prepare_tau(theta, tau)
    coeff_u = torch.sin((1.0 - tau_tensor) * theta)
    coeff_v = torch.sin(tau_tensor * theta)
    denom = sin_theta.clamp_min(eps)
    coeff_u = coeff_u / denom
    coeff_v = coeff_v / denom
    out = coeff_u.unsqueeze(-1) * u + coeff_v.unsqueeze(-1) * v
    # Linear fallback for the near-aligned case
    lerp = normalize(((1.0 - tau_tensor).unsqueeze(-1) * u) + (tau_tensor.unsqueeze(-1) * v))
    needs_lerp = sin_theta.abs() < eps
    out = torch.where(needs_lerp.unsqueeze(-1), lerp, out)
    return normalize(out).to(dtype=u.dtype)


def log_map(base: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Log map on the hypersphere from ``base`` to ``target``."""
    dtype = _promote_dtype(base, target)
    base = normalize(base.to(dtype))
    target = normalize(target.to(dtype))
    return log_map_normalized(base, target, eps=eps)


def log_map_normalized(base: Tensor, target: Tensor, eps: float = 1e-6) -> Tensor:
    """Log map assuming ``base`` and ``target`` are already unit vectors."""
    dtype = _promote_dtype(base, target)
    base = base.to(dtype)
    target = target.to(dtype)
    cos_theta = safe_dot(base, target, eps=eps).unsqueeze(-1)
    theta = torch.arccos(cos_theta.clamp(-1.0 + eps, 1.0 - eps))
    sin_theta = torch.sin(theta).clamp_min(eps)
    direction = target - cos_theta * base
    scale = theta / sin_theta
    tangent = scale * direction
    tangent = torch.where(theta <= eps, torch.zeros_like(tangent), tangent)
    return tangent


def exp_map(base: Tensor, tangent: Tensor, eps: float = 1e-6) -> Tensor:
    """Exponential map on the hypersphere from ``base`` along ``tangent``."""
    dtype = _promote_dtype(base, tangent)
    base = normalize(base.to(dtype))
    tangent = tangent.to(dtype)
    return exp_map_normalized(base, tangent, eps=eps)


def exp_map_normalized(base: Tensor, tangent: Tensor, eps: float = 1e-6) -> Tensor:
    """Exponential map assuming ``base`` is already normalized."""
    dtype = _promote_dtype(base, tangent)
    base = base.to(dtype)
    tangent = tangent.to(dtype)
    norm = torch.linalg.norm(tangent, dim=-1, keepdim=True)
    clamp_norm = norm.clamp_min(eps)
    direction = tangent / clamp_norm
    cos = torch.cos(norm)
    sin = torch.sin(norm)
    moved = cos * base + sin * direction
    moved = torch.where(norm <= eps, base, moved)
    return normalize(moved).to(dtype=base.dtype)


__all__ = [
    "angle",
    "exp_map",
    "exp_map_normalized",
    "log_map",
    "log_map_normalized",
    "normalize",
    "safe_dot",
    "slerp",
]
