from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple


@dataclass
class GLTConfig:
    """Hyperparameters toggling Geodesic Latent Trajectories."""

    enabled: bool = False
    norm_eps: float = 1e-8
    clamp_eps: float = 1e-6

    lambda_ce: float = 1.0
    lambda_local: float = 0.3
    lambda_global: float = 0.05
    lambda_angle: float = 0.1
    lambda_bi: float = 0.1

    ce_offsets: Tuple[int, ...] = (-1, 0, 1)
    ce_offset_weights: Optional[Tuple[float, ...]] = None
    ce_offset_chunk_size: int = 1
    enable_geom_losses: bool = True

    global_num_spans: int = 1
    global_span_len: int = 256

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def glt_config_from_dict(data: Optional[Dict[str, Any]]) -> Optional[GLTConfig]:
    if not data:
        return None
    return GLTConfig(**data)
