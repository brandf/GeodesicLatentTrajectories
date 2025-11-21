from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class SequenceExtrapolationConfig:
    """Hyperparameters for Sequence Extrapolation (SE)."""

    enabled: bool = False
    extrapolation_length: int = 8
    extrapolation_layers: int = 3
    predict_horizon: int = 2
    velocity_softcap: float = 0.0
    loss_weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def seqex_config_from_dict(data: Optional[Dict[str, Any]]) -> Optional[SequenceExtrapolationConfig]:
    if not data:
        return None
    return SequenceExtrapolationConfig(**data)
