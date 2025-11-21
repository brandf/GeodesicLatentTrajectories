from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceExtrapolator(nn.Module):
    """
    Lightweight transformer head that predicts latent-space velocities from a local
    window of untangled states. Blocks are supplied by the caller to match the main
    stack architecture (same dims/activations).
    """

    def __init__(
        self,
        block_builder: Callable[[int], nn.Module],
        num_layers: int,
        window: int,
        model_dim: int,
        velocity_softcap: float = 0.0,
        norm_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(block_builder(i) for i in range(num_layers))
        self.delta = nn.Linear(model_dim, model_dim, bias=False)
        torch.nn.init.zeros_(self.delta.weight)
        self.window = max(1, int(window))
        self.velocity_softcap = float(velocity_softcap)
        self.norm_fn = norm_fn or (lambda x: F.rms_norm(x, (x.size(-1),)))

    def _build_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        positions = torch.arange(seq_len, device=device)
        diff = positions[:, None] - positions[None, :]
        return (diff >= 0) & (diff < self.window)

    def forward(self, latents: torch.Tensor, cos_sin) -> torch.Tensor:
        attn_mask = self._build_mask(latents.size(1), latents.device)
        x = latents
        for block in self.blocks:
            x = block(x, cos_sin, kv_cache=None, attn_mask=attn_mask)
        x = self.norm_fn(x)
        delta = self.delta(x)
        if self.velocity_softcap > 0:
            cap = self.velocity_softcap
            delta = cap * torch.tanh(delta / cap)
        return delta
