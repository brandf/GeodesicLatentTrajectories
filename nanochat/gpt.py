"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
"""

import math
from functools import partial
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from glt.config import GLTConfig
from glt.geometry import (
    normalize as glt_normalize,
    log_map_normalized as glt_log_map_norm,
    exp_map_normalized as glt_exp_map_norm,
)
from glt.losses import (
    angular_spacing_loss,
    bidirectional_midpoint_loss,
    global_straightness_loss,
    local_midpoint_loss,
)
from nanochat.common import get_dist_info, print0
from nanochat.muon import Muon, DistMuon
from nanochat.adamw import DistAdamW

@dataclass
class GPTConfig:
    sequence_len: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768


def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last time into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3) # re-assemble
    out = out.to(x.dtype) # ensure input/output dtypes match
    return out

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin) # QK rotary embedding
        q, k = norm(q), norm(k) # QK norm
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2) # make head be batch dim, i.e. (B, T, H, D) -> (B, H, T, D)

        # Apply KV cache: insert current k,v into cache, get the full view so far
        if kv_cache is not None:
            k, v = kv_cache.insert_kv(self.layer_idx, k, v)
        Tq = q.size(2) # number of queries in this forward pass
        Tk = k.size(2) # number of keys/values in total (in the cache + current forward pass)

        # Attention: queries attend to keys/values autoregressively. A few cases to handle:
        enable_gqa = self.n_head != self.n_kv_head # Group Query Attention (GQA): duplicate key/value heads to match query heads if desired
        if kv_cache is None or Tq == Tk:
            # During training (no KV cache), attend as usual with causal attention
            # And even if there is KV cache, we can still use this simple version when Tq == Tk
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)
        elif Tq == 1:
            # During inference but with a single query in this forward pass:
            # The query has to attend to all the keys/values in the cache
            y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)
        else:
            # During inference AND we have a chunk of queries in this forward pass:
            # First, each query attends to all the cached keys/values (i.e. full prefix)
            attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool, device=q.device) # True = keep, False = mask
            prefix_len = Tk - Tq
            if prefix_len > 0: # can't be negative but could be zero
                attn_mask[:, :prefix_len] = True
            # Then, causal attention within this chunk
            attn_mask[:, prefix_len:] = torch.tril(torch.ones((Tq, Tq), dtype=torch.bool, device=q.device))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, enable_gqa=enable_gqa)

        # Re-assemble the heads side by side and project back to residual stream
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, glt_config: Optional[GLTConfig] = None):
        super().__init__()
        self.config = config
        self.glt_config = glt_config if glt_config and glt_config.enabled else None
        self._glt_enabled = self.glt_config is not None
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(config.vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # To support meta device initialization, we init the rotary embeddings here, but it's fake
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # zero out classifier weights
        torch.nn.init.zeros_(self.lm_head.weight)
        # zero out c_proj weights in all blocks
        for block in self.transformer.h:
            torch.nn.init.zeros_(block.mlp.c_proj.weight)
            torch.nn.init.zeros_(block.attn.c_proj.weight)
        # init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        # Cast the embeddings from fp32 to bf16: optim can tolerate it and it saves memory: both in the model and the activations
        if self.transformer.wte.weight.device.type == "cuda":
            self.transformer.wte.to(dtype=torch.bfloat16)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    # TODO: bump base theta more, e.g. 100K is more common more recently
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16() # keep them in bfloat16
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _project_latents(self, hidden: torch.Tensor) -> torch.Tensor:
        # Temporarily disable hyperspherical normalization so GLT path matches baseline
        return hidden

    def _build_offset_predictions(self, latents: torch.Tensor, offsets: Tuple[int, ...]) -> Dict[int, torch.Tensor]:
        """
        Produce extrapolated latents for every offset in `offsets`.
        Offsets > 0 step forward, offsets < 0 step backward, 0 reconstructs the current latent.
        """
        dtype_work = torch.float32 if latents.dtype in (torch.float16, torch.bfloat16) else latents.dtype
        normalized = glt_normalize(latents.to(dtype_work), eps=self.glt_config.norm_eps)
        eps = self.glt_config.clamp_eps
        forward_vec = torch.zeros_like(normalized)
        curr = normalized[:, :-1, :]
        next_lat = normalized[:, 1:, :]
        forward_vec[:, :-1, :] = glt_log_map_norm(curr, next_lat, eps=eps)
        backward_vec = torch.zeros_like(normalized)
        prev_lat = normalized[:, :-1, :]
        backward_curr = normalized[:, 1:, :]
        backward_vec[:, 1:, :] = glt_log_map_norm(backward_curr, prev_lat, eps=eps)
        preds: Dict[int, torch.Tensor] = {}
        for offset in offsets:
            if offset == 0:
                pred = normalized
            elif offset > 0:
                pred = glt_exp_map_norm(normalized, forward_vec * float(offset), eps=eps)
            else:
                pred = glt_exp_map_norm(normalized, backward_vec * float(-offset), eps=eps)
            preds[offset] = pred.to(dtype=latents.dtype)
        return preds

    @staticmethod
    def _select_inference_offset(offsets: Tuple[int, ...]) -> int:
        positives = sorted([o for o in offsets if o > 0])
        if positives:
            return positives[0]
        if 0 in offsets:
            return 0
        return offsets[0]

    def _apply_head(self, latents: torch.Tensor) -> torch.Tensor:
        softcap = 15
        logits = self.lm_head(latents)
        return softcap * torch.tanh(logits / softcap)

    def _compute_multi_ce_loss(
        self,
        offset_predictions: Dict[int, torch.Tensor],
        targets: torch.Tensor,
        reduction: str,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        offsets = self.glt_config.ce_offsets
        weights = self.glt_config.ce_offset_weights
        if weights is None:
            weights_list = [1.0 / len(offsets)] * len(offsets)
        else:
            weights_list = list(weights)
        ce_values: Dict[int, torch.Tensor] = {}
        total = 0.0
        for offset, weight in zip(offsets, weights_list):
            preds = offset_predictions[offset]
            if offset > 0:
                pred_slice = preds[:, :-offset, :]
                target_slice = targets[:, offset:]
            elif offset < 0:
                k = -offset
                pred_slice = preds[:, k:, :]
                target_slice = targets[:, :-k]
            else:
                pred_slice = preds
                target_slice = targets
            if pred_slice.numel() == 0:
                continue
            logits = self._apply_head(pred_slice)
            logits = logits.float()
            ce = F.cross_entropy(logits.view(-1, logits.size(-1)), target_slice.reshape(-1), ignore_index=-1, reduction=reduction)
            ce_values[offset] = ce.detach()
            total = total + weight * ce
        return total, ce_values

    def _build_valid_mask(self, targets: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if targets is None:
            return None
        mask = targets >= 0
        if torch.all(mask):
            return None
        return mask

    def _compute_glt_losses(self, latents: torch.Tensor, mask: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert self.glt_config is not None
        cfg = self.glt_config
        eps = cfg.clamp_eps
        losses = {
            "local": local_midpoint_loss(latents, mask=mask, eps=eps),
            "global": global_straightness_loss(
                latents,
                mask=mask,
                num_spans=cfg.global_num_spans,
                span_len=cfg.global_span_len,
                eps=eps,
            ),
            "angle": angular_spacing_loss(latents, mask=mask, eps=eps),
            "bi": bidirectional_midpoint_loss(latents, mask=mask, eps=eps),
        }
        return losses

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """ Return the estimated FLOPs per token for the model. Ref: https://arxiv.org/abs/2204.02311 """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_embedding = self.transformer.wte.weight.numel()
        l, h, q, t = self.config.n_layer, self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
        return num_flops_per_token

    def setup_optimizers(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()
        # Separate out all parameters into 3 groups (matrix, embedding, lm_head)
        matrix_params = list(self.transformer.h.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params)
        # Create the AdamW optimizer for the embedding and lm_head
        # Scale the LR for the AdamW parameters by ∝1/√dmodel (having tuned the LRs for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        if rank == 0:
            print(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")
        adam_groups = [
            dict(params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale),
            dict(params=embedding_params, lr=embedding_lr * dmodel_lr_scale),
        ]
        adamw_kwargs = dict(betas=(0.8, 0.95), eps=1e-10, weight_decay=weight_decay)
        AdamWFactory = DistAdamW if ddp else partial(torch.optim.AdamW, fused=True)
        adamw_optimizer = AdamWFactory(adam_groups, **adamw_kwargs)
        # Create the Muon optimizer for the linear layers
        muon_kwargs = dict(lr=matrix_lr, momentum=0.95)
        MuonFactory = DistMuon if ddp else Muon
        muon_optimizer = MuonFactory(matrix_params, **muon_kwargs)
        # Combine them the two optimizers into one list
        optimizers = [adamw_optimizer, muon_optimizer]
        for opt in optimizers:
            for group in opt.param_groups:
                group["initial_lr"] = group["lr"]
        return optimizers

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', return_loss_breakdown=False, return_visuals=False):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Forward the trunk of the Transformer
        x = self.transformer.wte(idx)
        x = norm(x)
        for block in self.transformer.h:
            x = block(x, cos_sin, kv_cache)
        x = norm(x)
        pre_norm = x
        latents = self._project_latents(x)

        # Forward the lm_head (compute logits)
        ce_offsets = self.glt_config.ce_offsets if self._glt_enabled else (1,)
        offset_predictions: Dict[int, torch.Tensor] = {}
        if self._glt_enabled:
            offset_predictions = self._build_offset_predictions(latents, ce_offsets)
            inference_offset = self._select_inference_offset(ce_offsets)
            logits_latents = offset_predictions[inference_offset]
        else:
            logits_latents = latents
        logits = self._apply_head(logits_latents) # logits softcap
        if targets is None:
            return logits
        logits = logits.float() # use tf32/fp32 for logits
        ce_components: Dict[int, torch.Tensor] = {}
        if self._glt_enabled:
            ce_loss, ce_components = self._compute_multi_ce_loss(offset_predictions, targets, loss_reduction)
        else:
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            ce_components[1] = ce_loss.detach()
        if loss_reduction == 'none':
            return ce_loss
        mask = self._build_valid_mask(targets)
        total_loss = ce_loss
        breakdown = {"ce": ce_loss.detach()} if return_loss_breakdown else None
        if return_loss_breakdown:
            for offset, value in ce_components.items():
                breakdown[f"ce/{offset:+d}"] = value.detach()
        if self._glt_enabled and self.glt_config.enable_geom_losses:
            latents_for_loss = latents.float()
            glt_losses = self._compute_glt_losses(latents_for_loss, mask)
            cfg = self.glt_config
            total_loss = (
                cfg.lambda_ce * ce_loss
                + cfg.lambda_local * glt_losses["local"]
                + cfg.lambda_global * glt_losses["global"]
                + cfg.lambda_angle * glt_losses["angle"]
                + cfg.lambda_bi * glt_losses["bi"]
            )
            if breakdown is not None:
                for key, value in glt_losses.items():
                    breakdown[f"glt/{key}"] = value.detach()
        extras = {}
        if breakdown is not None:
            extras["loss_breakdown"] = breakdown
        if return_visuals and targets is not None:
            visuals = {
                "latents": latents.detach().to(device="cpu", dtype=torch.float32),
                "pre_norm": pre_norm.detach().to(device="cpu", dtype=torch.float32),
                "mask": None if mask is None else mask.detach().to("cpu"),
            }
            extras["visuals"] = visuals
        if extras:
            return total_loss, extras
        return total_loss

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
