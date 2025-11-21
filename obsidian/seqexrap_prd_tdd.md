# Sequence Extrapolation (SE) – PRD + TDD

## Intent & Scope
- Treat existing GLT experiments as historical context only; keep the nanochat base stack and run10 profiles intact.
- Add a new SE pathway that can be toggled independently via `--se`, using the same model dimensions as the main stack and untied embeddings.
- Goal: minimal inductive bias beyond "untangle -> extrapolate" with a small transformer head that learns latent velocities.

## Requirements
- CLI: expose SE through `run10.sh --se` (mutually exclusive with `--glt`) and forward to `scripts/base_train.py` flags `--enable_se`, `--se_extrap_len` (default 8), `--se_extrap_layers` (default 3), `--se_predict_horizon` (default 2), `--se_velocity_softcap` (default 0 to disable), `--se_loss_weight` (default 1.0).
- Config: `seqexrap.config.SequenceExtrapolationConfig` persists in checkpoints under `se_config` and is logged to wandb.
- Model: use the main transformer to produce the "untangled sequence" (normed hidden states). A small extrapolation head (same Block architecture as the trunk) runs on a local causal window (`se_extrap_len`) to predict latent velocities per token.
- Extrapolation: stack predictions for horizons up to `se_predict_horizon`. For each step `h`, add the predicted velocity to the current latent to form an extrapolated point, disembed via the usual LM head, and compute CE vs. target `+h` (ignoring tail positions without ground truth). Aggregate by mean over horizons and scale by `se_loss_weight`.
- Logging: log `se/loss`, `loss_components/ce`, `loss_components/se/ce@+h`, `loss_components/se/velocity_norm_mean`, `loss_components/se/velocity_norm_max`, `loss_components/se/valid_tokens`, plus standard train metrics. Console prints include `se[...]` summaries when enabled.
- Optimization: SE parameters share the matrix optimizer group (Muon) with the main blocks; embeddings/unembedding stay on AdamW. Zero-init the SE head projections (including velocity projector) to start from "no extrapolation".

## Design Details
- Files: `nanochat/gpt.py` (SE forward path + optimizer grouping), `seqexrap/config.py`, `seqexrap/extrapolator.py`, `run10.sh`, `scripts/base_train.py`.
- Extrapolation head: `SequenceExtrapolator` builds its own `Block` stack (same RMSNorm + QK norm + rotary attention). Attention uses a boolean mask enforcing causal slices limited to `se_extrap_len` tokens. Velocity projection is soft-capped when `se_velocity_softcap > 0` (`cap * tanh(delta / cap)`).
- Forward path (`nanochat/gpt.py`):
  - If SE enabled, run `_run_sequence_extrapolation` and skip GLT/baseline CE path.
  - Horizon loop maintains `current` latents; after each velocity add, compute logits for that horizon and CE against `targets[:, h-1:]`, slicing predictions to match length. Horizon losses are averaged; a missing `targets` short-circuits with first-horizon logits for inference.
  - Visuals (latents, pre_norm, mask) still emit when requested.
- Data: uses existing `(x, y)` from the tokenizer loader; for horizon `h`, the last `h-1` positions are naturally dropped via slicing, so no dataloader changes are needed.
- Safety: GLT and SE cannot be enabled together (checked in `run10.sh` and `scripts/base_train.py`).

## TDD / Validation Plan
- Unit-style checks (add under `tests/` when ready):
  - SE mask correctness: for a small `T`, verify the attention mask only keeps indices `i-j < se_extrap_len` and `i >= j`.
  - Horizon slicing: ensure `_run_sequence_extrapolation` returns CE tensors matching expected valid lengths and ignores missing tail tokens.
  - Softcap: with a large crafted velocity, confirm output is bounded by `se_velocity_softcap` and gradients still flow.
  - Optimizer grouping: assert all SE parameters appear in the Muon group (counts match `model.parameters()`).
- Training/integration checks:
  - `bash run10.sh --gpu 5090 --se --run=test_se` runs without GLT flags; wandb shows `se/loss` trending and per-horizon components in `loss_components/se/*`.
  - Console line includes `se[...]` summaries; `loss_components/se/valid_tokens` matches `batch*T - (h-1)` for horizon `h`.
  - Check checkpoints carry `se_config` and resume with the same settings.

## Open Questions / Next Steps
- Tune weighting across horizons (currently uniform mean); consider token-count–weighted averaging or schedule for longer horizons.
- Evaluate whether additional regularization on velocity magnitude or curriculum on `se_predict_horizon` improves stability.
- Add targeted visualizations (e.g., velocity norm histograms) if wandb signals become hard to interpret.
