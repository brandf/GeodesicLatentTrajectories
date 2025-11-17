# GLT Multi-Offset Chunking Plan

## Current Issue
- Offsets `[1]` and `[0,1]` fit, but `[−1,0,1]` still OOMs even with `--glt_ce_chunk` > 1.
- Current chunking uses `tensor_split`, which keeps references to the full buffer; peak memory remains tied to full `(B·T, vocab)` activations.
- Goal: support ≥3 offsets at run10 batch size without resorting to checkpointing or disabling gradients.

## Plan
1. **Instrument memory** inside `_compute_multi_ce_loss` to confirm which tensors dominate (normalized latents, tangents, logits chunks).
2. **Rework chunking** to iterate over token slices sequentially (no `tensor_split` views). Use a generator that slices the flattened `(tokens, D)` buffer and frees each slice immediately.
3. **Process offsets chunk-by-chunk.** For each offset, loop over token slices sequentially *without creating tensor_split views* (use explicit begin/end slicing), run the head, compute CE (`sum`/`mean`), and accumulate. Ensure each chunk is free before the next chunk is created. Keep `reduction='none'` behavior unchanged.
4. **Reuse normalized latents everywhere** (already in place) to avoid redundant copies when feeding GLT penalties.
5. **Validate** by rerunning `[−1,0,1]` with increasing `--glt_ce_chunk` values to ensure peak memory scales inversely with chunk count; compare CUDA max memory snapshots to confirm reductions.

## Notes
- Checkpointing auxiliary offsets or exposing additional CLI switches is a last resort if serial chunking cannot control memory on its own.
