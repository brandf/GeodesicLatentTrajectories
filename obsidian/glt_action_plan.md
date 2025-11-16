# GLT Training Action Plan

## Current Symptoms
- `train/loss` for GLT trails baseline by ~1.5–2 points through step ~300 (still true even after scaling GLT penalties).
- `train/grad_norm` rebounds from ~0.05 to 0.35+ after the first 50 steps instead of settling like baseline; sweeping lambda scales (0.1/0.3/0.5) drops grad norm substantially vs. baseline but the upward trend persists, especially at 0.5.
- Scalar GLT diagnostics (curvature, angle variance) rise monotonically for all lambda scales tested; none of the sweeps produced the desired downward trend.

## Hypotheses
1. **Bug in loss masking or weighting** – masked tokens or padding may leak into GLT losses, inflating curvature/angle statistics and gradients.
2. **Incorrect normalization/inference path** – latents might be renormalized twice or not at all inside GLT losses, skewing magnitudes.
3. **Loss-weight mismatch** – default `lambda_*` values borrowed from PRD may be too large for $10 runs, overwhelming CE gradients.
4. **Optimizer interaction** – Muon/Adam schedule tuned for baseline may need different warmup or grad clipping when extra penalties are active.
5. **Visualization/metric bug** – diagnostics could be logging pre-normalized values or mis-scaling angles, creating a false alarm.

## Debugging Steps
0. **Basic sanity checks before more sweeps**
   - Re-run $10 baseline and GLT with `glt_lambda_scale=0` to confirm the code path is identical to baseline when penalties are disabled.
   - Capture a single batch, compute GLT losses manually, and verify they drop to ~0 on synthetic geodesic data.
   - Ensure the model outputs identical logits when `enable_glt=False` vs. `enable_glt=True` but `lambda_scale=0`; any discrepancy would signal bugs in normalization/inference wiring.
   - Check that `loss_components/glt/*` vanish when masks are empty or when sequences are length <3.
1. **Unit tests for GLT losses**
   - Feed synthetic sequences that follow perfect SLERP arcs; verify `local/global/angle/bi` all report near-zero.
   - Add tests for masked timesteps to ensure losses drop out entirely where they should.
2. **Instrumentation in training**
   - Log per-loss magnitudes (`loss_components/ce`, `loss_components/glt/*`) every step to check relative scale.
   - Capture a few batches’ `mask` tensors and ensure they’re dense (no accidental zeros).
3. **Gradient sanity**
   - Temporarily disable individual GLT terms to see which one drives the rising grad norm.
   - Compare grad norms with `enable_glt=False` but still running visualization to confirm instrumentation isn’t perturbing training.
4. **Hyperparameter sweep**
   - Sweeps over `glt_lambda_scale` in {0.1, 0.3, 0.5} show little effect on CE loss but do reduce grad norm at lower scales; continue exploring <0.1 and >0.5 along with schedule ramps to find a sweet spot.
   - Experiment with re-enabling warmup (`warmup_ratio > 0`) to soften the initial shock from GLT penalties—especially important for higher lambda scales that push grad norm up.
5. **Visualization validation**
   - Run the scalar logging scripts offline on a saved batch to confirm curvature/angle computations align with the math appendix.
   - Add assertions that `pre_norm_radius_mean` stays close to √d; large drift would imply projection issues.

## Confidence-Building Experiments
1. **A/B training**
   - Run short ($10) jobs with: baseline, GLT (current), GLT with reduced lambdas, GLT with only local loss. Overlay W&B charts to isolate regressions.
2. **Eval parity**
   - Ensure `scripts.base_eval` uses the same GLT-normalized latents; compare CE-only validation to baseline to prove inference path is correct.
3. **Loss-surface probes**
   - Record gradient contributions (`∂L_ce`, `∂L_glt`) via backward hooks to see if GLT dominates specific layers; sweeps imply the relative contribution stays high even at 0.1 scale.
4. **Reproducible notebook**
   - Capture a single batch, compute all GLT metrics/losses in an interactive script, and share with the team for peer verification.

## Open Questions
- Are GLT losses computed before or after dropout/LayerNorm? Misordered ops could explain curvature growth.
- Do we need curriculum scheduling (e.g., ramping `lambda_*` from 0→target) to avoid early training shocks?
- Should geodesic constraints be applied to a subset of layers instead of the final hidden state?

## Next Steps
0. Latents are now normalized only inside the GLT loss computation; rerun the lambda-scale sweeps with this change to see if curvature improves without wrecking CE.
1. Write the unit tests (loss correctness & masking) under `tests/test_glt_losses.py`.
2. Add per-component logging and optional lambda scaling flags to `scripts.base_train.py`. (Done: sweeps already logging `loss_components/glt/*`; keep expanding instrumentation to gradient contributions.)
3. Run broader factor sweeps (lambda scale, warmup ratio, optional curriculum) and track curvature/angle metrics alongside grad norm to see if any host combination reverses the rising trend.
4. Reconvene after inspecting results to decide whether we’re hitting a bug or fundamental assumption limits.
