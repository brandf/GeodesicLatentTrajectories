"""
Train model. Run as:

python base_train.py

or distributed as:

torchrun --nproc_per_node=8 base_train.py

If you are only on CPU/Macbook, you'll want to train a much much smaller LLM. Example:
python -m scripts.base_train --depth=4 --max_seq_len=512 --device_batch_size=1 --eval_tokens=512 --core_metric_every=-1 --total_batch_size=512 --num_iterations=20
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import time
from contextlib import nullcontext

import wandb
import torch

from glt.config import GLTConfig
from glt.viz_config import VizConfig
from glt.logging import VizBatch, maybe_log_visualizations
from nanochat.gpt import GPT, GPTConfig
from nanochat.dataloader import tokenizing_distributed_data_loader, tokenizing_distributed_data_loader_with_state
from nanochat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, get_base_dir, autodetect_device_type
from nanochat.tokenizer import get_tokenizer, get_token_bytes
from nanochat.checkpoint_manager import save_checkpoint, load_checkpoint
from nanochat.loss_eval import evaluate_bpb
from nanochat.engine import Engine
from scripts.base_eval import evaluate_model
print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy" # wandb run name default ("dummy" is special - we won't log to wandb)
# Runtime
device_type = "" # cuda|cpu|mps (empty => autodetect good device type default, in order: CUDA > MPS > CPU)
# Model architecture
depth = 20 # the depth of the Transformer model to train, rest of the kwargs are derived
max_seq_len = 2048 # max context length
# Training horizon. Only one of these 3 will be used, in this order of precedence.
num_iterations = -1 # explicit number of steps of the optimization (-1 = disable)
target_flops = -1.0 # calculate num_iterations to reach target_flops. Useful for scaling laws experiments (-1 = disable)
target_param_data_ratio = 20 # calculate num_iterations to maintain fixed data:param ratio (Chinchilla=20) (-1 = disable)
# Optimization
device_batch_size = 32 # per-device batch size (set to not OOM)
total_batch_size = 524288 # total desired batch size, in #tokens
embedding_lr = 0.2 # learning rate for the embedding parameters (Adam)
unembedding_lr = 0.004 # learning rate for the unembedding parameters (Adam)
weight_decay = 0.0 # weight decay for the embedding/unembedding parameters (Adam)
matrix_lr = 0.02 # learning rate for the matrix parameters (Muon)
grad_clip = 1.0 # gradient clipping value (0.0 = disabled)
warmup_ratio = 0.0 # ratio of iterations for LR warmup
warmdown_ratio = 0.2 # ratio of iterations for LR warmdown
final_lr_frac = 0.0 # final LR is this fraction of the initial LR
resume_from_step = -1 # resume training from this step of the optimization (-1 = disable)
# Evaluation
eval_every = 250 # every how many steps to evaluate the model for val bpb
eval_tokens = 20*524288 # number of tokens to evaluate val loss on
core_metric_every = 2000 # every how many steps to evaluate the core metric (-1 = disable)
core_metric_max_per_task = 500 # examples per task in estimating the core metric
sample_every = 2000 # every how many steps to sample from the model
save_every = -1 # every how many steps to save model checkpoints (-1 = disable, and save only at the end of the run)
# Output
model_tag = "" # optionally override the model tag for the output checkpoint directory name
log_every = 100 # how often to log train metrics (CE loss) to wandb
# Geodesic Latent Trajectories (optional)
enable_glt = False
glt_norm_eps = 1e-8
glt_clamp_eps = 1e-6
glt_lambda_ce = 1.0
glt_lambda_local = 0.3
glt_lambda_global = 0.05
glt_lambda_angle = 0.1
glt_lambda_bi = 0.1
glt_lambda_scale = 0.3 # downscale GLT penalties for small-budget runs
glt_ce_offsets = [-1, 0, 1]
glt_ce_offset_weights = None
glt_enable_geom_losses = True
glt_ce_chunk_size = 2048
glt_global_num_spans = 1
glt_global_span_len = 256
# Visualization
viz_enabled = False
viz_scalar_every = 10
viz_hist_every = 500
viz_image_every = 1000
viz_sequence_index = 0
# now allow CLI to override the settings via the configurator lol
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('nanochat', 'configurator.py')).read()) # overrides from command line or config file
user_config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb logging init
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat", name=run, config=user_config)

# Tokenizer will be useful for evaluation, also we need the vocab size
tokenizer = get_tokenizer()
token_bytes = get_token_bytes(device=device)
vocab_size = tokenizer.get_vocab_size()
print0(f"Vocab size: {vocab_size:,}")

# Model kwargs are derived from the desired depth of the model
num_layers = depth
model_dim = depth * 64 # aspect ratio 64 (usually this is varied from 64 -> 128 as model size increases)
num_heads = max(1, (model_dim + 127) // 128) # head dim 128 (the division here is ceil div)
num_kv_heads = num_heads # default is 1:1 GQA (Group Query Attention) ratio (i.e. GQA is disabled)
print0(f"num_layers: {num_layers}")
print0(f"model_dim: {model_dim}")
print0(f"num_heads: {num_heads}")
print0(f"num_kv_heads: {num_kv_heads}")

# Optimizer / data / training length related hyperparameters
# figure out the needed gradient accumulation to reach the desired total batch size
tokens_per_fwdbwd = device_batch_size * max_seq_len # tokens per iteration for a single rank
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size # total tokens per iteration for all ranks
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd
print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Tokens / micro-batch: {world_tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Checkpoint / GLT configuration
base_dir = get_base_dir()
output_dirname = model_tag if model_tag else f"d{depth}" # e.g. d12
checkpoint_dir = os.path.join(base_dir, "base_checkpoints", output_dirname)
resuming = resume_from_step != -1
model_data = None
optimizer_data = None
meta_data = None
if resuming:
    print0(f"Resuming optimization from step {resume_from_step}")
    model_data, optimizer_data, meta_data = load_checkpoint(checkpoint_dir, resume_from_step, device, load_optimizer=True, rank=ddp_rank)

glt_config_kwargs = None
if enable_glt:
    scaled_lambda_local = glt_lambda_local * glt_lambda_scale
    scaled_lambda_global = glt_lambda_global * glt_lambda_scale
    scaled_lambda_angle = glt_lambda_angle * glt_lambda_scale
    scaled_lambda_bi = glt_lambda_bi * glt_lambda_scale
    ce_offsets_tuple = tuple(int(o) for o in glt_ce_offsets)
    ce_offset_weights_tuple = None
    if glt_ce_offset_weights is not None:
        assert len(glt_ce_offset_weights) == len(glt_ce_offsets), "glt_ce_offset_weights length must match glt_ce_offsets"
        ce_offset_weights_tuple = tuple(float(w) for w in glt_ce_offset_weights)
    glt_config_kwargs = GLTConfig(
        enabled=True,
        norm_eps=glt_norm_eps,
        clamp_eps=glt_clamp_eps,
        lambda_ce=glt_lambda_ce,
        lambda_local=scaled_lambda_local,
        lambda_global=scaled_lambda_global,
        lambda_angle=scaled_lambda_angle,
        lambda_bi=scaled_lambda_bi,
        ce_offsets=ce_offsets_tuple,
        ce_offset_weights=ce_offset_weights_tuple,
        ce_offset_chunk_size=int(glt_ce_chunk_size),
        enable_geom_losses=glt_enable_geom_losses,
        global_num_spans=glt_global_num_spans,
        global_span_len=glt_global_span_len,
    ).to_dict()
if resuming and meta_data is not None:
    saved_glt = meta_data.get("glt_config", None)
    if saved_glt is None and enable_glt:
        print0("[GLT] Checkpoint does not contain GLT metadata; disabling GLT for resume.")
    glt_config_kwargs = saved_glt
glt_config = GLTConfig(**glt_config_kwargs) if glt_config_kwargs else None
if glt_config:
    print0(f"[GLT] Enabled with config: {glt_config_kwargs}")
else:
    print0("[GLT] Disabled")
viz_cfg = VizConfig(
    enabled=viz_enabled,
    scalar_every=viz_scalar_every,
    hist_every=viz_hist_every,
    image_every=viz_image_every,
    sequence_index=viz_sequence_index,
)
if viz_cfg.enabled:
    print0(f"[Viz] Enabled with scalar_every={viz_cfg.scalar_every}, hist_every={viz_cfg.hist_every}, image_every={viz_cfg.image_every}")
user_config["enable_glt"] = bool(glt_config)
user_config["glt_norm_eps"] = glt_config.norm_eps if glt_config else glt_norm_eps
user_config["glt_clamp_eps"] = glt_config.clamp_eps if glt_config else glt_clamp_eps
user_config["glt_lambda_ce"] = glt_config.lambda_ce if glt_config else glt_lambda_ce
user_config["glt_lambda_local"] = glt_config.lambda_local if glt_config else glt_lambda_local
user_config["glt_lambda_global"] = glt_config.lambda_global if glt_config else glt_lambda_global
user_config["glt_lambda_angle"] = glt_config.lambda_angle if glt_config else glt_lambda_angle
user_config["glt_lambda_bi"] = glt_config.lambda_bi if glt_config else glt_lambda_bi
user_config["glt_lambda_scale"] = glt_lambda_scale
user_config["glt_ce_offsets"] = list(glt_config.ce_offsets) if glt_config else glt_ce_offsets
user_config["glt_ce_offset_weights"] = list(glt_config.ce_offset_weights) if glt_config and glt_config.ce_offset_weights else glt_ce_offset_weights
user_config["glt_ce_chunk_size"] = glt_config.ce_offset_chunk_size if glt_config else glt_ce_chunk_size
user_config["glt_enable_geom_losses"] = glt_config.enable_geom_losses if glt_config else glt_enable_geom_losses
user_config["glt_global_num_spans"] = glt_config.global_num_spans if glt_config else glt_global_num_spans
user_config["glt_global_span_len"] = glt_config.global_span_len if glt_config else glt_global_span_len
user_config["viz_enabled"] = viz_enabled
user_config["viz_scalar_every"] = viz_scalar_every
user_config["viz_hist_every"] = viz_hist_every
user_config["viz_image_every"] = viz_image_every
user_config["viz_sequence_index"] = viz_sequence_index
if not use_dummy_wandb:
    wandb_run.config.update(
        {k: user_config[k] for k in [
            "enable_glt",
            "glt_norm_eps",
            "glt_clamp_eps",
            "glt_lambda_ce",
            "glt_lambda_local",
            "glt_lambda_global",
            "glt_lambda_angle",
            "glt_lambda_bi",
            "glt_ce_offsets",
            "glt_ce_offset_weights",
            "glt_ce_chunk_size",
            "glt_enable_geom_losses",
            "glt_global_num_spans",
            "glt_global_span_len",
            "viz_enabled",
            "viz_scalar_every",
            "viz_hist_every",
            "viz_image_every",
            "viz_sequence_index",
        ]},
        allow_val_change=True,
    )

# -----------------------------------------------------------------------------
# Initialize the Model

# Create a new model with random weights
model_config_kwargs = dict(sequence_len=max_seq_len, vocab_size=vocab_size, n_layer=num_layers, n_head=num_heads, n_kv_head=num_kv_heads, n_embd=model_dim)
with torch.device("meta"):
    model_config = GPTConfig(**model_config_kwargs)
    model = GPT(model_config, glt_config=glt_config)
model.to_empty(device=device)
model.init_weights()

if resuming:
    model.load_state_dict(model_data, strict=True, assign=True)
    del model_data # free up this memory after the copy

orig_model = model # original, uncompiled model, for saving raw model state_dict and for inference/evaluation (because the shapes may change shape)
model = torch.compile(model, dynamic=False) # the inputs to model will never change shape so dynamic=False is safe
num_params = sum(p.numel() for p in model.parameters())
print0(f"Number of parameters: {num_params:,}")
num_flops_per_token = model.estimate_flops()
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations. Either it is given, or from target flops, or from target data:param ratio (in that order)
assert num_iterations > 0 or target_param_data_ratio > 0 or target_flops > 0
if num_iterations > 0:
    print0(f"Using user-provided number of iterations: {num_iterations:,}")
elif target_flops > 0:
    # calculate the number of iterations from the target flops
    num_iterations = round(target_flops / (num_flops_per_token * total_batch_size))
    print0(f"Calculated number of iterations from target FLOPs: {num_iterations:,}")
elif target_param_data_ratio > 0:
    # calculate the number of iterations from the target param data ratio
    target_tokens = target_param_data_ratio * num_params
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated number of iterations from target data:param ratio: {num_iterations:,}")
else:
    raise ValueError("No training horizon specified")
total_tokens = total_batch_size * num_iterations
print0(f"Total number of training tokens: {total_tokens:,}")
print0(f"Tokens : Params ratio: {total_batch_size * num_iterations / num_params:.2f}") # Chinchilla is ~20
print0(f"Total training FLOPs estimate: {num_flops_per_token * total_tokens:e}")

# -----------------------------------------------------------------------------
# Initialize the Optimizer (Muon for Linear layers, AdamW for embedding and lm_head)
optimizers = model.setup_optimizers(unembedding_lr=unembedding_lr, embedding_lr=embedding_lr, matrix_lr=matrix_lr, weight_decay=weight_decay)
adamw_optimizer, muon_optimizer = optimizers

if resuming:
    for opt, dat in zip(optimizers, optimizer_data):
        opt.load_state_dict(dat)
    del optimizer_data # free up the memory

# -----------------------------------------------------------------------------
# Initialize the DataLoaders for train/val
tokens_dir = os.path.join(base_dir, "tokenized_data")
dataloader_resume_state_dict = None if not resuming else meta_data.get("dataloader_state_dict")
train_loader = tokenizing_distributed_data_loader_with_state(device_batch_size, max_seq_len, split="train", device=device, resume_state_dict=dataloader_resume_state_dict)
build_val_loader = lambda: tokenizing_distributed_data_loader(device_batch_size, max_seq_len, split="val", device=device)
x, y, dataloader_state_dict = next(train_loader) # kick off load of the very first batch of data

# -----------------------------------------------------------------------------
# Set up hyperparameter schedulers

# Learning rate scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# Momentum scheduler for Muon optimizer
def get_muon_momentum(it):
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum

# -----------------------------------------------------------------------------
# Loop state (variables updated by the training loop)

if not resuming:
    step = 0
    min_val_bpb = float("inf")
    smooth_total_loss = 0 # EMA of total (optimization) loss
    smooth_ce_loss = 0 # EMA of baseline CE loss
    total_training_time = 0 # total wall-clock time of training
else:
    step = meta_data["step"]
    loop_state = meta_data["loop_state"]
    min_val_bpb = loop_state["min_val_bpb"]
    smooth_total_loss = loop_state.get("smooth_total_loss", loop_state.get("smooth_train_loss", 0))
    smooth_ce_loss = loop_state.get("smooth_ce_loss", smooth_total_loss)
    total_training_time = loop_state["total_training_time"]

# -----------------------------------------------------------------------------
# Training loop
while True:
    last_step = step == num_iterations # loop runs num_iterations+1 times so that we can eval/save at the end
    flops_so_far = num_flops_per_token * total_batch_size * step

    # once in a while: evaluate the val bpb (all ranks participate)
    if last_step or step % eval_every == 0:
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        with autocast_ctx:
            val_bpb = evaluate_bpb(model, val_loader, eval_steps, token_bytes)
        print0(f"Step {step:05d} | Validation bpb: {val_bpb:.4f}")
        if val_bpb < min_val_bpb:
            min_val_bpb = val_bpb
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "val/bpb": val_bpb,
        }, step=step)
        model.train()

    # once in a while: estimate the CORE metric (all ranks participate)
    # use the original uncompiled model because the inputs keep changing shape
    results = {}
    if core_metric_every > 0 and (last_step or (step > 0 and step % core_metric_every == 0)):
        model.eval()
        with autocast_ctx:
            results = evaluate_model(orig_model, tokenizer, device, max_per_task=core_metric_max_per_task)
        print0(f"Step {step:05d} | CORE metric: {results['core_metric']:.4f}")
        wandb_run.log({
            "step": step,
            "total_training_flops": flops_so_far,
            "core_metric": results["core_metric"],
            "centered_results": results["centered_results"],
        }, step=step)
        model.train()

    # once in a while: sample from the model (only on master process)
    # use the original uncompiled model because the inputs keep changing shape
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "The capital of France is",
            "The chemical symbol of gold is",
            "If yesterday was Friday, then tomorrow will be",
            "The opposite of hot is",
            "The planets of the solar system are:",
            "My favorite color is",
            "If 5*x + 3 = 13, then x is",
        ]
        engine = Engine(orig_model, tokenizer) # use orig_model to avoid recompilation
        for prompt in prompts:
            tokens = tokenizer(prompt, prepend="<|bos|>")
            with autocast_ctx:
                sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=16, temperature=0)
            print0(tokenizer.decode(sample[0]))
        model.train()

    # save checkpoint: at the end of the run, or every save_every steps, except at the first step or the resume step
    if last_step or (step > 0 and step != resume_from_step and save_every > 0 and step % save_every == 0):
        save_checkpoint(
            checkpoint_dir,
            step,
            orig_model.state_dict(), # model parameters
            [opt.state_dict() for opt in optimizers], # optimizer states
            { # metadata saved as json
                "step": step,
                "val_bpb": val_bpb, # loss at last step
                "model_config": model_config_kwargs,
                "glt_config": glt_config_kwargs,
                "user_config": user_config, # inputs to the training script
                "device_batch_size": device_batch_size,
                "max_seq_len": max_seq_len,
                "dataloader_state_dict": dataloader_state_dict,
                "loop_state": { # all loop state (other than step) so that we can resume training
                    "min_val_bpb": min_val_bpb,
                    "smooth_total_loss": smooth_total_loss,
                    "smooth_ce_loss": smooth_ce_loss,
                    "total_training_time": total_training_time,
                },
            },
            rank=ddp_rank,
        )

    # termination conditions (TODO: possibly also add loss explosions etc.)
    if last_step:
        break

    # -------------------------------------------------------------------------
    # single training step
    # evaluate the gradient
    synchronize()
    t0 = time.time()
    loss_breakdown_acc = {} if glt_config else None
    need_viz_data = viz_cfg.needs_batch_data(step)
    viz_batch_data = None
    last_ce_loss = None
    for micro_step in range(grad_accum_steps):
        model_kwargs = {}
        if glt_config:
            model_kwargs["return_loss_breakdown"] = True
        request_visuals = need_viz_data and viz_batch_data is None
        if request_visuals:
            model_kwargs["return_visuals"] = True
        with autocast_ctx:
            out = model(x, y, **model_kwargs)
        if isinstance(out, tuple):
            loss, extras = out
            breakdown = extras.get("loss_breakdown")
            visuals = extras.get("visuals")
        else:
            loss = out
            breakdown = None
            visuals = None
        train_loss = loss.detach() # for logging
        if breakdown is not None:
            if loss_breakdown_acc is not None:
                for k, v in breakdown.items():
                    loss_breakdown_acc[k] = loss_breakdown_acc.get(k, 0.0) + float(v.detach().cpu())
            ce_component = breakdown.get("ce")
            if ce_component is not None:
                last_ce_loss = ce_component.detach()
        if visuals is not None and viz_batch_data is None:
            viz_batch_data = VizBatch(
                latents=visuals["latents"],
                pre_norm=visuals["pre_norm"],
                mask=visuals["mask"],
            )
        loss = loss / grad_accum_steps # each .backward() is a grad sum => normalize loss here
        loss.backward()
        x, y, dataloader_state_dict = next(train_loader) # prefetch the next batch while the GPU is busy with forward/backward
    loss_breakdown_avg = None
    if loss_breakdown_acc is not None and loss_breakdown_acc:
        loss_breakdown_avg = {k: v / grad_accum_steps for k, v in loss_breakdown_acc.items()}
    # gradient clipping
    grad_clip_enabled = grad_clip > 0.0
    if grad_clip_enabled:
        grad_norm_tensor = torch.nn.utils.clip_grad_norm_(orig_model.parameters(), grad_clip)
        grad_norm = grad_norm_tensor.item() # GPU tensor -> CPU float (note: cpu-gpu sync point)
    # step the optimizers
    lrm = get_lr_multiplier(step)
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * lrm
    muon_momentum = get_muon_momentum(step)
    for group in muon_optimizer.param_groups:
        group["momentum"] = muon_momentum
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    # -------------------------------------------------------------------------

    # logging
    ema_beta = 0.9 # EMA decay factor for some smoothing just for nicer logging
    train_total_loss_val = train_loss.item()
    train_ce_loss_val = last_ce_loss.item() if last_ce_loss is not None else train_total_loss_val
    smooth_total_loss = ema_beta * smooth_total_loss + (1 - ema_beta) * train_total_loss_val
    smooth_ce_loss = ema_beta * smooth_ce_loss + (1 - ema_beta) * train_ce_loss_val
    debiased_total_loss = smooth_total_loss / (1 - ema_beta**(step + 1))
    debiased_ce_loss = smooth_ce_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    flops_per_sec = num_flops_per_token * total_batch_size / dt
    promised_flops_per_sec_h100 = 989e12 * ddp_world_size # bfloat16 H100 SXM and without 2:4 sparsity
    mfu = 100 * flops_per_sec / promised_flops_per_sec_h100 # in %
    if step > 10:
        total_training_time += dt # only count the time after the first 10 steps
    print_grad_norm = f" grad norm: {grad_norm:.4f} |" if grad_clip_enabled else ""
    glt_print = ""
    if glt_config and loss_breakdown_avg:
        glt_parts = ", ".join(
            f"{k}={v:.4f}" for k, v in loss_breakdown_avg.items() if k.startswith("glt/")
        )
        if glt_parts:
            glt_print = f" | glt[{glt_parts}]"
    if glt_config:
        loss_text = f"loss_ce: {debiased_ce_loss:.6f} | loss_total: {debiased_total_loss:.6f}"
    else:
        loss_text = f"loss: {debiased_total_loss:.6f}"
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | {loss_text} |{print_grad_norm} lrm: {lrm:.2f} | dt: {dt * 1000:.2f}ms | tok/sec: {tok_per_sec:,} | mfu: {mfu:.2f} | total time: {total_training_time/60:.2f}m{glt_print}")
    if log_every > 0 and step % log_every == 0:
        log_data = {
            "step": step,
            "total_training_flops": flops_so_far,
            "total_training_time": total_training_time,
            "train/loss": debiased_ce_loss,
            "train/lrm": lrm,
            "train/dt": dt,
            "train/tok_per_sec": tok_per_sec,
            "train/mfu": mfu,
        }
        if glt_config:
            log_data["glt/loss"] = debiased_total_loss
        if grad_clip_enabled:
            log_data["train/grad_norm"] = grad_norm
        if loss_breakdown_avg:
            for metric_name, metric_value in loss_breakdown_avg.items():
                log_data[f"loss_components/{metric_name}"] = metric_value
        wandb_run.log(log_data, step=step)
    if viz_cfg.needs_batch_data(step) and viz_batch_data is not None:
        maybe_log_visualizations(step, viz_cfg, viz_batch_data, wandb_run)

    # state update
    step += 1

# print a few more stats
print0(f"Peak memory usage: {get_max_memory() / 1024 / 1024:.2f}MiB")
print0(f"Total training time: {total_training_time/60:.2f}m")
print0(f"Minimum validation bpb: {min_val_bpb:.4f}")

# Log to report
from nanochat.report import get_report
get_report().log(section="Base model training", data=[
    user_config, # CLI args
    {"GLT config": glt_config_kwargs or "disabled"},
    { # stats about the training setup
        "Number of parameters": num_params,
        "Number of FLOPs per token": f"{num_flops_per_token:e}",
        "Calculated number of iterations": num_iterations,
        "Number of training tokens": total_tokens,
        "Tokens : Params ratio": total_batch_size * num_iterations / num_params,
        "DDP world size": ddp_world_size,
        "warmup_ratio": warmup_ratio,
        "warmdown_ratio": warmdown_ratio,
        "final_lr_frac": final_lr_frac,
    },
    { # stats about training outcomes
        "Minimum validation bpb": min_val_bpb,
        "Final validation bpb": val_bpb,
        "CORE metric estimate": results.get("core_metric", None),
        "MFU %": f"{mfu:.2f}%",
        "Total training flops": f"{flops_so_far:e}",
        "Total training time": f"{total_training_time/60:.2f}m",
        "Peak memory usage": f"{get_max_memory() / 1024 / 1024:.2f}MiB",
    }
])

# cleanup
wandb_run.finish() # wandb run finish
compute_cleanup()
