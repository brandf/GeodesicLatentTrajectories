#!/bin/bash

# $10 single-GPU nanochat run.
# Example usage:
#   bash run10.sh --gpu 5090
#   bash run10.sh --gpu h100 --run=mycustomrun

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash run10.sh [--gpu 5090|h100] [base_train_overrides...]

The optional base_train_overrides are forwarded verbatim to
`python -m scripts.base_train` after the default $10 profile flags.
Pass --glt to enable Geodesic Latent Trajectories losses during training.
EOF
}

GPU_CHOICE="5090"
BASE_TRAIN_OVERRIDES=()
ENABLE_GLT=0
GLT_OFFSETS="[-1,0,1]"
GLT_OFFSET_WEIGHTS=""
GLT_ENABLE_GEOM=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            if [[ $# -lt 2 ]]; then
                echo "error: --gpu requires an argument (5090 or h100)" >&2
                exit 1
            fi
            GPU_CHOICE="${2,,}"
            shift 2
            ;;
        --gpu=*)
            GPU_CHOICE="${1#*=}"
            GPU_CHOICE="${GPU_CHOICE,,}"
            shift
            ;;
        --glt)
            ENABLE_GLT=1
            shift
            ;;
        --glt_offsets)
            if [[ $# -lt 2 ]]; then
                echo "error: --glt_offsets requires an argument (e.g. '[-1,0,1]')" >&2
                exit 1
            fi
            GLT_OFFSETS="$2"
            shift 2
            ;;
        --glt_offsets=*)
            GLT_OFFSETS="${1#*=}"
            shift
            ;;
        --glt_offset_weights)
            if [[ $# -lt 2 ]]; then
                echo "error: --glt_offset_weights requires an argument (e.g. '[1,0.5,0.5]')" >&2
                exit 1
            fi
            GLT_OFFSET_WEIGHTS="$2"
            shift 2
            ;;
        --glt_offset_weights=*)
            GLT_OFFSET_WEIGHTS="${1#*=}"
            shift
            ;;
        --glt_no_geom)
            GLT_ENABLE_GEOM=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            BASE_TRAIN_OVERRIDES+=("$1")
            shift
            ;;
    esac
done

case "$GPU_CHOICE" in
    5090)
        DEVICE_BATCH_SIZE=20
        GPU_LABEL="RTX 5090"
        ;;
    h100)
        DEVICE_BATCH_SIZE=40
        GPU_LABEL="NVIDIA H100"
        ;;
    *)
        echo "Unsupported --gpu option: $GPU_CHOICE (expected 5090 or h100)" >&2
        exit 1
        ;;
esac

DEPTH=12
MODEL_DIM=$((DEPTH * 64))
NUM_HEADS=$(( (MODEL_DIM + 127) / 128 ))
NUM_KV_HEADS=$NUM_HEADS
MAX_SEQ_LEN=2048
TARGET_FLOPS="2.5e21"
TARGET_TOTAL_BATCH=524288
WORLD_TOKENS_PER_STEP=$((DEVICE_BATCH_SIZE * MAX_SEQ_LEN))
GRAD_ACCUM_STEPS=$(( (TARGET_TOTAL_BATCH + WORLD_TOKENS_PER_STEP - 1) / WORLD_TOKENS_PER_STEP ))
if (( GRAD_ACCUM_STEPS < 1 )); then
    GRAD_ACCUM_STEPS=1
fi
EFFECTIVE_TOTAL_BATCH=$((GRAD_ACCUM_STEPS * WORLD_TOKENS_PER_STEP))

echo "[run10] GPU profile: $GPU_LABEL (--gpu=$GPU_CHOICE)"
echo "[run10] depth=$DEPTH, d_model=$MODEL_DIM, heads=$NUM_HEADS, kv_heads=$NUM_KV_HEADS"
echo "[run10] device_batch_size=$DEVICE_BATCH_SIZE, seq_len=$MAX_SEQ_LEN"
echo "[run10] grad_accum_steps=$GRAD_ACCUM_STEPS, effective_total_batch=${EFFECTIVE_TOTAL_BATCH} tokens"
if (( EFFECTIVE_TOTAL_BATCH != TARGET_TOTAL_BATCH )); then
    echo "[run10] note: total batch rounded from $TARGET_TOTAL_BATCH to $EFFECTIVE_TOTAL_BATCH to maintain integer grad accumulation"
fi
if (( ENABLE_GLT )); then
    echo "[run10] GLT enabled (will pass --enable_glt=True to base_train)"
fi

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml

python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 160 &
DATASET_DOWNLOAD_PID=$!
cleanup_dataset_download() {
    if kill -0 "$DATASET_DOWNLOAD_PID" >/dev/null 2>&1; then
        echo "[run10] stopping background dataset download"
        kill "$DATASET_DOWNLOAD_PID"
    fi
}
trap cleanup_dataset_download EXIT
python -m scripts.tok_train --max_chars=2000000000
python -m scripts.tok_eval
echo "[run10] waiting for dataset download to finish..."
wait "$DATASET_DOWNLOAD_PID"
trap - EXIT

BASE_TRAIN_CMD=(
    python -m scripts.base_train
    --depth="$DEPTH"
    --max_seq_len="$MAX_SEQ_LEN"
    --device_batch_size="$DEVICE_BATCH_SIZE"
    --total_batch_size="$EFFECTIVE_TOTAL_BATCH"
    --target_flops="$TARGET_FLOPS"
    --run="$WANDB_RUN"
    --eval_every=200
    --log_every=20
)
if (( ENABLE_GLT )); then
    BASE_TRAIN_CMD+=(--enable_glt=True)
    BASE_TRAIN_CMD+=(--glt_ce_offsets="$GLT_OFFSETS")
    BASE_TRAIN_CMD+=(--glt_enable_geom_losses="$GLT_ENABLE_GEOM")
    if [[ -n "$GLT_OFFSET_WEIGHTS" ]]; then
        BASE_TRAIN_CMD+=(--glt_ce_offset_weights="$GLT_OFFSET_WEIGHTS")
    fi
fi
BASE_TRAIN_CMD+=("${BASE_TRAIN_OVERRIDES[@]}")
"${BASE_TRAIN_CMD[@]}"

python -m scripts.base_loss -- --device_batch_size="$DEVICE_BATCH_SIZE"
python -m scripts.base_eval

python -m nanochat.report generate
