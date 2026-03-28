#!/bin/bash
# Pull the latest checkpoint from the GPU server.
# Usage: bash scripts/pull_checkpoint.sh [checkpoint-dir]
#   checkpoint-dir defaults to "checkpoints/qwen-cricket"

set -e

GPU_HOST="root@205.147.102.130"
GPU_KEY="/home/azureuser/.ssh/id_rsa_e2e_tir"
REMOTE_DIR="/root/gullyGPT/${1:-checkpoints/qwen-cricket}"
LOCAL_DIR="${1:-checkpoints/qwen-cricket}"

echo "Pulling checkpoint from $GPU_HOST:$REMOTE_DIR → $LOCAL_DIR"

mkdir -p "$LOCAL_DIR"

# Find the best checkpoint (load_best_model_at_end saves to output_dir directly at end of training)
# During training, intermediate checkpoints are in checkpoint-NNN subdirs
if ssh -i "$GPU_KEY" "$GPU_HOST" "[ -f $REMOTE_DIR/adapter_model.safetensors ]"; then
    echo "Final checkpoint found — pulling full directory"
    scp -i "$GPU_KEY" -r "$GPU_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
else
    # Pull latest intermediate checkpoint
    LATEST=$(ssh -i "$GPU_KEY" "$GPU_HOST" \
        "ls -d $REMOTE_DIR/checkpoint-* 2>/dev/null | sort -V | tail -1")
    if [ -z "$LATEST" ]; then
        echo "No checkpoint found at $REMOTE_DIR"
        exit 1
    fi
    CKPT_NAME=$(basename "$LATEST")
    echo "Latest checkpoint: $CKPT_NAME"
    mkdir -p "$LOCAL_DIR/$CKPT_NAME"
    scp -i "$GPU_KEY" -r "$GPU_HOST:$LATEST/" "$LOCAL_DIR/$CKPT_NAME/"
    echo "Pulled to $LOCAL_DIR/$CKPT_NAME"
fi

echo "Done."
