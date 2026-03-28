#!/bin/bash
# Pull the latest checkpoint from the GPU server.
# Usage: bash scripts/pull_checkpoint.sh [checkpoint-dir]
#   checkpoint-dir defaults to $CHECKPOINT_DIR from .env

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.env"

LOCAL_DIR="${1:-${CHECKPOINT_DIR:-checkpoints/qwen-cricket}}"
REMOTE_DIR="$GPU_REMOTE_DIR/$LOCAL_DIR"

echo "Pulling checkpoint from $GPU_HOST:$REMOTE_DIR → $LOCAL_DIR"

mkdir -p "$LOCAL_DIR"

if ssh -i "$GPU_KEY" "$GPU_HOST" "[ -f $REMOTE_DIR/adapter_model.safetensors ]"; then
    echo "Final checkpoint found — pulling full directory"
    # Use rsync to copy contents directly (avoids nested directory)
    rsync -av -e "ssh -i $GPU_KEY" "$GPU_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
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
    rsync -av -e "ssh -i $GPU_KEY" "$GPU_HOST:$LATEST/" "$LOCAL_DIR/$CKPT_NAME/"
    echo "Pulled to $LOCAL_DIR/$CKPT_NAME"
fi

echo "Done. Checkpoint at: $LOCAL_DIR"
