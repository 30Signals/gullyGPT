#!/bin/bash
# Watch training progress on GPU server. Ctrl-C to stop.
# Usage: bash scripts/watch_training.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.env"

while true; do
    clear
    echo "=== gullyGPT training monitor === $(date)"
    echo ""
    ssh -i "$GPU_KEY" "$GPU_HOST" \
        "grep -oP '\d+/1134' $GPU_REMOTE_DIR/train.log | tail -1 | tr -d '\n' && \
         echo ' steps' && \
         grep -oP 'loss.*?(?=,|\s*\d+%|\n)' $GPU_REMOTE_DIR/train.log | tail -3 && \
         nvidia-smi | grep -E 'MiB|Temp|%'"
    sleep 30
done
