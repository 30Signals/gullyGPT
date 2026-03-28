#!/bin/bash
# Watch training progress on GPU server. Ctrl-C to stop.
# Usage: bash scripts/watch_training.sh

GPU_HOST="root@205.147.102.130"
GPU_KEY="/home/azureuser/.ssh/id_rsa_e2e_tir"

while true; do
    clear
    echo "=== gullyGPT training monitor === $(date)"
    echo ""
    ssh -i "$GPU_KEY" "$GPU_HOST" \
        "grep -oP '\d+/1134' /root/gullyGPT/train.log | tail -1 | tr -d '\n' && \
         echo ' steps' && \
         grep -oP 'loss.*?(?=,|\s*\d+%|\n)' /root/gullyGPT/train.log | tail -3 && \
         nvidia-smi | grep -E 'MiB|Temp|%'"
    sleep 30
done
