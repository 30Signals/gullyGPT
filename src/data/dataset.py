"""
Phase 1c: HuggingFace Dataset wrapper for causal LM training.
Packs multiple match sequences into fixed-length context windows (2048 tokens).

Usage:
    from src.data.dataset import CricketDataset
    ds = CricketDataset("data/processed/train.txt", tokenizer, max_len=2048)
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CricketDataset(Dataset):
    """
    Reads a text file of match sequences (one match per double-newline block),
    tokenizes everything, then chunks into fixed-length windows for causal LM.
    """

    def __init__(
        self,
        file_path: Path | str,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 2048,
        stride: Optional[int] = None,
    ):
        file_path = Path(file_path)
        self.max_len = max_len
        self.stride = stride or max_len  # non-overlapping by default

        text = file_path.read_text(encoding="utf-8")

        # Tokenize entire corpus at once — efficient for packing
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        # Chunk into windows
        self.chunks = []
        for start in range(0, len(token_ids) - max_len, self.stride):
            chunk = token_ids[start : start + max_len]
            self.chunks.append(chunk)

        print(f"Dataset: {len(token_ids):,} tokens → {len(self.chunks):,} chunks of {max_len}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def make_datasets(data_dir: Path, tokenizer: PreTrainedTokenizer, max_len: int = 2048):
    """Convenience: returns (train_dataset, val_dataset)."""
    data_dir = Path(data_dir)
    train_ds = CricketDataset(data_dir / "train.txt", tokenizer, max_len)
    val_ds = CricketDataset(data_dir / "val.txt", tokenizer, max_len)
    return train_ds, val_ds
