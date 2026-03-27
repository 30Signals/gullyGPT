"""
Phase 1c: HuggingFace Dataset wrapper for causal LM training.

Two modes:
  - MatchDataset (default): one sample per match, truncated to max_len.
    Clean match boundaries, ~12k samples for 12k matches. Recommended.
  - CricketDataset: packs entire corpus into fixed windows (legacy).

Usage:
    from src.data.dataset import make_datasets
    train_ds, val_ds = make_datasets("data/processed", tokenizer, max_len=1024)
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class MatchDataset(Dataset):
    """
    One sample per match. Reads a text file where matches are separated by
    double newlines, tokenizes each match independently, and truncates/pads
    to max_len. Clean match boundaries — no cross-match token bleed.
    """

    def __init__(
        self,
        file_path: Path | str,
        tokenizer: PreTrainedTokenizer,
        max_len: int = 1024,
    ):
        file_path = Path(file_path)
        self.max_len = max_len
        self.tokenizer = tokenizer

        text = file_path.read_text(encoding="utf-8")
        matches = [m.strip() for m in text.split("\n\n") if m.strip()]

        self.samples = []
        for match_text in matches:
            ids = tokenizer.encode(
                match_text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_len,
            )
            self.samples.append(ids)

        total_tokens = sum(len(s) for s in self.samples)
        print(f"MatchDataset: {len(self.samples)} matches, {total_tokens:,} tokens "
              f"(avg {total_tokens // len(self.samples) if self.samples else 0} tokens/match)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ids = torch.tensor(self.samples[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


class CricketDataset(Dataset):
    """
    Legacy: tokenize entire corpus and chunk into fixed-length windows.
    Produces many more samples but matches bleed across chunk boundaries.
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
        self.stride = stride or max_len

        text = file_path.read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text, add_special_tokens=False)

        self.chunks = []
        for start in range(0, len(token_ids) - max_len, self.stride):
            chunk = token_ids[start : start + max_len]
            self.chunks.append(chunk)

        print(f"CricketDataset: {len(token_ids):,} tokens → {len(self.chunks):,} chunks of {max_len}")

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        ids = torch.tensor(self.chunks[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


def make_datasets(
    data_dir: Path,
    tokenizer: PreTrainedTokenizer,
    max_len: int = 1024,
    per_match: bool = True,
):
    """Returns (train_dataset, val_dataset). Uses MatchDataset by default."""
    data_dir = Path(data_dir)
    cls = MatchDataset if per_match else CricketDataset
    train_ds = cls(data_dir / "train.txt", tokenizer, max_len)
    val_ds = cls(data_dir / "val.txt", tokenizer, max_len)
    return train_ds, val_ds
