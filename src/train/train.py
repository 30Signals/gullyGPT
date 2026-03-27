"""
Phase 2: LoRA fine-tune Qwen2.5-3B on cricket sequences.
bfloat16, no gradient checkpointing, small batch — fast on L4 (23GB).

Usage:
    python src/train/train.py --config src/train/config.yaml
    python src/train/train.py --config src/train/config.yaml --resume-from checkpoints/qwen-cricket/checkpoint-500
"""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import make_datasets


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="src/train/config.yaml")
    parser.add_argument("--resume-from", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    print(f"Loading tokenizer: {cfg['model']}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model (bfloat16): {cfg['model']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    # Disable KV cache (not needed for training)
    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Building datasets...")
    train_ds, val_ds = make_datasets(cfg["data_dir"], tokenizer, cfg["max_seq_len"])

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accumulation"],
        learning_rate=float(cfg["lr"]),
        warmup_steps=cfg.get("warmup_steps", 100),
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=cfg.get("logging_steps", 20),
        eval_strategy="steps",
        eval_steps=cfg.get("eval_steps", 200),
        save_steps=cfg.get("save_steps", 200),
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
    )

    trainer.train(resume_from_checkpoint=args.resume_from)
    trainer.save_model(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])
    print(f"\nDone. Model saved to {cfg['output_dir']}")


if __name__ == "__main__":
    main()
