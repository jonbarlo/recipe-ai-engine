#!/usr/bin/env python3
"""
Training script for Recipe AI Engine using GPT2 (smaller model, lower VRAM requirements).

Requirements (install from requirements-train.txt):
  - transformers, datasets, peft, accelerate, trl, torch

Notes:
  - GPT2 is much smaller than Llama 2 (124M vs 7B parameters)
  - Works well on lower VRAM systems (4-8GB)
  - Good for testing the training pipeline
  - Faster training but lower quality than larger models
"""

import argparse
import json
import os
from typing import Dict, List

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer


def load_json_dataset(dataset_path: str) -> Dataset:
    """Load a JSON dataset and convert it to instruction-tuning text examples.

    Expected JSON structure (list of objects) with fields like:
      - title: str
      - ingredients: list[str] or list[dict]
      - instructions: list[str]

    We will map to a single 'text' field suitable for SFT.
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    def normalize_ingredients(ingredients) -> List[str]:
        if not ingredients:
            return []
        if isinstance(ingredients, list) and ingredients and isinstance(ingredients[0], dict):
            return [f"{it.get('item', '')}: {it.get('amount', '')}".strip() for it in ingredients]
        return [str(x) for x in ingredients]

    def build_text(example: Dict) -> str:
        ingredients = normalize_ingredients(example.get("ingredients", []))
        instructions = example.get("instructions", [])
        title = example.get("title", "Recipe")

        ing_str = "\n".join([f"- {i}" for i in ingredients]) if ingredients else ""
        inst_str = "\n".join([f"{idx+1}. {step}" for idx, step in enumerate(instructions)]) if instructions else ""

        # Simple instruction-tuning style prompt-response in one text
        text = (
            f"You are an expert recipe generator. Create a detailed recipe in JSON format.\n"
            f"Title: {title}\n"
            f"Ingredients:\n{ing_str}\n\n"
            f"Instructions:\n{inst_str}\n"
        )
        return text

    texts = [{"text": build_text(example)} for example in data]
    return Dataset.from_list(texts)


def build_model_and_tokenizer(base_model: str = "gpt2"):
    """Build GPT2 model and tokenizer (no quantization needed for small model)."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,  # Use float32 for GPT2 (small enough)
    )

    return model, tokenizer


def apply_lora(model, r: int = 8, alpha: int = 16, dropout: float = 0.1):
    """Apply LoRA to GPT2 model with appropriate target modules."""
    target_modules = [
        "c_attn",
        "c_proj", 
        "c_fc",
    ]
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def parse_args():
    parser = argparse.ArgumentParser(description="Train GPT2 for recipe generation")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset file")
    parser.add_argument("--output", required=True, help="Output directory for trained model")
    parser.add_argument("--model", default="gpt2", help="GPT2 model variant (gpt2, gpt2-medium, gpt2-large)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--cutoff-len", type=int, default=512, help="Max sequence length")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full fine-tuning)")
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout")
    parser.add_argument("--test", action="store_true", help="Run a small test train")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("📦 Loading dataset...")
    dataset = load_json_dataset(args.dataset)
    if args.test:
        dataset = dataset.select(range(min(32, len(dataset))))

    print("🤗 Loading GPT2 model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer(args.model)

    if args.no_lora:
        print("⚠️ Training without LoRA adapters (full fine-tuning).")
    else:
        print("🔧 Applying LoRA adapters...")
        model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    print("🏋️ Starting supervised fine-tuning...")
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        logging_steps=5,
        save_steps=100,
        output_dir=args.output,
        save_total_limit=2,
        optim="adamw_torch",
        report_to=[],
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
    )
    
    # Clear GPU memory before training
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    trainer.train()

    print("💾 Saving model and tokenizer...")
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("✅ GPT2 training complete!")
    print(f"➡️  Model saved to: {args.output}")
    print("➡️  You can now use this model with Ollama or test it directly.")


if __name__ == "__main__":
    main()
