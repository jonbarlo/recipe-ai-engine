#!/usr/bin/env python3
"""
Training script for Recipe AI Engine using Mistral (good balance of quality and efficiency).

Requirements (install from requirements-train.txt):
  - transformers, datasets, peft, accelerate, bitsandbytes, trl, torch (CUDA build)

Notes:
  - Mistral 7B is similar size to Llama 2 7B but often performs better
  - Requires 16GB+ VRAM for QLoRA training
  - Good balance between quality and resource usage
  - Supports QLoRA for efficient fine-tuning
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
    BitsAndBytesConfig,
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


def build_model_and_tokenizer(
    base_model: str = "mistralai/Mistral-7B-v0.1",
    use_qlora: bool = True,
):
    """Build Mistral model and tokenizer with QLoRA support."""
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_qlora:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    # Handle low VRAM scenarios
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16 if use_qlora else torch.float16,
        )
    except ValueError as e:
        if "GPU RAM" in str(e) or "VRAM" in str(e):
            print("⚠️ GPU memory insufficient, trying CPU offload...")
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map="auto",
                quantization_config=quant_config,
                torch_dtype=torch.bfloat16 if use_qlora else torch.float16,
                llm_int8_enable_fp32_cpu_offload=True,
            )
        else:
            raise e

    return model, tokenizer


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Apply LoRA to Mistral model with appropriate target modules."""
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
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
    parser = argparse.ArgumentParser(description="Train Mistral for recipe generation")
    parser.add_argument("--dataset", required=True, help="Path to JSON dataset file")
    parser.add_argument("--output", required=True, help="Output directory for trained model")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1", help="Mistral model variant")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--cutoff-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (requires large VRAM)")
    parser.add_argument("--no-qlora", action="store_true", help="Disable 4-bit QLoRA quantization")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--merge-adapter", action="store_true", help="Merge LoRA adapters into base model")
    parser.add_argument("--test", action="store_true", help="Run a small test train")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    print("📦 Loading dataset...")
    dataset = load_json_dataset(args.dataset)
    if args.test:
        dataset = dataset.select(range(min(64, len(dataset))))

    print("🤗 Loading Mistral model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer(
        base_model=args.model,
        use_qlora=not args.no_qlora,
    )

    if args.no_lora:
        print("⚠️ Training without LoRA adapters (requires significant VRAM).")
    else:
        print("🔧 Applying LoRA adapters...")
        model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)

    print("🏋️ Starting supervised fine-tuning...")
    from transformers import TrainingArguments
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        output_dir=args.output,
        save_total_limit=2,
        bf16=True,
        optim="paged_adamw_8bit" if not args.no_qlora else "adamw_torch",
        report_to=[],
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=args.cutoff_len,
        packing=True,
        dataset_text_field="text",
        args=training_args,
    )

    trainer.train()

    print("💾 Saving model and tokenizer...")
    trainer.model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    if args.merge_adapter and not args.no_lora:
        print("🧬 Merging LoRA adapters into base model...")
        # Reload base model in FP16 without quantization for a clean merge
        base_model_fp16 = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        merged = base_model_fp16
        try:
            # Load adapters and merge
            from peft import PeftModel
            peft_model = PeftModel.from_pretrained(merged, args.output)
            merged = peft_model.merge_and_unload()
            merge_dir = os.path.join(args.output, "merged")
            os.makedirs(merge_dir, exist_ok=True)
            merged.save_pretrained(merge_dir)
            tokenizer.save_pretrained(merge_dir)
            print(f"✅ Merged model saved to: {merge_dir}")
        except Exception as merge_exc:
            print(f"❌ Merge failed: {merge_exc}")

    print("✅ Mistral training complete!")
    print(f"➡️  Model saved to: {args.output}")
    print("➡️  You can now use this model with Ollama or test it directly.")


if __name__ == "__main__":
    main()
