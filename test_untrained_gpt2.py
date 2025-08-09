#!/usr/bin/env python3
"""
Test untrained GPT2 to compare with our trained model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_untrained_gpt2():
    """Test the base GPT2 model without any training."""
    
    print("🤖 Testing untrained GPT2 model...")
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompt
    prompt = "Create a recipe using chicken, rice, and soy sauce:\n\nIngredients:\n- chicken\n- rice\n- soy sauce\n\nInstructions:\n"
    
    print(f"📝 Prompt: {prompt}")
    
    # Tokenize and generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract generated part
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
    generated_part = tokenizer.decode(outputs[0][prompt_tokens.shape[1]:], skip_special_tokens=True)
    
    print(f"\n📄 Untrained GPT2 Output:")
    print(f"{generated_part}")
    print(f"\n💡 This shows what GPT2 generates without any recipe training.")

if __name__ == "__main__":
    test_untrained_gpt2()
