#!/usr/bin/env python3
"""
Test script for the trained GPT2 recipe model.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_trained_model(model_path: str):
    """Load the trained GPT2 model with LoRA adapters."""
    print(f"🤗 Loading trained model from: {model_path}")
    
    # Load base model and tokenizer
    base_model = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model)
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(model, model_path)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_recipe(model, tokenizer, ingredients: list, max_length: int = 512):
    """Generate a recipe using the trained model."""
    
    # Create prompt
    ingredients_str = ", ".join(ingredients)
    prompt = f"You are an expert recipe generator. Create a detailed recipe in JSON format.\nTitle: Recipe with {ingredients_str}\nIngredients:\n"
    
    # Add ingredients to prompt
    for ingredient in ingredients:
        prompt += f"- {ingredient}\n"
    
    prompt += "\nInstructions:\n"
    
    print(f"🎯 Generating recipe for: {ingredients_str}")
    print(f"📝 Prompt: {prompt[:200]}...")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
    generated_part = tokenizer.decode(outputs[0][prompt_tokens.shape[1]:], skip_special_tokens=True)
    
    return generated_text, generated_part

def main():
    model_path = "./recipe-model-gpt2-gpu"
    
    try:
        # Load model
        model, tokenizer = load_trained_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Test ingredients
        test_ingredients = [
            ["chicken breast", "rice", "soy sauce"],
            ["tomatoes", "pasta", "garlic"],
            ["eggs", "milk", "bread"],
        ]
        
        for i, ingredients in enumerate(test_ingredients, 1):
            print(f"\n{'='*50}")
            print(f"🧪 Test {i}: {', '.join(ingredients)}")
            print(f"{'='*50}")
            
            try:
                full_text, generated_part = generate_recipe(model, tokenizer, ingredients)
                print(f"\n📄 Generated Recipe:")
                print(f"{generated_part}")
                
            except Exception as e:
                print(f"❌ Error generating recipe: {e}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure the model was trained successfully and the path is correct.")

if __name__ == "__main__":
    main()
