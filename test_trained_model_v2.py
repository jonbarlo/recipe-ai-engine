#!/usr/bin/env python3
"""
Improved test script for the trained GPT2 recipe model.
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

def generate_recipe_improved(model, tokenizer, ingredients: list):
    """Generate a recipe using improved prompting."""
    
    # Create a more natural prompt
    ingredients_str = ", ".join(ingredients)
    prompt = f"Create a recipe using {ingredients_str}:\n\nIngredients:\n"
    
    # Add ingredients
    for ingredient in ingredients:
        prompt += f"- {ingredient}\n"
    
    prompt += "\nInstructions:\n"
    
    print(f"🎯 Generating recipe for: {ingredients_str}")
    print(f"📝 Using improved prompt...")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    
    # Generate with better parameters
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.8,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.2,  # Prevent repetition
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract the generated part (after the prompt)
    prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
    generated_part = tokenizer.decode(outputs[0][prompt_tokens.shape[1]:], skip_special_tokens=True)
    
    return generated_text, generated_part

def test_with_different_prompts(model, tokenizer):
    """Test with various prompt styles."""
    
    test_cases = [
        {
            "name": "Simple Recipe",
            "ingredients": ["chicken", "rice", "soy sauce"],
            "prompt": "Make a simple recipe with chicken, rice, and soy sauce:\n\n"
        },
        {
            "name": "Pasta Dish", 
            "ingredients": ["pasta", "tomatoes", "garlic"],
            "prompt": "Create a pasta dish using pasta, tomatoes, and garlic:\n\n"
        },
        {
            "name": "Breakfast",
            "ingredients": ["eggs", "bread", "butter"],
            "prompt": "Make a breakfast recipe with eggs, bread, and butter:\n\n"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*60}")
        print(f"🧪 Test {i}: {test_case['name']}")
        print(f"📋 Ingredients: {', '.join(test_case['ingredients'])}")
        print(f"{'='*60}")
        
        # Create prompt
        prompt = test_case['prompt'] + "Ingredients:\n"
        for ingredient in test_case['ingredients']:
            prompt += f"- {ingredient}\n"
        prompt += "\nInstructions:\n"
        
        # Tokenize and generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.9,
                top_p=0.95,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.3,
                no_repeat_ngram_size=2,
            )
        
        # Extract generated part
        prompt_tokens = tokenizer.encode(prompt, return_tensors="pt")
        generated_part = tokenizer.decode(outputs[0][prompt_tokens.shape[1]:], skip_special_tokens=True)
        
        print(f"\n📄 Generated Recipe:")
        print(f"{generated_part}")
        print(f"\n{'─'*60}")

def main():
    model_path = "./recipe-model-gpt2-gpu"
    
    try:
        # Load model
        model, tokenizer = load_trained_model(model_path)
        print("✅ Model loaded successfully!")
        
        # Test with different prompts
        test_with_different_prompts(model, tokenizer)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure the model was trained successfully.")

if __name__ == "__main__":
    main()
