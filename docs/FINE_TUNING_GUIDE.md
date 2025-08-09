# Recipe AI Fine-tuning Guide

## Overview

This guide will help you fine-tune your recipe AI model using your existing datasets. Fine-tuning will significantly improve the model's performance for recipe generation.

## Prerequisites

### 1. **Install Fine-tuning Dependencies**
```bash
pip install -r requirements-train.txt
```

### 2. **Hardware Requirements**
- **Minimum**: 16GB GPU VRAM (RTX 3080 or better)
- **Recommended**: 24GB+ GPU VRAM (RTX 4090, A100)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space

### 3. **Dataset Format**

Your recipe dataset should be in one of these formats:

#### **JSON Format** (Recommended)
```json
[
  {
    "title": "Chicken Stir Fry",
    "ingredients": [
      {"item": "chicken breast", "amount": "1 lb"},
      {"item": "soy sauce", "amount": "2 tbsp"},
      {"item": "vegetables", "amount": "2 cups"}
    ],
    "instructions": [
      "Heat oil in a wok",
      "Add chicken and cook until browned",
      "Add vegetables and stir fry"
    ]
  }
]
```

#### **CSV Format**
```csv
title,ingredients,instructions
"Chicken Stir Fry","chicken,soy sauce,vegetables","Heat oil,Add chicken,Cook vegetables"
```

#### **Hugging Face Dataset**
```python
from datasets import Dataset

dataset = Dataset.from_dict({
    "title": ["Chicken Stir Fry"],
    "ingredients": [["chicken", "soy sauce", "vegetables"]],
    "instructions": [["Heat oil", "Add chicken", "Cook vegetables"]]
})
```

## Quick Start

### 1. **Install Training Dependencies**
```bash
# GPU training (recommended)
make install-train-gpu

# CPU training (slower but works everywhere)
make install-train-cpu

# Check GPU availability
make check-gpu
```

### 2. **Quick Test Training**
```bash
# GPT2 (fastest, lowest VRAM)
make train-gpt2-gpu

# Mistral (good balance)
make train-mistral-gpu

# Llama 2 (highest quality)
make train-llama-gpu
```

### 3. **Full Production Training**
```bash
# GPT2 full training
make train-gpt2-full-gpu

# Mistral full training
make train-mistral-full-gpu

# Llama 2 full training
make train-llama-full-gpu
```

### 4. **Manual Training**
```bash
python scripts/train.py \
    --dataset datasets/recipe_dataset_1000.json \
    --model meta-llama/Llama-2-7b-hf \
    --output ./recipe-model \
    --epochs 5 \
    --batch-size 2 \
    --learning-rate 1e-4 \
    --test
```

## Configuration Options

### **Model Selection**
- **GPT2** (Fastest, lowest VRAM, good for testing)
- **Mistral 7B** (Good balance of quality and speed)
- **Llama 2 7B** (Highest quality, requires more VRAM)
- **Llama 2 13B** (Best quality, requires 24GB+ VRAM)

### **Training Parameters**
- `--epochs`: Number of training epochs (1-10)
- `--batch-size`: Batch size (1-8, depends on VRAM)
- `--learning-rate`: Learning rate (1e-5 to 1e-3)
- `--no-lora`: Disable LoRA (uses more VRAM)

### **LoRA Configuration**
LoRA (Low-Rank Adaptation) is enabled by default for efficient fine-tuning:
- **Memory efficient**: Uses ~8GB VRAM instead of 24GB+
- **Fast training**: 2-4 hours instead of 8-12 hours
- **Good quality**: Maintains most of the improvement

## Dataset Preparation

### **Minimum Requirements**
- **Size**: 1,000+ recipes (10,000+ recommended)
- **Quality**: Well-formatted recipes with clear instructions
- **Diversity**: Different cuisines, difficulty levels, ingredients

### **Data Cleaning Tips**
1. **Remove duplicates**
2. **Standardize ingredient names**
3. **Fix instruction formatting**
4. **Add missing information**

### **Example Data Cleaning Script**
```python
import json
import pandas as pd

def clean_recipe_dataset(input_file, output_file):
    """Clean and standardize recipe dataset"""
    
    # Load data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    cleaned_data = []
    for recipe in data:
        # Clean title
        title = recipe.get('title', '').strip()
        if not title or len(title) < 3:
            continue
        
        # Clean ingredients
        ingredients = recipe.get('ingredients', [])
        if not ingredients:
            continue
        
        # Clean instructions
        instructions = recipe.get('instructions', [])
        if not instructions or len(instructions) < 2:
            continue
        
        # Standardize format
        cleaned_recipe = {
            'title': title,
            'ingredients': ingredients,
            'instructions': instructions
        }
        
        cleaned_data.append(cleaned_recipe)
    
    # Save cleaned data
    with open(output_file, 'w') as f:
        json.dump(cleaned_data, f, indent=2)
    
    print(f"Cleaned {len(cleaned_data)} recipes")

# Usage
clean_recipe_dataset('raw_recipes.json', 'cleaned_recipes.json')
```

## Training Process

### **Phase 1: Setup (5-10 minutes)**
1. Load base model and tokenizer
2. Prepare and tokenize dataset
3. Setup LoRA configuration

### **Phase 2: Training (2-8 hours)**
1. Train model on recipe dataset
2. Save checkpoints every 500 steps
3. Monitor training metrics

### **Phase 3: Testing (5-10 minutes)**
1. Load fine-tuned model
2. Test with sample ingredients
3. Evaluate quality

## Monitoring Training

### **Key Metrics to Watch**
- **Loss**: Should decrease over time
- **Learning Rate**: Should follow schedule
- **Memory Usage**: Should stay within limits

### **Common Issues**
- **Out of Memory**: Reduce batch size
- **Slow Training**: Use smaller model or LoRA
- **Poor Quality**: Increase dataset size or epochs

## Integration with Ollama

### **Convert Fine-tuned Model to Ollama**
```bash
# Create Ollama model file
cat > Modelfile << EOF
FROM ./recipe-model
TEMPLATE """Recipe: Create a recipe using {{.Ingredients}}

Ingredients:
{{.Ingredients}}

Instructions:
"""
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER max_tokens 512
EOF

# Create Ollama model
ollama create recipe-ai ./recipe-model
```

### **Use Fine-tuned Model**
```python
# Update your Ollama client to use the fine-tuned model
generator = OllamaRecipeGenerator(model_name="recipe-ai")
```

## Expected Results

### **Quality Improvements**
- **Before fine-tuning**: 4.0/10 quality score
- **After fine-tuning**: 7.5-8.5/10 quality score

### **Specific Improvements**
- ✅ Better ingredient utilization
- ✅ More detailed instructions
- ✅ Cuisine-specific recipes
- ✅ Proper cooking techniques
- ✅ Consistent formatting

## Troubleshooting

### **Memory Issues**
```bash
# Reduce batch size
python scripts/train.py --batch-size 1

# Use smaller model
python scripts/train.py --model gpt2

# Disable LoRA (uses more memory)
python scripts/train.py --no-lora
```

### **Quality Issues**
```bash
# Increase training epochs
python scripts/train.py --epochs 10

# Lower learning rate
python scripts/train.py --learning-rate 1e-5

# Use larger model
python scripts/train.py --model meta-llama/Llama-2-13b-hf
```

### **Dataset Issues**
- **Small dataset**: Collect more recipes
- **Poor quality**: Clean and standardize data
- **Format issues**: Convert to JSON format

## Training Scripts

The project includes dedicated training scripts for different models:

### **GPT2 Training (`scripts/train_gpt2.py`)**
- **Best for**: Quick testing, low VRAM systems
- **VRAM usage**: ~4GB
- **Speed**: Fastest training
- **Quality**: Good for basic recipe generation

### **Mistral Training (`scripts/train_mistral.py`)**
- **Best for**: Production use, good balance
- **VRAM usage**: ~8GB
- **Speed**: Medium training time
- **Quality**: Excellent recipe generation

### **Llama 2 Training (`scripts/train.py`)**
- **Best for**: Highest quality recipes
- **VRAM usage**: ~12GB
- **Speed**: Slower training
- **Quality**: Best recipe generation

## Next Steps

1. **Prepare your dataset** in the correct format
2. **Install fine-tuning dependencies** (`make install-train-gpu`)
3. **Choose your model** (GPT2 for testing, Mistral for production, Llama 2 for best quality)
4. **Run the training** using the appropriate `make` command
5. **Test the fine-tuned model**
6. **Integrate with your existing system**

## Example Commands

### **Quick Test Training**
```bash
# GPT2 test (fastest)
make train-gpt2-gpu

# Mistral test (good balance)
make train-mistral-gpu

# Llama 2 test (highest quality)
make train-llama-gpu
```

### **Production Training**
```bash
# GPT2 full training
make train-gpt2-full-gpu

# Mistral full training
make train-mistral-full-gpu

# Llama 2 full training
make train-llama-full-gpu
```

### **Manual Training**
```bash
python scripts/train.py \
    --dataset datasets/full_recipe_dataset.json \
    --model meta-llama/Llama-2-7b-hf \
    --epochs 5 \
    --batch-size 4 \
    --learning-rate 2e-4 \
    --output ./production-recipe-model
```

This fine-tuning process will significantly improve your recipe generation quality and make your AI much more capable of creating professional-quality recipes! 