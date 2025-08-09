# Recipe AI Engine - Makefile
# Standard Python project commands

.PHONY: help install test recipe clean

# Default target
help:
	@echo Recipe AI Engine - Available Commands:
	@echo.
	@echo Installation:
	@echo   install        - Install all requirements and package
	@echo.
	@echo Testing:
	@echo   test           - Run all tests
	@echo   test-generation - Test recipe generation
	@echo   test-quality   - Test recipe quality
	@echo.
	@echo Recipe Generation:
	@echo   recipe         - Generate a recipe (interactive)
	@echo   recipe-quick   - Quick recipe with sample ingredients
	@echo.
	@echo API:
	@echo   api            - Run FastAPI server with uvicorn
	@echo   api-dev        - Run FastAPI server with auto-reload (development)
	@echo.
	@echo Development:
	@echo   setup-model    - Setup fine-tuned recipe model
	@echo   hf-login       - Login to Hugging Face using token from .env
	@echo   clean          - Clean up temporary files
	@echo.
	@echo Training Installation:
	@echo   install-train-cpu - Install training requirements (CPU only)
	@echo   install-train-gpu - Install training requirements (GPU/CUDA)
	@echo   install-train     - Install training requirements (default: GPU)
	@echo.
	@echo Training Commands:
	@echo   train-gpt2-cpu/gpu     - Train GPT2 (CPU/GPU)
	@echo   train-mistral-cpu/gpu  - Train Mistral (CPU/GPU)
	@echo   train-llama-cpu/gpu    - Train Llama 2 (CPU/GPU)
	@echo   train-gpt2/mistral/llama - Default GPU training
	@echo.

# Installation commands
install:
	@echo Installing all requirements...
	python -m pip install -r requirements.txt
	@echo Installing package in development mode...
	python -m pip install -e .
	@echo Installation complete!

# Testing commands
test: test-generation test-quality
	@echo All tests completed!

test-generation:
	@echo Testing recipe generation...
	python tests/test_recipe_generation.py

test-quality:
	@echo Testing recipe quality...
	python tests/test_recipe_quality.py
	@echo Results saved to tests/recipe_quality_test_results.json

# Recipe generation commands
recipe:
	@echo Recipe AI Engine - Interactive Mode
	@echo Enter ingredients (comma-separated):
	@python -c "from recipe_ai_engine import RecipeRequest, RecipeGenerator; import sys; ingredients = input('Ingredients: ').split(','); request = RecipeRequest(ingredients=[i.strip() for i in ingredients]); generator = RecipeGenerator(); recipe = generator.generate_recipe(request); print(f'\nGenerated Recipe:\n{recipe.title}\n\nIngredients:\n{[f\"- {ing[\"item\"]}: {ing[\"amount\"]}\" for ing in recipe.ingredients]}\n\nInstructions:\n{[f\"{i+1}. {step}\" for i, step in enumerate(recipe.instructions)]}\n\nCooking Time: {recipe.cooking_time}\nDifficulty: {recipe.difficulty}\nServings: {recipe.servings}')"

recipe-quick:
	@echo Generating quick recipe with sample ingredients...
	@python -c "from recipe_ai_engine import RecipeRequest, RecipeGenerator; request = RecipeRequest(ingredients=['chicken breast', 'rice', 'vegetables']); generator = RecipeGenerator(); recipe = generator.generate_recipe(request); print(f'Quick Recipe: {recipe.title}')"

# API server
api:
	@echo Starting FastAPI server on http://localhost:8000 ...
	uvicorn recipe_ai_engine.api.app:app --host 0.0.0.0 --port 8000

api-dev:
	@echo Starting FastAPI server (dev, auto-reload) on http://localhost:8000 ...
	uvicorn recipe_ai_engine.api.app:app --host 0.0.0.0 --port 8000 --reload

# Development commands
setup-model:
	@echo "Setting up fine-tuned recipe model..."
	python scripts/setup_fine_tuned_model.py

clean:
	@echo "Cleaning up temporary files..."
	@powershell -Command "Get-ChildItem -Recurse -Include '*.pyc' | Remove-Item -Force"
	@powershell -Command "Get-ChildItem -Recurse -Directory -Name '__pycache__' | ForEach-Object { Remove-Item -Recurse -Force $_ }"
	@powershell -Command "Get-ChildItem -Recurse -Include '*.log' | Remove-Item -Force"
	@echo "Cleanup complete!"

# Training installation commands
install-train-cpu:
	@echo "Installing training requirements (CPU only)..."
	python -m pip install -r requirements-train.txt

install-train-gpu:
	@echo "Installing training requirements with CUDA support..."
	python -m pip install -r requirements-train.txt --index-url https://download.pytorch.org/whl/cu121

# GPU detection
check-gpu:
	@echo "Checking GPU availability..."
	@python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# Default training install (GPU if available, CPU otherwise)
install-train: install-train-gpu

# Login to Hugging Face using token from .env (Windows PowerShell)
hf-login:
	@echo "Logging into Hugging Face via token from .env (HUGGINGFACE_HUB_TOKEN or HUGGINGFACE_TOKEN)..."
	@python -c "from dotenv import load_dotenv; load_dotenv(); import os,subprocess; t=os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HUGGINGFACE_TOKEN'); assert t, 'Set HUGGINGFACE_HUB_TOKEN or HUGGINGFACE_TOKEN in .env'; subprocess.run(['huggingface-cli','login','--token', t], check=True)"

# Training commands for different models (GPU/CPU variants)
# GPT2 Training
train-gpt2-cpu:
	@echo "Training GPT2 model on CPU (slow but works everywhere)..."
	python scripts/train_gpt2.py --dataset datasets/recipe_dataset_1000.json --output ./recipe-model-gpt2-cpu --epochs 1 --batch-size 2 --test

train-gpt2-gpu:
	@echo "Training GPT2 model on GPU (fast, requires CUDA)..."
	python scripts/train_gpt2.py --dataset datasets/recipe_dataset_50000.json --output ./recipe-model-gpt2-gpu --epochs 1 --batch-size 2 --test

train-gpt2-full-cpu:
	@echo "Training GPT2 model on CPU (full training with full dataset)..."
	python scripts/train_gpt2.py --dataset datasets/full_recipe_dataset.json --output ./recipe-model-gpt2-cpu --epochs 3 --batch-size 2

train-gpt2-full-gpu:
	@echo "Training GPT2 model on GPU (full training with full dataset)..."
	python scripts/train_gpt2.py --dataset datasets/full_recipe_dataset.json --output ./recipe-model-gpt2-gpu --epochs 3 --batch-size 1

# Mistral Training
train-mistral-cpu:
	@echo "Training Mistral model on CPU (slow, high VRAM models)..."
	python scripts/train_mistral.py --dataset datasets/recipe_dataset_1000.json --output ./recipe-model-mistral-cpu --epochs 1 --batch-size 1 --test

train-mistral-gpu:
	@echo "Training Mistral model on GPU (recommended for 7B models)..."
	python scripts/train_mistral.py --dataset datasets/recipe_dataset_1000.json --output ./recipe-model-mistral-gpu --epochs 1 --batch-size 2 --test

train-mistral-full-cpu:
	@echo "Training Mistral model on CPU (full training with full dataset)..."
	python scripts/train_mistral.py --dataset datasets/full_recipe_dataset.json --output ./recipe-model-mistral-cpu --epochs 3 --batch-size 1

train-mistral-full-gpu:
	@echo "Training Mistral model on GPU (full training with full dataset)..."
	python scripts/train_mistral.py --dataset datasets/full_recipe_dataset.json --output ./recipe-model-mistral-gpu --epochs 3 --batch-size 2

# Llama 2 Training
train-llama-cpu:
	@echo "Training Llama 2 model on CPU (very slow, not recommended)..."
	python scripts/train.py --dataset datasets/recipe_dataset_1000.json --model meta-llama/Llama-2-7b-hf --output ./recipe-model-llama-cpu --epochs 1 --batch-size 1 --learning-rate 2e-4 --test

train-llama-gpu:
	@echo "Training Llama 2 model on GPU (recommended for 7B models)..."
	python scripts/train.py --dataset datasets/recipe_dataset_1000.json --model meta-llama/Llama-2-7b-hf --output ./recipe-model-llama-gpu --epochs 1 --batch-size 2 --learning-rate 2e-4 --test

train-llama-full-cpu:
	@echo "Training Llama 2 model on CPU (full training with full dataset, very slow)..."
	python scripts/train.py --dataset datasets/full_recipe_dataset.json --model meta-llama/Llama-2-7b-hf --output ./recipe-model-llama-cpu --epochs 3 --batch-size 1 --learning-rate 2e-4

train-llama-full-gpu:
	@echo "Training Llama 2 model on GPU (full training with full dataset)..."
	python scripts/train.py --dataset datasets/full_recipe_dataset.json --model meta-llama/Llama-2-7b-hf --output ./recipe-model-llama-gpu --epochs 3 --batch-size 2 --learning-rate 2e-4

# Default training commands (GPU if available, CPU otherwise)
train-gpt2: train-gpt2-gpu
train-gpt2-full: train-gpt2-full-gpu
train-mistral: train-mistral-gpu
train-mistral-full: train-mistral-full-gpu
train-llama: train-llama-gpu
train-llama-full: train-llama-full-gpu

# Cleanup commands for cached models
clean-cache:
	@echo "Cleaning Hugging Face cache..."
	@powershell -Command "Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface\hub\models--* -ErrorAction SilentlyContinue"
	@echo "Cache cleaned!"

clean-models:
	@echo "Cleaning trained models..."
	@powershell -Command "Remove-Item -Recurse -Force ./recipe-model-* -ErrorAction SilentlyContinue"
	@echo "Models cleaned!"