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
	@echo.
	@echo Development:
	@echo   setup-model    - Setup fine-tuned recipe model
	@echo   hf-login       - Login to Hugging Face using token from .env
	@echo   clean          - Clean up temporary files
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
	uvicorn recipe_ai_engine.api.routes:app --host 0.0.0.0 --port 8000

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

install-train:
	@echo "Installing training requirements..."
	python -m pip install -r requirements-train.txt

# Login to Hugging Face using token from .env (Windows PowerShell)
hf-login:
	@echo "Logging into Hugging Face via token from .env (HUGGINGFACE_HUB_TOKEN or HUGGINGFACE_TOKEN)..."
	@python -c "from dotenv import load_dotenv; load_dotenv(); import os,subprocess; t=os.getenv('HUGGINGFACE_HUB_TOKEN') or os.getenv('HUGGINGFACE_TOKEN'); assert t, 'Set HUGGINGFACE_HUB_TOKEN or HUGGINGFACE_TOKEN in .env'; subprocess.run(['huggingface-cli','login','--token', t], check=True)"

train-model-mistral:
	@echo "Training Mistral model..."
	python scripts/train.py --dataset datasets/recipe_dataset_1000.json --model mistralai/Mistral-7B-Instruct-v0.3 --output F:\recipe-mistral-model --epochs 1 --batch-size 2 --test

remove-train-mistral:
	@echo "Removing Mistral model..."
	@powershell -Command "Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface\hub\models--mistralai--Mistral-7B-Instruct-v0.3"

remove-train-llama:
	@echo "Removing Llama model..."
	@powershell -Command "Remove-Item -Recurse -Force $env:USERPROFILE\.cache\huggingface\hub\models--meta-llama--Llama-2-7b-hf"

remove-train-qwen2:

train-model-llama:
	@echo "Training Llama model..."
	python scripts/train.py --dataset datasets/recipe_dataset_1000.json --model meta-llama/Llama-2-7b-hf --output ./recipe-model --epochs 1 --batch-size 2 --learning-rate 2e-4 --test
train-model-qwen2:
	@echo "Training GPT-4o model..."
	python scripts/train.py --dataset datasets/recipe_dataset_1000.json --model Qwen/Qwen2-7B-Instruct --output ./recipe-model --epochs 1 --batch-size 2 --learning-rate 2e-4 --test