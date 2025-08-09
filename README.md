# Recipe AI Engine 🍳

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Required-orange.svg)](https://ollama.ai/)

A professional, self-hosted AI service for generating cooking recipes based on available ingredients. Built with Python, Ollama, and modern AI models. Perfect for developers, home cooks, and anyone who wants to create recipes from whatever ingredients they have on hand.

## 🌟 Why Recipe AI Engine?

- **🔒 Privacy First**: Your data stays on your machine - no cloud dependencies
- **🚀 Easy Setup**: Simple installation with Ollama integration
- **🎯 Smart Recipes**: AI generates recipes that actually use your ingredients
- **⚡ Fast & Reliable**: Local processing means no network delays
- **🔧 Developer Friendly**: Clean API, comprehensive tests, professional structure
- **🌍 Open Source**: MIT licensed, community-driven development

## 🚀 Features

- **🤖 AI-Powered Recipe Generation**: Create recipes from any ingredient list with intelligent ingredient utilization
- **🔒 Self-Hosted**: Complete privacy and control over your data - no cloud dependencies
- **🎯 Smart Ingredient Usage**: AI ensures all provided ingredients are used in the recipe
- **🌍 Multi-Cuisine Support**: Generate recipes for various cuisines and dietary preferences
- **⚙️ Professional Structure**: Follows Python conventions and best practices
- **🔧 Easy Integration**: Simple API for building applications and services
- **📊 Quality Analysis**: Built-in testing and quality assessment for recipe generation
- **🌡️ Configurable AI**: Adjust temperature, creativity, and other AI parameters
- **📝 Structured Output**: Consistent JSON-formatted recipe responses

## 📋 Prerequisites

- **Python 3.10+** - Modern Python with type hints and async support
- **[Ollama](https://ollama.ai/)** - Local AI model serving platform
- **`llama2:7b` model** - Downloaded and ready in Ollama
- **Git** - For cloning and version control
- **Make** - For build automation (optional, but recommended)

## 🛠️ Installation

### Quick Start (Recommended)
```bash
# Clone the repository
git clone https://github.com/jonbarlo/recipe-ai-engine.git
cd recipe-ai-engine

# Install all requirements and package
make install

# Setup the fine-tuned recipe model (optional)
make setup-model

# Test the installation
make test
```

### Manual Installation
```bash
# Clone the repository
git clone https://github.com/jonbarlo/recipe-ai-engine.git
cd recipe-ai-engine

# Install all requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Setup the recipe model (optional)
python scripts/setup_fine_tuned_model.py
```

### Ollama Setup
```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai/download

# Download the base model
ollama pull llama2:7b

# Verify installation
ollama list
```

### Using a Fine-Tuned Model

If you have a locally fine-tuned model directory, point the setup script to it:

```bash
set FINE_TUNED_MODEL_PATH=./recipe-model
python scripts/setup_fine_tuned_model.py
```

If `FINE_TUNED_MODEL_PATH` (or `./recipe-model`) is not found, the script will fall back to `llama2:7b`.

## 🧪 Testing

```bash
# Run all tests
make test

# Test recipe generation
make test-generation

# Test recipe quality
make test-quality

# Test environment variables
python scripts/test_env.py
```

## 🧩 Training / Fine-tuning (Optional)

For on-device fine-tuning using Hugging Face + QLoRA:

### **Quick Setup**
```bash
# Install training requirements (GPU recommended)
make install-train-gpu

# Check GPU availability
make check-gpu

# Login to Hugging Face (for gated models like Llama 2)
make hf-login
```

### **Training Commands**

#### **GPT2 Training (Fast, Low VRAM)**
```bash
# Test training (1000 recipes)
make train-gpt2-gpu

# Full training (complete dataset)
make train-gpt2-full-gpu
```

#### **Mistral Training (Good Balance)**
```bash
# Test training (1000 recipes)
make train-mistral-gpu

# Full training (complete dataset)
make train-mistral-full-gpu
```

#### **Llama 2 Training (High Quality)**
```bash
# Test training (1000 recipes)
make train-llama-gpu

# Full training (complete dataset)
make train-llama-full-gpu
```

#### **CPU Training (Slower but Works Everywhere)**
```bash
# Install CPU requirements
make install-train-cpu

# Train on CPU
make train-gpt2-cpu
make train-mistral-cpu
make train-llama-cpu
```

### **Manual Training**
```bash
# Install training requirements
pip install -r requirements-train.txt --index-url https://download.pytorch.org/whl/cu121

# Run training
python scripts/train.py \
  --dataset datasets/recipe_dataset_1000.json \
  --model meta-llama/Llama-2-7b-hf \
  --output ./recipe-model \
  --epochs 1 \
  --batch-size 2 \
  --test
```

### Test Results
The test suite includes:
- **Generation Tests**: Verify package imports and basic recipe generation functionality
- **Quality Tests**: Evaluate AI model performance with edge cases and recipe quality
- **Quality Analysis**: Assess recipe generation quality and consistency
- **Environment Tests**: Verify configuration system

## 🍽️ Usage Examples

### Quick Recipe Generation
```bash
# Generate a recipe with sample ingredients
make recipe-quick
```

### Interactive Recipe Generation
```bash
# Start interactive mode
make recipe
# Enter ingredients when prompted
```

### Command Line Usage
```bash
# Generate a recipe with specific ingredients
python -c "
from recipe_ai_engine import RecipeRequest, RecipeGenerator
request = RecipeRequest(ingredients=['chicken', 'rice', 'vegetables'])
generator = RecipeGenerator()
recipe = generator.generate_recipe(request)
print(f'Recipe: {recipe.title}')
"
```

### Ollama Usage

```bash
# List available models
ollama list

# Pull a model
ollama pull llama2:7b

# Run a model
ollama run llama2:7b "Generate a simple recipe with chicken, rice, and vegetables in JSON format"

# Run a model with a custom prompt
ollama run llama2:7b "Generate a simple recipe with chicken, rice, and vegetables in JSON format"

# Run a model with a custom prompt and output format
ollama run llama2:7b "Generate a simple recipe with chicken, rice, and vegetables in JSON format" --format json

# Run a model with a custom prompt and output format
ollama run llama2:7b "Generate a simple recipe with chicken, rice, and vegetables in JSON format" --format json

# Run a model with a custom prompt and output format
ollama run llama2:7b "Generate a simple recipe with chicken, rice, and vegetables in JSON format" --format json
```

Ollama custom model
```bash
# Create a custom model
ollama create recipe-ai:latest --model llama2:7b --base-url http://localhost:11434

ollama rm recipe-ai
```

### Python API Usage

```python
from recipe_ai_engine import RecipeRequest, RecipeGenerator

# Create a recipe request
request = RecipeRequest(
    ingredients=["chicken breast", "rice", "vegetables"],
    cuisine="Asian",
    servings=4,
    difficulty="medium"
)

# Generate the recipe
generator = RecipeGenerator()
recipe = generator.generate_recipe(request)

# Access recipe details
print(f"Recipe: {recipe.title}")
print(f"Cooking Time: {recipe.cooking_time}")
print(f"Difficulty: {recipe.difficulty}")
print(f"Servings: {recipe.servings}")

# Print ingredients
for ingredient in recipe.ingredients:
    print(f"- {ingredient['item']}: {ingredient['amount']}")

# Print instructions
for i, step in enumerate(recipe.instructions, 1):
    print(f"{i}. {step}")
```

## Web API (FastAPI)

Start the API server:

```bash
# Production-like (no reload)
make api

# Development (auto-reload on code changes)
make api-dev
```

Interactive docs will be available at `http://localhost:8000/docs`.

Endpoints:
- `GET /health`
- `POST /recipes/generate`

### Sample request

Using curl:
```bash
curl -X POST http://localhost:8000/recipes/generate \
  -H "Content-Type: application/json" \
  -d '{
        "ingredients": ["chicken", "rice", "vegetables"],
        "cuisine_type": "Asian",
        "difficulty_level": "medium",
        "serving_size": 2
      }'
```

Using Python:
```python
import requests

payload = {
  "ingredients": ["chicken", "rice", "vegetables"],
  "cuisine_type": "Asian",
  "difficulty_level": "medium",
  "serving_size": 2
}

resp = requests.post("http://localhost:8000/recipes/generate", json=payload, timeout=60)
resp.raise_for_status()
print(resp.json())
```

### Advanced Usage

```python
from recipe_ai_engine import RecipeRequest, RecipeGenerator
from recipe_ai_engine.ai import OllamaRecipeGenerator

# Custom AI generator with specific model
custom_generator = OllamaRecipeGenerator(model_name="recipe-ai:latest")
generator = RecipeGenerator(ai_generator=custom_generator)

# Generate multiple recipe variations
recipes = generator.generate_multiple_recipes(request, count=3)

for i, recipe in enumerate(recipes, 1):
    print(f"\nRecipe Variation {i}:")
    print(f"Title: {recipe.title}")
    print(f"Difficulty: {recipe.difficulty}")
```

## 📁 Project Structure

```
recipe-ai-engine/
├── recipe_ai_engine/          # Main package
│   ├── core/                  # Core functionality
│   │   ├── models.py          # Pydantic models
│   │   ├── exceptions.py      # Custom exceptions
│   │   └── config.py          # Configuration management
│   ├── ai/                    # AI model components
│   │   ├── base.py            # Base AI interface
│   │   ├── ollama_client.py   # Ollama implementation
│   │   └── prompts.py         # Prompt generation
│   └── recipes/               # Recipe-specific logic
│       ├── generator.py        # Main orchestrator
│       └── validator.py        # Recipe validation
├── docs/                      # Documentation
├── tests/                     # Test suite
├── scripts/                   # Utility scripts
├── datasets/                  # Recipe datasets
├── requirements.txt           # Main dependencies

├── Makefile                   # Build and test commands
└── README.md                  # This file
```

## 🔧 Configuration

The engine uses Pydantic Settings for configuration. Key settings:

```python
# Default configuration (can be overridden with environment variables)
ollama_base_url: str = "http://localhost:11434"
ollama_model_name: str = "llama2:7b"
temperature: float = 0.3
top_p: float = 0.9
max_tokens: int = 800
```

### Environment Variables

You can override settings using environment variables:

```bash
# Set the model to use
set OLLAMA_MODEL_NAME=recipe-ai:latest

# Adjust AI parameters
set AI_TEMPERATURE=0.7
set AI_TOP_P=0.8
set AI_MAX_TOKENS=1000

# Change timeouts
set REQUEST_TIMEOUT=120
set CONNECTION_TIMEOUT=10
```

### Available Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL_NAME` | `llama2:7b` | Ollama model to use |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `AI_TEMPERATURE` | `0.3` | AI model temperature |
| `AI_TOP_P` | `0.9` | AI model top-p parameter |
| `AI_MAX_TOKENS` | `800` | Maximum tokens to generate |
| `REQUEST_TIMEOUT` | `60` | Request timeout in seconds |
| `CONNECTION_TIMEOUT` | `5` | Connection timeout in seconds |

## 🧪 Available Commands

```bash
# Installation
make install        # Install all requirements and package

# Testing
make test           # Run all tests
make test-generation # Test recipe generation
make test-quality   # Test recipe quality

# Recipe Generation
make recipe         # Interactive recipe generation
make recipe-quick   # Quick recipe with sample ingredients

# Training Installation
make install-train-cpu # Install training requirements (CPU only)
make install-train-gpu # Install training requirements (GPU/CUDA)
make check-gpu      # Check GPU availability

# Training Commands
make train-gpt2-cpu/gpu     # Train GPT2 (CPU/GPU)
make train-mistral-cpu/gpu  # Train Mistral (CPU/GPU)
make train-llama-cpu/gpu    # Train Llama 2 (CPU/GPU)
make train-gpt2/mistral/llama # Default GPU training

# Development
make setup-model    # Setup fine-tuned recipe model
make hf-login       # Login to Hugging Face
make clean          # Clean up temporary files
make clean-cache    # Clean Hugging Face cache
make clean-models   # Clean trained models
make help           # Show all available commands
```

## 📊 Model Performance

The fine-tuned `recipe-ai` model provides:

- **Better Recipe Quality**: More detailed and accurate instructions
- **Consistent Formatting**: Proper ingredient and instruction structure
- **Cuisine Awareness**: Better understanding of different cooking styles
- **Ingredient Utilization**: More efficient use of provided ingredients

## 🔍 Troubleshooting

### Common Issues

1. **Ollama not running**
   ```bash
   # Start Ollama
   ollama serve
   ```

2. **Model not found**
   ```bash
   # Download the base model
   ollama pull llama2:7b
   
   # Setup the recipe model
   make setup-model
   ```

3. **Import errors**
   ```bash
   # Reinstall requirements
   make install-all
   ```

## 🤝 Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Follow Python conventions

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built with [Ollama](https://ollama.ai/)
- Uses [Llama 2](https://huggingface.co/meta-llama) as the base model
- Follows Python packaging best practices

## CV
Python 3, OLLAMA, LLMs (GPT2, Llama2, Mistral), Pydantic, FastAPI, Hugging Face, QLoRA, PyTorch, Transformers, CUDA, GPU, CPU, Linux, Windows, MacOS, Docker, Git, Makefile, VSCode, PyCharm, Cursor