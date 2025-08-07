"""
Recipe AI Engine - Self-hosted recipe generation service
"""

__version__ = "0.1.0"

from .core.models import RecipeRequest, RecipeResponse
from .recipes.generator import RecipeGenerator
from .ai.ollama_client import OllamaRecipeGenerator

__all__ = [
    "RecipeRequest",
    "RecipeResponse", 
    "RecipeGenerator",
    "OllamaRecipeGenerator"
] 