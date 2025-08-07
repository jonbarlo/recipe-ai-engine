"""
Base interface for recipe generators
"""

from typing import Protocol
from ..core.models import RecipeRequest, RecipeResponse


class RecipeGenerator(Protocol):
    """Protocol for recipe generation implementations"""
    
    def generate_recipe(self, request: RecipeRequest) -> RecipeResponse:
        """Generate a recipe from the given request"""
        ...
    
    def test_connection(self) -> bool:
        """Test if the AI model is accessible"""
        ... 