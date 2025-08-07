"""
Main recipe generator that orchestrates AI model and validation
"""

from typing import List
from ..core.models import RecipeRequest, RecipeResponse
from ..core.exceptions import RecipeGenerationError
from ..ai.ollama_client import OllamaRecipeGenerator
from .validator import RecipeValidator


class RecipeGenerator:
    """Main recipe generator with validation and enhancement"""
    
    def __init__(self, ai_generator: OllamaRecipeGenerator = None):
        self.ai_generator = ai_generator or OllamaRecipeGenerator()
        self.validator = RecipeValidator()
    
    def generate_recipe(self, request: RecipeRequest) -> RecipeResponse:
        """Generate a validated recipe"""
        
        # Validate input request
        self.validator.validate_request(request)
        
        # Generate recipe using AI model
        recipe = self.ai_generator.generate_recipe(request)
        
        # Validate and enhance the generated recipe
        recipe = self.validator.validate_and_enhance_recipe(recipe, request)
        
        return recipe
    
    def generate_multiple_recipes(self, request: RecipeRequest, count: int = 3) -> List[RecipeResponse]:
        """Generate multiple recipe variations"""
        
        recipes = []
        for i in range(count):
            try:
                recipe = self.generate_recipe(request)
                recipes.append(recipe)
            except RecipeGenerationError as e:
                # Log error but continue with other recipes
                print(f"Failed to generate recipe {i+1}: {e}")
                continue
        
        return recipes
    
    def test_connection(self) -> bool:
        """Test if the recipe generator is working"""
        return self.ai_generator.test_connection() 