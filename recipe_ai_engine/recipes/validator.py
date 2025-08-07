"""
Recipe validation and enhancement logic
"""

from typing import List
from ..core.models import RecipeRequest, RecipeResponse
from ..core.exceptions import RecipeValidationError


class RecipeValidator:
    """Validates and enhances recipes"""
    
    def validate_request(self, request: RecipeRequest) -> None:
        """Validate the recipe request"""
        
        if not request.ingredients:
            raise RecipeValidationError("At least one ingredient is required")
        
        if len(request.ingredients) < 1:
            raise RecipeValidationError("At least one ingredient is required")
        
        if request.serving_size < 1:
            raise RecipeValidationError("Serving size must be at least 1")
        
        if request.serving_size > 20:
            raise RecipeValidationError("Serving size cannot exceed 20")
        
        valid_difficulties = ["easy", "medium", "hard"]
        if request.difficulty_level not in valid_difficulties:
            raise RecipeValidationError(f"Difficulty must be one of: {valid_difficulties}")
    
    def validate_and_enhance_recipe(self, recipe: RecipeResponse, request: RecipeRequest) -> RecipeResponse:
        """Validate and enhance the generated recipe"""
        
        # Validate required fields
        if not recipe.title:
            recipe.title = "Generated Recipe"
        
        if not recipe.ingredients:
            raise RecipeValidationError("Recipe must have ingredients")
        
        if not recipe.instructions:
            raise RecipeValidationError("Recipe must have instructions")
        
        # Enhance recipe with missing information
        recipe = self._enhance_recipe(recipe, request)
        
        return recipe
    
    def _enhance_recipe(self, recipe: RecipeResponse, request: RecipeRequest) -> RecipeResponse:
        """Enhance recipe with additional information"""
        
        # Ensure servings match request
        if recipe.servings != request.serving_size:
            recipe.servings = request.serving_size
        
        # Ensure difficulty matches request
        if recipe.difficulty != request.difficulty_level:
            recipe.difficulty = request.difficulty_level
        
        # Ensure cuisine type is set
        if not recipe.cuisine_type and request.cuisine_type:
            recipe.cuisine_type = request.cuisine_type
        
        # Add dietary notes if not present
        if not recipe.dietary_notes and request.dietary_restrictions:
            recipe.dietary_notes = f"Dietary restrictions: {', '.join(request.dietary_restrictions)}"
        
        return recipe
    
    def validate_ingredients(self, ingredients: List[str]) -> List[str]:
        """Validate and clean ingredient list"""
        
        cleaned_ingredients = []
        for ingredient in ingredients:
            # Remove extra whitespace
            cleaned = ingredient.strip()
            if cleaned:
                cleaned_ingredients.append(cleaned)
        
        return cleaned_ingredients 