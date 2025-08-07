"""
Prompt generation for recipe AI models
"""

from typing import List
from ..core.models import RecipeRequest
from ..core.exceptions import PromptGenerationError


class PromptGenerator:
    """Generates prompts for recipe generation"""
    
    @staticmethod
    def create_recipe_prompt(request: RecipeRequest) -> str:
        """Create a structured prompt for recipe generation"""
        
        try:
            ingredients_str = ", ".join(request.ingredients)
            
            # Create specialized prompt based on request characteristics
            prompt = PromptGenerator._create_specialized_prompt(request, ingredients_str)
            
            return prompt
            
        except Exception as e:
            raise PromptGenerationError(f"Failed to create recipe prompt: {e}")
    
    @staticmethod
    def _create_specialized_prompt(request: RecipeRequest, ingredients_str: str) -> str:
        """Create a specialized prompt based on request characteristics"""
        
        # Base prompt with better JSON formatting guidance
        base_prompt = f"""You are an expert recipe generator. Create a delicious recipe using these ingredients: {ingredients_str}.

CRITICAL: Respond ONLY with a valid JSON object. No extra text before or after the JSON.

IMPORTANT JSON FORMAT RULES:
- Use double quotes for all strings
- No trailing commas
- All ingredient amounts must be strings
- Instructions must be an array of strings

JSON FORMAT:
{{
    "title": "Creative Recipe Name",
    "ingredients": [
        {{"item": "ingredient name", "amount": "quantity"}}
    ],
    "instructions": [
        "Step 1: Detailed instruction",
        "Step 2: Detailed instruction"
    ],
    "cooking_time": "X minutes",
    "preparation_time": "X minutes",
    "difficulty": "{request.difficulty_level}",
    "servings": {request.serving_size},
    "cuisine_type": "{request.cuisine_type or 'General'}",
    "dietary_notes": "{', '.join(request.dietary_restrictions) if request.dietary_restrictions else 'Standard diet'}"
}}

"""
        
        # Add specialized guidance based on request characteristics
        if len(request.ingredients) > 10:
            base_prompt += "NOTE: This is a complex recipe with many ingredients. Create detailed, step-by-step instructions.\n\n"
        elif len(request.ingredients) <= 3:
            base_prompt += "NOTE: This is a simple recipe with few ingredients. Focus on creative ways to use these ingredients.\n\n"
        
        # Add cuisine-specific guidance
        if request.cuisine_type and request.cuisine_type.lower() != "any":
            base_prompt += f"NOTE: Create a {request.cuisine_type} style recipe with appropriate cooking techniques.\n\n"
        
        # Add dietary restriction guidance
        if request.dietary_restrictions:
            restrictions_str = ", ".join(request.dietary_restrictions)
            base_prompt += f"NOTE: This recipe must be {restrictions_str}. Ensure all ingredients and techniques are appropriate.\n\n"
        
        # Add difficulty-specific guidance
        if request.difficulty_level == "easy":
            base_prompt += "NOTE: Keep instructions simple and beginner-friendly.\n\n"
        elif request.difficulty_level == "hard":
            base_prompt += "NOTE: Include advanced cooking techniques and detailed instructions.\n\n"
        
        base_prompt += "Recipe:"
        
        return base_prompt
    
    @staticmethod
    def create_ingredient_substitution_prompt(ingredients: List[str], missing_ingredient: str) -> str:
        """Create a prompt for ingredient substitution suggestions"""
        
        ingredients_str = ", ".join(ingredients)
        
        prompt = f"""You are a cooking expert. Suggest substitutes for "{missing_ingredient}" using these available ingredients: {ingredients_str}.

Respond with a JSON object:
{{
    "substitution": "suggested substitute",
    "reason": "why this works as a substitute",
    "adjustments": "any recipe adjustments needed"
}}"""

        return prompt 