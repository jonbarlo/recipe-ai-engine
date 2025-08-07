"""
Ollama client implementation for recipe generation
"""

import json
import re
import requests
from typing import Dict, Optional
from ..core.models import RecipeRequest, RecipeResponse
from ..core.exceptions import OllamaConnectionError, JSONParsingError, ModelResponseError
from ..core.config import settings
from .base import RecipeGenerator
from .prompts import PromptGenerator


class OllamaRecipeGenerator:
    """Recipe generator using Ollama AI models"""
    
    def __init__(self, model_name: str = None, base_url: str = None):
        self.model_name = model_name or settings.ollama_model_name
        self.base_url = base_url or settings.ollama_base_url
        self.api_url = f"{self.base_url}/api/generate"
        self.prompt_generator = PromptGenerator()
    
    def generate_recipe(self, request: RecipeRequest) -> RecipeResponse:
        """Generate a recipe using the AI model"""
        
        prompt = self.prompt_generator.create_recipe_prompt(request)
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": settings.temperature,
                "top_p": settings.top_p,
                "max_tokens": settings.max_tokens
            }
        }
        
        try:
            response = requests.post(
                self.api_url, 
                json=payload, 
                timeout=settings.request_timeout
            )
            response.raise_for_status()
            
            result = response.json()
            recipe_text = result.get("response", "")
            
            # Try to extract JSON from the response
            recipe_data = self._extract_json_from_response(recipe_text)
            
            if recipe_data:
                return RecipeResponse(**recipe_data)
            else:
                # Fallback: parse the text response manually
                return self._parse_text_response(recipe_text, request)
                
        except requests.exceptions.RequestException as e:
            raise OllamaConnectionError(f"Failed to communicate with Ollama: {e}")
        except Exception as e:
            raise ModelResponseError(f"Failed to generate recipe: {e}")
    
    def _extract_json_from_response(self, text: str) -> Optional[Dict]:
        """Extract JSON from the model response"""
        try:
            # Look for JSON in the response
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                
                # Try to fix common JSON issues
                json_str = self._fix_json_issues(json_str)
                
                return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            # Don't raise exception, just return None for fallback
            print(f"JSON parsing failed: {e}")
            return None
        
        return None
    
    def _fix_json_issues(self, json_str: str) -> str:
        """Fix common JSON issues from AI model responses"""
        
        # Remove trailing commas before closing brackets/braces
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Only fix trailing commas, don't modify quotes or structure
        # The AI is already generating valid JSON structure
        
        return json_str
    
    def _parse_text_response(self, text: str, request: RecipeRequest) -> RecipeResponse:
        """Parse text response when JSON extraction fails"""
        
        lines = text.split('\n')
        
        # Extract title more intelligently
        title = self._extract_title_from_text(text, request)
        
        # Parse ingredients and instructions more robustly
        ingredients, instructions = self._parse_ingredients_and_instructions(text, request)
        
        # Validate and enhance the parsed data
        ingredients = self._validate_ingredients(ingredients, request.ingredients)
        instructions = self._validate_instructions(instructions, request.ingredients)
        
        return RecipeResponse(
            title=title,
            ingredients=ingredients,
            instructions=instructions,
            cooking_time="30 minutes",
            preparation_time="15 minutes",
            difficulty=request.difficulty_level,
            servings=request.serving_size,
            cuisine_type=request.cuisine_type,
            dietary_notes="Generated recipe"
        )
    
    def _extract_title_from_text(self, text: str, request: RecipeRequest) -> str:
        """Extract a meaningful title from the text response"""
        lines = text.split('\n')
        
        # Look for lines that might be titles
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that are clearly not titles
            if any(skip in line.lower() for skip in ['ingredient', 'instruction', 'step', 'directions', 'preparation']):
                continue
            
            # Look for lines that could be recipe titles
            if (len(line) > 5 and len(line) < 100 and 
                not line.startswith('*') and not line.startswith('-') and 
                not line[0].isdigit() and ':' not in line):
                
                # Clean up the title
                title = line.replace('Recipe:', '').replace('Title:', '').strip()
                if title and title.lower() != 'generated recipe':
                    return title
        
        # If no good title found, create one based on ingredients
        if len(request.ingredients) <= 3:
            return f"{' '.join(request.ingredients).title()} Recipe"
        else:
            return f"{request.ingredients[0].title()} and More Recipe"
    
    def _parse_ingredients_and_instructions(self, text: str, request: RecipeRequest) -> tuple:
        """Parse ingredients and instructions from text"""
        lines = text.split('\n')
        
        ingredients = []
        instructions = []
        in_ingredients = False
        in_instructions = False
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if any(keyword in line.lower() for keyword in ['ingredient', 'ingredients']):
                in_ingredients = True
                in_instructions = False
                continue
            elif any(keyword in line.lower() for keyword in ['instruction', 'step', 'direction', 'method']):
                in_ingredients = False
                in_instructions = True
                continue
            
            # Parse ingredients
            if in_ingredients and line:
                if not line.startswith(('*', '-', '1.', '2.', '3.')):
                    # Try to extract ingredient and amount
                    ingredient_data = self._parse_ingredient_line(line)
                    if ingredient_data:
                        ingredients.append(ingredient_data)
            
            # Parse instructions
            elif in_instructions and line:
                if (line.startswith(('*', '-', '1.', '2.', '3.', '4.', '5.')) or 
                    any(word in line.lower() for word in ['heat', 'cook', 'add', 'mix', 'stir', 'preheat'])):
                    instructions.append(line)
        
        return ingredients, instructions
    
    def _parse_ingredient_line(self, line: str) -> dict:
        """Parse a single ingredient line"""
        # Remove common prefixes
        line = re.sub(r'^[\d\-*\.\s]+', '', line)
        
        # Try to extract amount and item
        amount_match = re.search(r'^([\d\/\s]+(?:cup|tbsp|tsp|oz|lb|g|kg|ml|cl|piece|pieces|slice|slices|can|cans|bunch|bunches|head|heads|medium|large|small|whole|half|quarter|third|tablespoon|teaspoon|ounce|pound|gram|kilogram|milliliter|centiliter))', line, re.IGNORECASE)
        
        if amount_match:
            amount = amount_match.group(1).strip()
            item = line[amount_match.end():].strip()
            if item:
                return {"item": item, "amount": amount}
        
        # If no amount found, use the whole line as item
        return {"item": line, "amount": "as needed"}
    
    def _validate_ingredients(self, ingredients: list, request_ingredients: list) -> list:
        """Validate and enhance ingredients list"""
        if not ingredients:
            # Create ingredients from request
            return [{"item": ingredient, "amount": "as needed"} for ingredient in request_ingredients]
        
        # Ensure all request ingredients are included
        used_items = {ing['item'].lower() for ing in ingredients}
        missing_ingredients = []
        
        for req_ing in request_ingredients:
            if req_ing.lower() not in used_items:
                missing_ingredients.append({"item": req_ing, "amount": "as needed"})
        
        return ingredients + missing_ingredients
    
    def _validate_instructions(self, instructions: list, request_ingredients: list) -> list:
        """Validate and enhance instructions"""
        if not instructions:
            # Create basic instructions that use the ingredients
            basic_instructions = [
                f"1. Prepare and wash all ingredients",
                f"2. Heat a pan or pot over medium heat",
                f"3. Add {request_ingredients[0]} and cook for 2-3 minutes",
            ]
            
            if len(request_ingredients) > 1:
                basic_instructions.append(f"4. Add {', '.join(request_ingredients[1:])} and continue cooking")
            
            basic_instructions.extend([
                "5. Season with salt and pepper to taste",
                "6. Serve hot and enjoy!"
            ])
            
            return basic_instructions
        
        # Ensure instructions mention the ingredients
        instruction_text = ' '.join(instructions).lower()
        used_ingredients = []
        
        for ingredient in request_ingredients:
            if ingredient.lower() in instruction_text:
                used_ingredients.append(ingredient)
        
        # If no ingredients are used, add a note
        if not used_ingredients and request_ingredients:
            instructions.append(f"Note: Make sure to use all the provided ingredients: {', '.join(request_ingredients)}")
        
        return instructions
    
    def test_connection(self) -> bool:
        """Test if Ollama is running and accessible"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags", 
                timeout=settings.connection_timeout
            )
            return response.status_code == 200
        except:
            return False 