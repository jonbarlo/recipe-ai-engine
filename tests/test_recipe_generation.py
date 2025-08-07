"""
Test script for Recipe AI Engine generation functionality
Tests core recipe generation capabilities and ensures all components work together.
"""

from recipe_ai_engine import RecipeRequest, RecipeResponse, RecipeGenerator, OllamaRecipeGenerator


def test_recipe_generation():
    """Test recipe generation functionality and project structure"""
    
    print("🧪 Testing Recipe AI Engine Generation")
    print("=" * 50)
    
    # Test 1: Import and basic functionality
    print("\n1. Testing imports and basic functionality...")
    
    try:
        # Create a recipe request
        request = RecipeRequest(
            ingredients=["chicken breast", "rice", "vegetables"],
            cuisine_type="Asian",
            serving_size=2,
            difficulty_level="easy"
        )
        
        print("✅ RecipeRequest created successfully")
        print(f"   Ingredients: {request.ingredients}")
        print(f"   Cuisine: {request.cuisine_type}")
        print(f"   Servings: {request.serving_size}")
        
    except Exception as e:
        print(f"❌ Failed to create RecipeRequest: {e}")
        return
    
    # Test 2: AI Model connection
    print("\n2. Testing AI model connection...")
    
    try:
        generator = OllamaRecipeGenerator()
        
        if generator.test_connection():
            print("✅ Ollama connection successful")
        else:
            print("❌ Ollama connection failed")
            return
            
    except Exception as e:
        print(f"❌ Failed to test connection: {e}")
        return
    
    # Test 3: Recipe generation
    print("\n3. Testing recipe generation...")
    
    try:
        recipe = generator.generate_recipe(request)
        
        print("✅ Recipe generated successfully")
        print(f"   Title: {recipe.title}")
        print(f"   Ingredients: {len(recipe.ingredients)} items")
        print(f"   Instructions: {len(recipe.instructions)} steps")
        print(f"   Cooking time: {recipe.cooking_time}")
        print(f"   Difficulty: {recipe.difficulty}")
        print(f"   Servings: {recipe.servings}")
        
    except Exception as e:
        print(f"❌ Failed to generate recipe: {e}")
        return
    
    # Test 4: Main RecipeGenerator with validation
    print("\n4. Testing main RecipeGenerator with validation...")
    
    try:
        main_generator = RecipeGenerator()
        
        recipe = main_generator.generate_recipe(request)
        
        print("✅ Main generator with validation successful")
        print(f"   Title: {recipe.title}")
        print(f"   Cuisine: {recipe.cuisine_type}")
        print(f"   Dietary notes: {recipe.dietary_notes}")
        
    except Exception as e:
        print(f"❌ Failed to use main generator: {e}")
        return
    
    print("\n✅ All tests passed! Recipe generation is working correctly.")


if __name__ == "__main__":
    test_recipe_generation() 