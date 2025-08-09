#!/usr/bin/env python3
"""
Test script to evaluate recipe generation quality and AI model performance
with complex recipe requests and edge cases. Analyzes recipe quality, consistency,
and ingredient utilization across various scenarios.
"""

import json
import time
from recipe_ai_engine import RecipeRequest, RecipeGenerator

def test_recipe_quality():
    """Test recipe generation quality with various challenging scenarios."""
    
    print("Testing Recipe Generation Quality")
    print("=" * 50)
    
    # Initialize the recipe generator
    generator = RecipeGenerator()
    
    # Test cases that might expose limitations
    test_cases = [
        # 1. Very limited ingredients (edge case)
        {
            "name": "Minimal Ingredients",
            "ingredients": ["salt", "water"],
            "description": "Testing with only basic ingredients"
        },
        
        # 2. Unusual ingredient combinations
        {
            "name": "Unusual Combination",
            "ingredients": ["banana", "sardines", "chocolate", "pickles"],
            "description": "Testing with ingredients that don't typically go together"
        },
        
        # 3. Many ingredients (complex recipe)
        {
            "name": "Complex Recipe",
            "ingredients": ["chicken", "rice", "onion", "garlic", "ginger", "soy sauce", 
                          "sesame oil", "carrots", "bell peppers", "mushrooms", "broccoli",
                          "eggs", "green onions", "sesame seeds", "chili flakes"],
            "description": "Testing with many ingredients for complex recipe"
        },
        
        # 4. Dietary restrictions
        {
            "name": "Vegan Request",
            "ingredients": ["tofu", "vegetables", "rice"],
            "description": "Testing vegan recipe generation"
        },
        
        # 5. Specific cuisine style
        {
            "name": "Italian Style",
            "ingredients": ["pasta", "tomato", "basil", "olive oil", "garlic"],
            "description": "Testing Italian cuisine style"
        },
        
        # 6. Dessert with unusual ingredients
        {
            "name": "Dessert with Vegetables",
            "ingredients": ["zucchini", "chocolate", "eggs", "flour", "sugar"],
            "description": "Testing dessert with vegetables"
        },
        
        # 7. Breakfast with dinner ingredients
        {
            "name": "Breakfast with Dinner Ingredients",
            "ingredients": ["salmon", "quinoa", "spinach", "eggs", "avocado"],
            "description": "Testing breakfast recipe with dinner ingredients"
        },
        
        # 8. Very specific cooking method request
        {
            "name": "Air Fryer Recipe",
            "ingredients": ["chicken wings", "potato", "oil", "spices"],
            "description": "Testing specific cooking method (air fryer)"
        },
        
        # 9. Seasonal/occasion specific
        {
            "name": "Holiday Recipe",
            "ingredients": ["turkey", "cranberries", "sweet potato", "herbs"],
            "description": "Testing holiday-themed recipe"
        },
        
        # 10. Health-focused
        {
            "name": "Low-Carb Recipe",
            "ingredients": ["cauliflower", "chicken", "cheese", "butter"],
            "description": "Testing low-carb recipe generation"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['name']}")
        print(f"Description: {test_case['description']}")
        print(f"Ingredients: {', '.join(test_case['ingredients'])}")
        print("-" * 40)
        
        try:
            # Create request
            request = RecipeRequest(
                ingredients=test_case['ingredients'],
                cuisine_type="any",
                dietary_restrictions=[],
                cooking_time=30,
                difficulty_level="medium",
                servings=4
            )
            
            # Time the generation
            start_time = time.time()
            recipe = generator.generate_recipe(request)
            generation_time = time.time() - start_time
            
            # Analyze the result
            analysis = analyze_recipe_quality(recipe, test_case)
            
            result = {
                "test_case": test_case['name'],
                "ingredients": test_case['ingredients'],
                "recipe": recipe.model_dump(),
                "generation_time": generation_time,
                "analysis": analysis,
                "success": True
            }
            
            print(f"OK: Generated in {generation_time:.2f}s")
            print(f"Title: {recipe.title}")
            print(f"Instructions: {len(recipe.instructions)} steps")
            print(f"Quality Score: {analysis['quality_score']}/10")
            print(f"Issues: {', '.join(analysis['issues'])}")
            
        except Exception as e:
            print(f"FAIL: {str(e)}")
            result = {
                "test_case": test_case['name'],
                "ingredients": test_case['ingredients'],
                "error": str(e),
                "success": False
            }
        
        results.append(result)
        time.sleep(1)  # Small delay between requests
    
    # Summary analysis
    print("\n" + "=" * 50)
    print("SUMMARY ANALYSIS")
    print("=" * 50)
    
    successful_tests = [r for r in results if r['success']]
    failed_tests = [r for r in results if not r['success']]
    
    print(f"Successful tests: {len(successful_tests)}/{len(test_cases)}")
    print(f"Failed tests: {len(failed_tests)}/{len(test_cases)}")
    
    if successful_tests:
        avg_time = sum(r['generation_time'] for r in successful_tests) / len(successful_tests)
        avg_quality = sum(r['analysis']['quality_score'] for r in successful_tests) / len(successful_tests)
        
        print(f"Average generation time: {avg_time:.2f}s")
        print(f"Average quality score: {avg_quality:.1f}/10")
        
        # Most common issues
        all_issues = []
        for r in successful_tests:
            all_issues.extend(r['analysis']['issues'])
        
        if all_issues:
            from collections import Counter
            issue_counts = Counter(all_issues)
            print("\nMost common issues:")
            for issue, count in issue_counts.most_common(5):
                print(f"  - {issue}: {count} times")
    
    if failed_tests:
        print("\nFailed test cases:")
        for test in failed_tests:
            print(f"  - {test['test_case']}: {test['error']}")
    
    # Save detailed results
    import os
    results_file = os.path.join("tests", "recipe_quality_test_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {results_file}")
    
    return results

def analyze_recipe_quality(recipe, test_case):
    """Analyze the quality of a generated recipe."""
    issues = []
    quality_score = 10  # Start with perfect score
    
    # Check title quality
    if not recipe.title or recipe.title.lower() in ["generated recipe", "recipe", "untitled"]:
        issues.append("Generic title")
        quality_score -= 2
    
    # Check instructions
    if not recipe.instructions or len(recipe.instructions) < 2:
        issues.append("Too few instructions")
        quality_score -= 3
    
    if len(recipe.instructions) > 15:
        issues.append("Too many instructions")
        quality_score -= 1
    
    # Check ingredient usage
    used_ingredients = set()
    for instruction in recipe.instructions:
        for ingredient in test_case['ingredients']:
            if ingredient.lower() in instruction.lower():
                used_ingredients.add(ingredient)
    
    unused_ingredients = set(test_case['ingredients']) - used_ingredients
    if unused_ingredients:
        issues.append(f"Unused ingredients: {', '.join(unused_ingredients)}")
        quality_score -= len(unused_ingredients)
    
    # Check for logical issues
    instruction_text = ' '.join(recipe.instructions).lower()
    if 'cook' in instruction_text and 'heat' not in instruction_text and 'pan' not in instruction_text:
        issues.append("Missing cooking method details")
        quality_score -= 1
    
    if 'salt' in test_case['ingredients'] and 'salt' not in instruction_text:
        issues.append("Salt not mentioned in instructions")
        quality_score -= 1
    
    # Check for recipe coherence
    if len(recipe.instructions) > 0:
        first_step = recipe.instructions[0].lower()
        if not any(word in first_step for word in ['preheat', 'heat', 'prepare', 'wash', 'cut']):
            issues.append("First step not preparatory")
            quality_score -= 1
    
    # Ensure quality score doesn't go below 0
    quality_score = max(0, quality_score)
    
    return {
        "quality_score": quality_score,
        "issues": issues,
        "used_ingredients": len(used_ingredients),
        "total_ingredients": len(test_case['ingredients']),
        "instruction_count": len(recipe.instructions)
    }

if __name__ == "__main__":
    test_recipe_quality() 