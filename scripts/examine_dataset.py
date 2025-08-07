#!/usr/bin/env python3
"""
Examine the recipe dataset structure
"""

import pandas as pd
import json

def examine_dataset():
    """Examine the recipe dataset structure"""
    
    # Load the first parquet file
    df = pd.read_parquet('datasets/train-00000-of-00004-237b1b1141fdcfa1.parquet')
    
    print("📊 Dataset Information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Total recipes: {len(df)}")
    
    print("\n📋 Sample Recipes:")
    for i in range(3):
        recipe = df.iloc[i]['input']
        print(f"\n--- Recipe {i+1} ---")
        print(recipe)
        print("-" * 50)
    
    # Check the format of recipes
    print("\n🔍 Recipe Format Analysis:")
    sample_recipe = df.iloc[0]['input']
    
    # Split into sections
    lines = sample_recipe.split('\n')
    title = lines[0]
    
    # Find ingredients and directions sections
    ingredients_start = None
    directions_start = None
    
    for i, line in enumerate(lines):
        if 'Ingredients:' in line:
            ingredients_start = i
        elif 'Directions:' in line:
            directions_start = i
    
    print(f"Title: {title}")
    print(f"Ingredients section starts at line: {ingredients_start}")
    print(f"Directions section starts at line: {directions_start}")
    
    # Extract ingredients and directions
    if ingredients_start and directions_start:
        ingredients_lines = lines[ingredients_start+1:directions_start]
        directions_lines = lines[directions_start+1:]
        
        print(f"\nIngredients ({len(ingredients_lines)} items):")
        for line in ingredients_lines:
            if line.strip():
                print(f"  {line}")
        
        print(f"\nDirections ({len(directions_lines)} steps):")
        for line in directions_lines:
            if line.strip():
                print(f"  {line}")
    
    # Check dataset size across all files
    total_recipes = 0
    for i in range(4):
        try:
            file_path = f'datasets/train-{i:05d}-of-00004-*.parquet'
            import glob
            files = glob.glob(file_path)
            if files:
                df_part = pd.read_parquet(files[0])
                total_recipes += len(df_part)
                print(f"File {i+1}: {len(df_part)} recipes")
        except Exception as e:
            print(f"Error reading file {i+1}: {e}")
    
    print(f"\n📈 Total recipes across all files: {total_recipes}")
    
    return df

def convert_to_fine_tuning_format():
    """Convert the dataset to the format needed for fine-tuning"""
    
    print("\n🔄 Converting dataset to fine-tuning format...")
    
    # Load all parquet files
    all_recipes = []
    
    for i in range(4):
        try:
            import glob
            file_pattern = f'datasets/train-{i:05d}-of-00004-*.parquet'
            files = glob.glob(file_pattern)
            if files:
                df = pd.read_parquet(files[0])
                all_recipes.extend(df['input'].tolist())
                print(f"Loaded {len(df)} recipes from file {i+1}")
        except Exception as e:
            print(f"Error loading file {i+1}: {e}")
    
    print(f"Total recipes loaded: {len(all_recipes)}")
    
    # Convert to our fine-tuning format
    converted_recipes = []
    
    for i, recipe_text in enumerate(all_recipes[:100]):  # Process first 100 for testing
        try:
            # Parse the recipe text
            lines = recipe_text.split('\n')
            title = lines[0].strip()
            
            # Find sections
            ingredients = []
            instructions = []
            
            in_ingredients = False
            in_directions = False
            
            for line in lines[1:]:
                line = line.strip()
                if not line:
                    continue
                
                if 'Ingredients:' in line:
                    in_ingredients = True
                    in_directions = False
                    continue
                elif 'Directions:' in line:
                    in_ingredients = False
                    in_directions = True
                    continue
                
                if in_ingredients and line.startswith('-'):
                    # Extract ingredient and amount
                    ingredient_text = line[1:].strip()
                    if ':' in ingredient_text:
                        amount, item = ingredient_text.split(':', 1)
                        ingredients.append({
                            "item": item.strip(),
                            "amount": amount.strip()
                        })
                    else:
                        ingredients.append({
                            "item": ingredient_text,
                            "amount": "as needed"
                        })
                
                elif in_directions and line.startswith('-'):
                    instructions.append(line[1:].strip())
            
            # Create converted recipe
            if title and ingredients and instructions:
                converted_recipe = {
                    "title": title,
                    "ingredients": ingredients,
                    "instructions": instructions
                }
                converted_recipes.append(converted_recipe)
                
        except Exception as e:
            print(f"Error converting recipe {i}: {e}")
            continue
    
    print(f"Successfully converted {len(converted_recipes)} recipes")
    
    # Save converted dataset
    output_file = "converted_recipes.json"
    with open(output_file, 'w') as f:
        json.dump(converted_recipes, f, indent=2)
    
    print(f"✅ Saved converted dataset to {output_file}")
    
    # Show sample converted recipe
    if converted_recipes:
        print("\n📝 Sample converted recipe:")
        sample = converted_recipes[0]
        print(f"Title: {sample['title']}")
        print(f"Ingredients: {len(sample['ingredients'])} items")
        print(f"Instructions: {len(sample['instructions'])} steps")
    
    return converted_recipes

if __name__ == "__main__":
    # Examine the dataset
    df = examine_dataset()
    
    # Convert to fine-tuning format
    converted_recipes = convert_to_fine_tuning_format() 