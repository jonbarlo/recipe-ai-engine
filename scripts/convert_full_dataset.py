#!/usr/bin/env python3
"""
Convert the full recipe dataset to fine-tuning format
"""

import pandas as pd
import json
import glob
from tqdm import tqdm

def convert_full_dataset():
    """Convert the entire dataset to fine-tuning format"""
    
    print("🔄 Converting full dataset to fine-tuning format...")
    
    # Find all parquet files
    parquet_files = glob.glob('datasets/train-*-of-00004-*.parquet')
    print(f"Found {len(parquet_files)} parquet files")
    
    all_converted_recipes = []
    
    for file_path in parquet_files:
        print(f"\n📁 Processing {file_path}...")
        
        # Load parquet file
        df = pd.read_parquet(file_path)
        print(f"Loaded {len(df)} recipes from {file_path}")
        
        # Convert recipes
        converted_recipes = []
        
        for i, recipe_text in enumerate(tqdm(df['input'], desc="Converting recipes")):
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
        
        print(f"Successfully converted {len(converted_recipes)} recipes from {file_path}")
        all_converted_recipes.extend(converted_recipes)
    
    print(f"\n📊 Total converted recipes: {len(all_converted_recipes)}")
    
    # Save full converted dataset
    output_file = "full_recipe_dataset.json"
    print(f"💾 Saving to {output_file}...")
    
    with open(output_file, 'w') as f:
        json.dump(all_converted_recipes, f, indent=2)
    
    print(f"✅ Saved full dataset to {output_file}")
    
    # Create a smaller subset for testing
    test_subset = all_converted_recipes[:1000]  # First 1000 recipes
    test_file = "recipe_dataset_test.json"
    
    with open(test_file, 'w') as f:
        json.dump(test_subset, f, indent=2)
    
    print(f"✅ Saved test subset ({len(test_subset)} recipes) to {test_file}")
    
    # Show statistics
    print("\n📈 Dataset Statistics:")
    print(f"Total recipes: {len(all_converted_recipes)}")
    print(f"Test subset: {len(test_subset)}")
    
    # Show sample recipe
    if all_converted_recipes:
        sample = all_converted_recipes[0]
        print(f"\n📝 Sample recipe:")
        print(f"Title: {sample['title']}")
        print(f"Ingredients: {len(sample['ingredients'])} items")
        print(f"Instructions: {len(sample['instructions'])} steps")
    
    return all_converted_recipes

def create_training_subset(max_recipes=10000):
    """Create a smaller subset for initial fine-tuning"""
    
    print(f"\n🎯 Creating training subset with {max_recipes} recipes...")
    
    # Load the full dataset
    with open('full_recipe_dataset.json', 'r') as f:
        all_recipes = json.load(f)
    
    # Create subset
    subset = all_recipes[:max_recipes]
    
    # Save subset
    subset_file = f"recipe_dataset_{max_recipes}.json"
    with open(subset_file, 'w') as f:
        json.dump(subset, f, indent=2)
    
    print(f"✅ Saved training subset to {subset_file}")
    
    return subset

if __name__ == "__main__":
    # Convert full dataset
    converted_recipes = convert_full_dataset()
    
    # Create training subsets
    create_training_subset(1000)   # 1K recipes for quick testing
    create_training_subset(10000)  # 10K recipes for initial training
    create_training_subset(50000)  # 50K recipes for good training 