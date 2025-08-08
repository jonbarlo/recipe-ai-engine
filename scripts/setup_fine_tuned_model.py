#!/usr/bin/env python3
"""
Setup script to create Ollama model from fine-tuned recipe model
"""

import os
import subprocess
import sys

def create_ollama_model():
    """Create Ollama model from fine-tuned recipe model if available.

    If FINE_TUNED_MODEL_PATH (or ./recipe-model) exists, use it as the base in the Modelfile.
    Otherwise, fall back to the base llama2:7b model (no fine-tuned weights).
    """
    
    print("🔧 Setting up fine-tuned recipe model in Ollama...")
    
    # Check if Ollama is running
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            print("❌ Ollama is not running. Please start Ollama first.")
            return False
    except:
        print("❌ Ollama is not running. Please start Ollama first.")
        return False
    
    # Determine fine-tuned model path
    fine_tuned_path = os.environ.get("FINE_TUNED_MODEL_PATH", "./recipe-model")
    use_fine_tuned = os.path.isdir(fine_tuned_path)

    if use_fine_tuned:
        base_line = f"FROM {fine_tuned_path}"
        print(f"🧠 Using fine-tuned model from: {fine_tuned_path}")
    else:
        base_line = "FROM llama2:7b"
        print("⚠️ Fine-tuned model directory not found; using base model llama2:7b.")
        print("   Set FINE_TUNED_MODEL_PATH to your trained model directory to use it.")

    # Create Modelfile for recipe-ai
    modelfile_content = f"""{base_line}
TEMPLATE "Recipe: Create a recipe using {{.Ingredients}}

Ingredients:
{{.Ingredients}}

Instructions:
"
PARAMETER temperature 0.7
PARAMETER top_p 0.9
SYSTEM You are a professional chef and recipe creator. Generate detailed, accurate recipes based on the provided ingredients. Always include proper cooking instructions, timing, and helpful tips.
"""
    
    # Write Modelfile
    with open("Modelfile", "w") as f:
        f.write(modelfile_content)
    
    print("📝 Created Modelfile for recipe-ai model")
    
    # Create the model in Ollama
    try:
        result = subprocess.run(
            ["ollama", "create", "recipe-ai", "-f", "Modelfile"],
            capture_output=True,
            text=True,
            check=True
        )
        print("✅ Successfully created recipe-ai model in Ollama")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create model: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("❌ Ollama CLI not found. Please install Ollama first.")
        return False

def test_fine_tuned_model():
    """Test the fine-tuned model with sample ingredients"""
    
    print("\n🧪 Testing fine-tuned model...")
    
    test_prompt = """Recipe: Create a recipe using chicken, rice, vegetables

Ingredients:
- chicken breast
- white rice
- mixed vegetables

Instructions:
"""
    
    try:
        import requests
        
        payload = {
            "model": "recipe-ai:latest",
            "prompt": test_prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "max_tokens": 512
            }
        }
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Model test successful!")
            print("Sample response:")
            print(result.get("response", "")[:200] + "...")
            return True
        else:
            print(f"❌ Model test failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def main():
    """Main setup function"""
    
    print("🚀 Setting up fine-tuned recipe model...")
    
    # Create the model
    if create_ollama_model():
        # Test the model
        test_fine_tuned_model()
        
        print("\n✅ Fine-tuned model setup complete!")
        print("📝 You can now use the recipe-ai model in your application.")
        print("🔧 The model will be used automatically by your Recipe AI Engine.")
        
        # Clean up
        if os.path.exists("Modelfile"):
            os.remove("Modelfile")
            print("🧹 Cleaned up temporary files")
    else:
        print("❌ Failed to set up fine-tuned model")
        sys.exit(1)

if __name__ == "__main__":
    main()
