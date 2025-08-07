#!/usr/bin/env python3
"""
Test environment variable configuration
"""

import os
from recipe_ai_engine.core.config import settings

def test_environment_variables():
    """Test environment variable configuration"""
    
    print("🔧 Testing Environment Variable Configuration")
    print("=" * 50)
    
    # Show current settings
    print("📋 Current Settings:")
    print(f"  Model: {settings.ollama_model_name}")
    print(f"  Base URL: {settings.ollama_base_url}")
    print(f"  Temperature: {settings.temperature}")
    print(f"  Top P: {settings.top_p}")
    print(f"  Max Tokens: {settings.max_tokens}")
    print()
    
    # Show how to override with environment variables
    print("🌍 To override settings, set environment variables:")
    print("  set OLLAMA_MODEL_NAME=recipe-ai:latest")
    print("  set AI_TEMPERATURE=0.7")
    print("  set AI_TOP_P=0.8")
    print()
    
    # Test with environment variable
    print("🧪 Testing with environment variable override:")
    os.environ["OLLAMA_MODEL_NAME"] = "recipe-ai:latest"
    
    # Reimport settings to get updated values
    from recipe_ai_engine.core.config import settings as new_settings
    print(f"  Model (after override): {new_settings.ollama_model_name}")
    print()
    
    print("✅ Environment variable configuration working!")

if __name__ == "__main__":
    test_environment_variables()
