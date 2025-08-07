"""
Configuration management for Recipe AI Engine
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "llama2:7b"
    
    # AI Model Parameters
    temperature: float = 0.3
    top_p: float = 0.9
    max_tokens: int = 800
    
    # Recipe Generation Settings
    default_difficulty: str = "medium"
    default_servings: int = 4
    default_cuisine: str = "General"
    
    # Timeout Settings
    request_timeout: int = 60
    connection_timeout: int = 5
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        # env_file = ".env"  # Temporarily disabled
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings() 