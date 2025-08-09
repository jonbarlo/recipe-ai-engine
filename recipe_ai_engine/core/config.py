"""
Configuration management for Recipe AI Engine
"""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings"""

    # pydantic v2 settings config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env keys (prevents ValidationError on extras)
    )

    # Ollama Configuration
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "llama2:7b"

    # AI Model Parameters (support AI_* env vars)
    temperature: float = Field(0.3, validation_alias="AI_TEMPERATURE")
    top_p: float = Field(0.9, validation_alias="AI_TOP_P")
    max_tokens: int = Field(800, validation_alias="AI_MAX_TOKENS")

    # Recipe Generation Settings
    default_difficulty: str = "medium"
    default_servings: int = 4
    default_cuisine: str = "General"

    # Timeout Settings
    request_timeout: int = 60
    connection_timeout: int = 5

    # Logging
    log_level: str = "INFO"

    # Optional Hugging Face token support (common env name)
    huggingface_token: Optional[str] = Field(default=None)


# Global settings instance
settings = Settings()