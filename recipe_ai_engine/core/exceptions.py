"""
Custom exceptions for Recipe AI Engine
"""


class RecipeGenerationError(Exception):
    """Base exception for recipe generation errors"""
    pass


class OllamaConnectionError(RecipeGenerationError):
    """Raised when unable to connect to Ollama service"""
    pass


class RecipeValidationError(RecipeGenerationError):
    """Raised when recipe validation fails"""
    pass


class PromptGenerationError(RecipeGenerationError):
    """Raised when prompt generation fails"""
    pass


class JSONParsingError(RecipeGenerationError):
    """Raised when JSON parsing from AI response fails"""
    pass


class ModelResponseError(RecipeGenerationError):
    """Raised when AI model response is invalid"""
    pass 