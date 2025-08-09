"""
Pydantic models for Recipe AI Engine
"""

from typing import List, Dict, Optional
from pydantic import BaseModel, Field
from typing import Literal


class RecipeRequest(BaseModel):
    """Request model for recipe generation"""
    
    ingredients: List[str] = Field(..., description="List of available ingredients")
    cuisine_type: Optional[str] = Field(None, description="Preferred cuisine type")
    dietary_restrictions: Optional[List[str]] = Field(None, description="Dietary restrictions")
    difficulty_level: Optional[str] = Field("medium", description="Recipe difficulty level")
    serving_size: Optional[int] = Field(4, description="Number of servings")
    
    class Config:
        json_schema_extra = {
            "example": {
                "ingredients": ["chicken breast", "rice", "vegetables"],
                "cuisine_type": "Asian",
                "dietary_restrictions": ["vegetarian"],
                "difficulty_level": "easy",
                "serving_size": 2
            }
        }


class RecipeResponse(BaseModel):
    """Response model for generated recipe"""
    
    title: str = Field(..., description="Recipe title")
    ingredients: List[Dict[str, str]] = Field(..., description="List of ingredients with amounts")
    instructions: List[str] = Field(..., description="Step-by-step cooking instructions")
    cooking_time: str = Field(..., description="Total cooking time")
    preparation_time: str = Field(..., description="Preparation time")
    difficulty: str = Field(..., description="Recipe difficulty level")
    servings: int = Field(..., description="Number of servings")
    cuisine_type: Optional[str] = Field(None, description="Cuisine type")
    dietary_notes: Optional[str] = Field(None, description="Dietary considerations")
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Chicken Stir Fry",
                "ingredients": [
                    {"item": "chicken breast", "amount": "1 lb"},
                    {"item": "rice", "amount": "1 cup"}
                ],
                "instructions": [
                    "Step 1: Cook the chicken",
                    "Step 2: Add rice and vegetables"
                ],
                "cooking_time": "20 minutes",
                "preparation_time": "10 minutes",
                "difficulty": "easy",
                "servings": 2,
                "cuisine_type": "Asian",
                "dietary_notes": "Contains meat"
            }
        } 


class HealthResponse(BaseModel):
    """Health check response model"""
    status: Literal["ok", "degraded"] = Field(..., description="Service status")
    model: str = Field(..., description="Configured Ollama model name")