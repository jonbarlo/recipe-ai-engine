from fastapi import APIRouter, Depends, HTTPException

from ..deps import get_generator
from ...core.models import RecipeRequest, RecipeResponse
from ...recipes.generator import RecipeGenerator


router = APIRouter(prefix="/recipes", tags=["recipes"])


@router.post("/generate", response_model=RecipeResponse)
async def generate_recipe(
    request: RecipeRequest,
    generator: RecipeGenerator = Depends(get_generator),
) -> RecipeResponse:
    try:
        return generator.generate_recipe(request)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


