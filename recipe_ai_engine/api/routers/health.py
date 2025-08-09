from fastapi import APIRouter, Depends

from ..deps import get_generator
from ...core.models import HealthResponse
from ...core.config import settings
from ...recipes.generator import RecipeGenerator


router = APIRouter(prefix="", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(generator: RecipeGenerator = Depends(get_generator)):
    is_ok = generator.test_connection()
    status = "ok" if is_ok else "degraded"
    return HealthResponse(status=status, model=settings.ollama_model_name)


