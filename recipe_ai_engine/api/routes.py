from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core.models import RecipeRequest, RecipeResponse, HealthResponse
from ..core.config import settings
from ..recipes.generator import RecipeGenerator


def get_generator() -> RecipeGenerator:
    return RecipeGenerator()


def create_app() -> FastAPI:
    app = FastAPI(title="Recipe AI Engine API", version="0.1.0")

    # Basic CORS (adjust in production)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health(generator: RecipeGenerator = Depends(get_generator)):
        is_ok = generator.test_connection()
        status = "ok" if is_ok else "degraded"
        return HealthResponse(status=status, model=settings.ollama_model_name)

    @app.post("/recipes/generate", response_model=RecipeResponse)
    async def generate_recipe(
        request: RecipeRequest,
        generator: RecipeGenerator = Depends(get_generator),
    ) -> RecipeResponse:
        try:
            return generator.generate_recipe(request)
        except Exception as exc:  # refine with domain exceptions if desired
            raise HTTPException(status_code=500, detail=str(exc))

    return app


# App instance for ASGI servers like uvicorn
app = create_app()


