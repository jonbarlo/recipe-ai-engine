import json
from fastapi.testclient import TestClient
from recipe_ai_engine.api.routes import create_app, get_generator
from recipe_ai_engine.core.models import RecipeRequest, RecipeResponse, HealthResponse


class FakeGenerator:
    def test_connection(self) -> bool:
        return True

    def generate_recipe(self, request: RecipeRequest) -> RecipeResponse:
        return RecipeResponse(
            title="Test Recipe",
            ingredients=[
                {"item": item, "amount": "1 unit"} for item in request.ingredients
            ],
            instructions=[
                "Prep ingredients",
                "Cook ingredients",
                "Serve and enjoy",
            ],
            cooking_time=20,
            difficulty="easy",
            servings=request.servings or 2,
            cuisine=request.cuisine or "International",
        )


def get_fake_generator():
    return FakeGenerator()


def test_health_endpoint():
    app = create_app()
    # Override dependency to avoid calling real Ollama
    app.dependency_overrides[get_generator] = get_fake_generator
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    data = HealthResponse(**resp.json())
    assert data.status in {"ok", "degraded"}
    assert isinstance(data.model, str)


def test_generate_recipe_endpoint():
    app = create_app()
    app.dependency_overrides[get_generator] = get_fake_generator
    client = TestClient(app)

    payload = {
        "ingredients": ["chicken", "rice", "vegetables"],
        "servings": 4,
        "cuisine": "Asian",
        "difficulty": "medium",
    }

    resp = client.post("/recipes/generate", json=payload)
    assert resp.status_code == 200

    recipe = RecipeResponse(**resp.json())
    assert recipe.title == "Test Recipe"
    assert len(recipe.ingredients) == len(payload["ingredients"])
    assert len(recipe.instructions) >= 2
    assert recipe.servings == 4
    assert recipe.cuisine == "Asian"
