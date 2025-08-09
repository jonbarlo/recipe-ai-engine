from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import health, recipes


def create_app() -> FastAPI:
    app = FastAPI(title="Recipe AI Engine API", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(recipes.router)

    return app


# App instance for ASGI servers like uvicorn
app = create_app()


