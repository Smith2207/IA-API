from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import get_settings
from app.db.connection import close_pool, init_db, init_pool
from app.services.search import FaissSearchIndex
from app.utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Iniciando PawPatrol IA API")

    try:
        init_pool()
        init_db()
        count = FaissSearchIndex.get_instance().rebuild_from_db()
        logger.info("Listo | mascotas indexadas: %s", count)
    except Exception:
        logger.exception("Error al inicializar base de datos o índice FAISS")
        raise

    yield

    close_pool()
    logger.info("API detenida")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="PawPatrol IA API",
        description=(
            "API de reconocimiento visual de mascotas perdidas "
            "(YOLOv8 + CLIP + FAISS + Neon PostgreSQL)"
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        logger.exception("Excepción no controlada en %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Error interno del servidor."},
        )

    app.include_router(router)
    return app


app = create_app()
