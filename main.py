"""Punto de entrada principal para la API OCR."""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging_config import configure_logging
from app.services.ocr_engine import PaddleOcrEngine


def create_application() -> FastAPI:
    """Crea la instancia de FastAPI con configuración y dependencias cargadas."""
    settings: Settings = get_settings()
    logger = configure_logging(settings)

    app = FastAPI(title=settings.app_name, version=settings.api_version)
    app.include_router(router)

    @app.on_event("startup")
    async def startup_event() -> None:
        logger.info("Iniciando aplicación %s", settings.app_name)
        app.state.settings = settings
        app.state.logger = logger
        app.state.semaphore = asyncio.Semaphore(settings.max_concurrent_requests)
        app.state.ocr_engine = PaddleOcrEngine(settings=settings, logger=logger.getChild("ocr_engine"))
        logger.info("Aplicación %s inicializada correctamente", settings.app_name)

    @app.on_event("shutdown")
    async def shutdown_event() -> None:
        logger.info("Deteniendo aplicación %s", settings.app_name)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:  # type: ignore[override]
        logger = request.app.state.logger
        logger.exception("Error no controlado en %s: %s", request.url.path, exc)
        return JSONResponse(status_code=500, content={"detail": "Error interno inesperado", "path": request.url.path})

    return app


app = create_application()
