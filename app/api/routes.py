"""Definición de rutas REST para la API OCR."""
from __future__ import annotations

import importlib.util
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.services.file_utils import validate_image_path
from app.services.image_preprocess import preprocess_image

router = APIRouter()


class OCRRequest(BaseModel):
    """Payload para solicitar OCR sobre una imagen existente."""

    image_path: str = Field(..., description="Ruta absoluta o relativa de la imagen en el servidor.")


@router.get("/health")
async def health_check(request: Request) -> dict:
    """Endpoint de verificación de salud del servicio."""
    settings: Settings = request.app.state.settings
    logger = request.app.state.logger.getChild("health")

    response = {
        "status": "healthy" if request.app.state.ocr_engine.is_ready() else "initializing",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": settings.api_version,
    }

    if settings.include_resource_metrics:
        metrics = gather_resource_metrics()
        if metrics:
            response["resources"] = metrics

    logger.debug("Respuesta de health check generada: %s", response)
    return response


@router.post("/ocr")
async def perform_ocr(payload: OCRRequest, request: Request) -> dict:
    """Procesa una imagen utilizando el motor OCR configurado."""
    settings: Settings = request.app.state.settings
    logger = request.app.state.logger.getChild("ocr")

    try:
        image_path = validate_image_path(Path(payload.image_path), settings)
    except ValueError as error:
        logger.warning("Validación de imagen fallida: %s", error)
        raise HTTPException(status_code=400, detail=str(error)) from error

    logger.info("Solicitud OCR recibida para %s", image_path)

    start_time = time.perf_counter()
    async with request.app.state.semaphore:
        preprocess_logger = logger.getChild("preprocess")
        try:
            processed_image = await run_in_threadpool(preprocess_image, image_path, settings, preprocess_logger)
            text = await run_in_threadpool(request.app.state.ocr_engine.recognize_text, processed_image)
        except HTTPException:
            raise
        except Exception as error:  # noqa: BLE001 - se registra y transforma en HTTP 500 controlado
            logger.exception("Error procesando OCR para %s", image_path)
            raise HTTPException(status_code=500, detail="Error interno procesando la imagen.") from error
        finally:
            # Libera memoria referencial explícitamente
            if "processed_image" in locals():
                del processed_image

    elapsed = time.perf_counter() - start_time
    logger.info("OCR completado para %s en %.2f segundos", image_path, elapsed)
    return {"text": text, "elapsed_seconds": round(elapsed, 3)}


def gather_resource_metrics() -> dict:
    """Obtiene métricas de CPU/RAM si psutil está disponible."""
    if importlib.util.find_spec("psutil") is None:
        return {}

    import psutil  # type: ignore import-outside-toplevel

    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "memory_percent": psutil.virtual_memory().percent,
    }
