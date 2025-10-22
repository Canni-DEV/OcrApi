"""Configuración del sistema de logging para la API OCR."""
from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from .config import Settings


def configure_logging(settings: Settings) -> logging.Logger:
    """Configura el logger principal de la aplicación."""
    log_path: Path = settings.log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(settings.app_name.replace(" ", "_"))
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = TimedRotatingFileHandler(
        filename=str(log_path),
        when="midnight",
        interval=1,
        backupCount=settings.log_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # Evita agregar handlers duplicados en reinicios/calientes
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logging.getLogger("uvicorn").handlers.clear()
    logging.getLogger("uvicorn.access").handlers.clear()

    return logger
