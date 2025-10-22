"""Utilidades para validación de rutas y archivos de imagen."""
from __future__ import annotations

from pathlib import Path

from app.core.config import Settings


def validate_image_path(image_path: Path, settings: Settings) -> Path:
    """Valida que la ruta exista, sea un archivo y respete las restricciones configuradas."""
    resolved_path = image_path.expanduser().resolve()
    if not resolved_path.exists():
        raise ValueError(f"La imagen {resolved_path} no existe.")
    if not resolved_path.is_file():
        raise ValueError(f"La ruta {resolved_path} no es un archivo válido.")

    file_size = resolved_path.stat().st_size
    if file_size > settings.max_image_size_bytes:
        raise ValueError(
            "La imagen excede el tamaño máximo permitido de "
            f"{settings.max_image_size_mb} MB."
        )

    allowed_roots = settings.allowed_image_paths
    if allowed_roots and not any(is_subpath(resolved_path, base) for base in allowed_roots):
        raise ValueError("La imagen solicitada se encuentra fuera de los directorios permitidos.")

    return resolved_path


def is_subpath(path: Path, base: Path) -> bool:
    """Indica si una ruta se encuentra dentro de otra ruta base."""
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True
