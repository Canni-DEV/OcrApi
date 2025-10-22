"""Configuración central de la aplicación OCR."""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import BaseSettings, Field, validator


class Settings(BaseSettings):
    """Valores de configuración obtenidos desde variables de entorno o valores por defecto."""

    app_name: str = "OCR API"
    api_version: str = "1.0.0"
    max_concurrent_requests: int = Field(3, ge=1, description="Número máximo de OCR simultáneos.")
    max_image_size_mb: int = Field(10, ge=1, description="Tamaño máximo permitido por imagen en MB.")
    max_image_dimension: int = Field(
        2500,
        ge=500,
        description="Dimensión máxima (ancho o alto) antes de reescalar la imagen.",
    )
    allowed_image_dirs: List[str] = Field(
        default_factory=list,
        description="Directorios permitidos para la lectura de imágenes.",
    )
    log_level: str = Field("INFO", description="Nivel de logging (DEBUG, INFO, WARNING, ERROR).")
    log_dir: str = Field("logs", description="Directorio base para almacenar los archivos de log.")
    log_filename: str = Field("ocr_api.log", description="Nombre del archivo principal de log.")
    log_backup_count: int = Field(7, ge=1, description="Cantidad de archivos de log históricos a mantener.")
    include_resource_metrics: bool = Field(
        True,
        description="Indica si el endpoint de salud debe incluir métricas de CPU/RAM.",
    )
    denoise_h: float = Field(10.0, ge=0.0, description="Parámetro h para fastNlMeansDenoising.")
    denoise_template_window_size: int = Field(7, ge=1, description="Tamaño de ventana de plantilla para denoising.")
    denoise_search_window_size: int = Field(21, ge=1, description="Tamaño de ventana de búsqueda para denoising.")
    ocr_language: str = Field("es", description="Código de idioma para el motor OCR.")
    ocr_use_gpu: bool = Field(False, description="Indica si se debe utilizar GPU para PaddleOCR.")
    ocr_enable_mkldnn: bool = Field(True, description="Habilita MKLDNN para acelerar inferencia en CPU.")
    ocr_angle_classifier: bool = Field(True, description="Activa el clasificador de ángulo en PaddleOCR.")

    class Config:
        env_file = ".env"
        case_sensitive = False

    @validator("allowed_image_dirs", pre=True)
    def split_allowed_dirs(cls, value):  # type: ignore[override]
        """Permite especificar múltiples directorios mediante una cadena separada por punto y coma."""
        if isinstance(value, str):
            return [item for item in value.split(";") if item.strip()]
        return value

    @property
    def max_image_size_bytes(self) -> int:
        """Devuelve el tamaño máximo permitido de la imagen en bytes."""
        return self.max_image_size_mb * 1024 * 1024

    @property
    def log_path(self) -> Path:
        """Obtiene la ruta completa del archivo de log principal."""
        return Path(self.log_dir).expanduser() / self.log_filename

    @property
    def allowed_image_paths(self) -> List[Path]:
        """Lista de rutas base permitidas para las imágenes."""
        return [Path(path).expanduser().resolve() for path in self.allowed_image_dirs]


@lru_cache()
def get_settings() -> Settings:
    """Obtiene una instancia cacheada de la configuración."""
    return Settings()
