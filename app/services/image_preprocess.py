"""Pipeline de preprocesamiento de imágenes para mejorar el OCR."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings


def preprocess_image(image_path: Path, settings: Settings, logger: logging.Logger) -> np.ndarray:
    """Ejecuta el pipeline completo de preprocesamiento."""
    logger.info("Cargando imagen desde %s", image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen {image_path}.")

    original_shape = image.shape
    logger.debug("Dimensiones originales: %s", original_shape)

    image = resize_if_needed(image, settings.max_image_dimension, logger)
    image = deskew_image(image, logger)
    image = convert_to_grayscale(image)
    image = normalize_contrast(image, logger)
    image = reduce_noise(image, settings, logger)
    image = binarize(image, logger)

    logger.debug("Preprocesamiento completado. Dimensiones finales: %s", image.shape)
    return image


def resize_if_needed(image: np.ndarray, max_dimension: int, logger: logging.Logger) -> np.ndarray:
    """Redimensiona la imagen si supera el tamaño máximo configurado."""
    height, width = image.shape[:2]
    max_current_dimension = max(height, width)
    if max_current_dimension <= max_dimension:
        return image

    scale = max_dimension / float(max_current_dimension)
    new_width = int(width * scale)
    new_height = int(height * scale)
    logger.info(
        "Redimensionando imagen de %sx%s a %sx%s para optimizar el procesamiento.",
        width,
        height,
        new_width,
        new_height,
    )
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def deskew_image(image: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Corrige inclinaciones leves en la imagen."""
    gray = convert_to_grayscale(image)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        logger.debug("No se detectó inclinación significativa (%.2f grados).", angle)
        return image

    logger.info("Corrigiendo inclinación aproximada de %.2f grados.", angle)
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convierte la imagen a escala de grises si es necesario."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_contrast(image: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Normaliza el brillo y contraste de la imagen."""
    logger.debug("Normalizando contraste de la imagen.")
    normalized = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return normalized


def reduce_noise(image: np.ndarray, settings: Settings, logger: logging.Logger) -> np.ndarray:
    """Aplica reducción de ruido conservando bordes."""
    logger.debug("Aplicando reducción de ruido con fastNlMeansDenoising.")
    return cv2.fastNlMeansDenoising(
        image,
        None,
        h=settings.denoise_h,
        templateWindowSize=settings.denoise_template_window_size,
        searchWindowSize=settings.denoise_search_window_size,
    )


def binarize(image: np.ndarray, logger: logging.Logger) -> np.ndarray:
    """Realiza binarización adaptativa para resaltar el texto."""
    logger.debug("Aplicando binarización adaptativa.")
    adaptive = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )
    return adaptive
