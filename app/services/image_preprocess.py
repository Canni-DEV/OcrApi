"""Image preprocessing pipeline to improve OCR (ASCII-safe)."""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from app.core.config import Settings


def preprocess_image(image_path: Path, settings: Settings, logger: logging.Logger) -> np.ndarray:
    """Run the full preprocessing pipeline.

    If settings.preprocess_autoconfig is True, apply heuristics per image to
    improve contrast, sharpness and decide a top crop percent automatically.
    Otherwise, use manual flags in settings and the configured crop percent.
    """
    logger.info("Cargando imagen desde %s", image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"No se pudo cargar la imagen {image_path}.")

    logger.debug("Dimensiones originales: %s", image.shape)

    image = resize_if_needed(image, settings.max_image_dimension, logger)
    image = deskew_image(image, settings, logger)
    gray = convert_to_grayscale(image)

    if settings.preprocess_autoconfig:
        # Metrics
        blur_score = variance_of_laplacian(gray)
        contrast = float(gray.std())
        logger.debug("Metrics: blur=%.1f, contrast=%.1f", blur_score, contrast)

        # Contrast enhancement for low-contrast photos
        if contrast < 60:
            gray = enhance_contrast_clahe(gray, logger)
            contrast = float(gray.std())

        # Light sharpening if blurry
        if blur_score < 120:
            gray = unsharp_mask(gray)

        # Denoise and binarize if metrics indicate
        if blur_score < 90:
            gray = reduce_noise(gray, settings, logger)
        if contrast < 50:
            gray = binarize(gray, logger)

        # Auto crop top percent based on text density
        auto_percent = decide_crop_percent(gray, logger)
        min_pct = max(50, int(settings.ocr_crop_height_percent))
        crop_pct = max(min_pct, min(100, auto_percent))
        if crop_pct < 100:
            logger.info("Aplicando recorte superior auto %d%% (min=%d%%)", crop_pct, min_pct)
            gray = crop_top_percent(gray, crop_pct)
    else:
        # Manual configuration path
        gray = normalize_contrast(gray, logger)
        if settings.preprocess_enable_denoise:
            gray = reduce_noise(gray, settings, logger)
        if settings.preprocess_enable_binarize:
            gray = binarize(gray, logger)
        pct = int(settings.ocr_crop_height_percent)
        pct = max(50, min(100, pct))
        if pct < 100:
            logger.info("Aplicando recorte superior %d%% (manual)", pct)
            gray = crop_top_percent(gray, pct)

    logger.debug("Preprocesamiento completado. Dimensiones finales: %s", gray.shape)
    return gray


def resize_if_needed(image: np.ndarray, max_dimension: int, logger: logging.Logger) -> np.ndarray:
    """Resize image if larger than max_dimension."""
    h, w = image.shape[:2]
    max_current_dimension = max(h, w)
    if max_current_dimension <= max_dimension:
        return image

    scale = max_dimension / float(max_current_dimension)
    new_w = int(w * scale)
    new_h = int(h * scale)
    logger.info("Redimensionando imagen de %sx%s a %sx%s para optimizar el procesamiento.", w, h, new_w, new_h)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def deskew_image(image: np.ndarray, settings: Settings, logger: logging.Logger) -> np.ndarray:
    """Correct small skew angles. Avoid large rotations (which often hurt OCR)."""
    gray = convert_to_grayscale(image)
    gray_inv = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray_inv, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    if abs(angle) < 0.5:
        logger.debug("No se detecto inclinacion significativa (%.2f grados).", angle)
        return image

    if abs(angle) > settings.preprocess_max_deskew_degrees:
        logger.info(
            "Inclinacion estimada %.2f supera umbral de correccion (%.1f grados). Se mantiene orientacion.",
            angle,
            settings.preprocess_max_deskew_degrees,
        )
        return image

    logger.info("Corrigiendo inclinacion aproximada de %.2f grados.", angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_contrast(image: np.ndarray, logger: logging.Logger) -> np.ndarray:
    logger.debug("Normalizando contraste de la imagen.")
    return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


def enhance_contrast_clahe(gray: np.ndarray, logger: logging.Logger) -> np.ndarray:
    logger.debug("Aplicando CLAHE para mejorar contraste.")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def unsharp_mask(gray: np.ndarray, amount: float = 1.0, radius: int = 3) -> np.ndarray:
    k = radius if radius % 2 == 1 else radius + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    sharpened = cv2.addWeighted(gray, 1 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def reduce_noise(image: np.ndarray, settings: Settings, logger: logging.Logger) -> np.ndarray:
    logger.debug("Aplicando reduccion de ruido con fastNlMeansDenoising.")
    return cv2.fastNlMeansDenoising(
        image,
        None,
        h=settings.denoise_h,
        templateWindowSize=settings.denoise_template_window_size,
        searchWindowSize=settings.denoise_search_window_size,
    )


def binarize(image: np.ndarray, logger: logging.Logger) -> np.ndarray:
    logger.debug("Aplicando binarizacion adaptativa.")
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        15,
    )


def variance_of_laplacian(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def decide_crop_percent(gray: np.ndarray, logger: logging.Logger) -> int:
    """Estimate how much of the top contains ~95% of text density (50-100%)."""
    h, w = gray.shape[:2]
    try:
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    except Exception:
        return 100
    inv = cv2.bitwise_not(th)
    row_density = (inv > 0).sum(axis=1).astype(np.float32)
    total = float(row_density.sum())
    if total <= 0.0:
        return 100
    cumsum = np.cumsum(row_density)
    y = int(np.searchsorted(cumsum, 0.95 * total))
    pct = int(np.ceil((y + 1) * 100.0 / h))
    pct = max(50, min(100, pct))
    logger.debug("Auto-crop decidido: %d%% (fila=%d de %d)", pct, y, h)
    return pct


def crop_top_percent(gray: np.ndarray, percent: int) -> np.ndarray:
    """Crop top percent of the image height."""
    h, w = gray.shape[:2]
    keep = max(1, min(h, int(h * percent / 100.0)))
    return gray[:keep, :]
