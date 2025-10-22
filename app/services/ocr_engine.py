"""Encapsulamiento del motor OCR basado en PaddleOCR."""
from __future__ import annotations

import logging
from threading import Lock
from typing import Iterable, List

import cv2
import numpy as np
from paddleocr import PaddleOCR

from app.core.config import Settings


class OcrEngineInterface:
    """Interfaz mínima para motores OCR."""

    def recognize_text(self, image: np.ndarray) -> str:
        """Realiza OCR sobre una imagen y devuelve el texto reconocido."""
        raise NotImplementedError

    def is_ready(self) -> bool:
        """Indica si el motor está inicializado y listo para procesar."""
        raise NotImplementedError


class PaddleOcrEngine(OcrEngineInterface):
    """Implementación de :class:`OcrEngineInterface` usando PaddleOCR."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self._logger = logger
        self._settings = settings
        self._lock = Lock()
        self._ocr = PaddleOCR(
            lang=settings.ocr_language,
            use_gpu=settings.ocr_use_gpu,
            enable_mkldnn=settings.ocr_enable_mkldnn,
            use_angle_cls=settings.ocr_angle_classifier,
        )
        self._logger.info(
            "Motor PaddleOCR inicializado (lang=%s, gpu=%s, mkldnn=%s)",
            settings.ocr_language,
            settings.ocr_use_gpu,
            settings.ocr_enable_mkldnn,
        )

    def recognize_text(self, image: np.ndarray) -> str:
        """Procesa la imagen y devuelve el texto detectado."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self._lock:
            self._logger.debug("Ejecutando OCR sobre la imagen preprocesada.")
            result = self._ocr.ocr(image)

        texts: List[str] = []
        for line in result:
            if not line:
                continue
            for _, (text, _probability) in line:
                texts.append(text)
        extracted_text = "\n".join(clean_segments(texts))
        self._logger.debug("OCR completado. Caracteres extraídos: %s", len(extracted_text))
        return extracted_text

    def is_ready(self) -> bool:
        return self._ocr is not None


def clean_segments(segments: Iterable[str]) -> Iterable[str]:
    """Limpia segmentos de texto eliminando espacios innecesarios."""
    for segment in segments:
        yield segment.strip()
