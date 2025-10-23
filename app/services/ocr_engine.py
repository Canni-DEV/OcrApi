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
        self._ocr: PaddleOCR | None = None
        self._device = "gpu" if settings.ocr_use_gpu else "cpu"

    def _ensure_initialized(self) -> None:
        if self._ocr is not None:
            return
        # Inicialización perezosa para evitar fallos de arranque cuando falta paddle
        try:
            # Newer PaddleOCR (>=2.7) signature
            self._ocr = PaddleOCR(
                lang=self._settings.ocr_language,
                device=self._device,
                enable_mkldnn=self._settings.ocr_enable_mkldnn,
                use_textline_orientation=self._settings.ocr_angle_classifier,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
            )
            self._logger.info(
                "Motor PaddleOCR inicializado (lang=%s, device=%s, mkldnn=%s)",
                self._settings.ocr_language,
                self._device,
                self._settings.ocr_enable_mkldnn,
            )
        except TypeError:
            # Backward compatibility with PaddleOCR 2.6.x (no 'device' or 'use_textline_orientation')
            self._ocr = PaddleOCR(
                lang=self._settings.ocr_language,
                use_angle_cls=self._settings.ocr_angle_classifier,
                enable_mkldnn=self._settings.ocr_enable_mkldnn,
                use_gpu=self._settings.ocr_use_gpu,
            )
            self._logger.info(
                "Motor PaddleOCR inicializado (lang=%s, device=%s, mkldnn=%s) [compat 2.6]",
                self._settings.ocr_language,
                self._device,
                self._settings.ocr_enable_mkldnn,
            )
        except ModuleNotFoundError as e:
            # Mensaje claro cuando falta 'paddle'
            if e.name == "paddle":
                self._logger.error(
                    "No se encontró el módulo 'paddle'. Instala paddlepaddle adecuado para tu plataforma/Python. "
                    "En CPU suele ser 'paddlepaddle'. Verifica además compatibilidad de versión de Python."
                )
            raise

    def recognize_text(self, image: np.ndarray) -> str:
        """Procesa la imagen y devuelve el texto detectado."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self._lock:
            # Inicializa en el primer uso
            self._ensure_initialized()
            self._logger.debug("Ejecutando OCR sobre la imagen preprocesada.")
            assert self._ocr is not None
            result = self._ocr.ocr(image)

        texts: List[str] = []
        for line in result:
            if not line:
                continue
            # PaddleOCR 3.x (OCRResult dict-like)
            if hasattr(line, "get"):
                try:
                    rec_texts = line.get("rec_texts")
                except Exception:
                    rec_texts = None
                if rec_texts:
                    for t in rec_texts:
                        if t is not None:
                            texts.append(str(t))
                    continue
            # PaddleOCR 2.x ([(poly, (text, score)), ...])
            if isinstance(line, (list, tuple)):
                for det in line:
                    if not det:
                        continue
                    try:
                        cand = det[1]
                        if isinstance(cand, (list, tuple)) and len(cand) >= 1:
                            texts.append(str(cand[0]))
                            continue
                    except Exception:
                        pass
                    if isinstance(det, dict) and "text" in det:
                        texts.append(str(det["text"]))
        extracted_text = "\n".join(clean_segments(texts))
        cleaned = postprocess_text(extracted_text)
        self._logger.debug("OCR completado. Caracteres extraidos: {0} -> {1} tras limpieza", len(extracted_text), len(cleaned))
        return cleaned
        self._logger.debug("OCR completado. Caracteres extraídos: %s", len(extracted_text))
        return extracted_text

    def is_ready(self) -> bool:
        return self._ocr is not None


def clean_segments(segments: Iterable[str]) -> Iterable[str]:
    """Limpia segmentos de texto eliminando espacios innecesarios."""
    for segment in segments:
        yield segment.strip()


def postprocess_text(text: str) -> str:
    """Postprocess OCR text: normalize, remove artifacts, fix common mojibake.

    - Normalize Unicode (NFC)
    - Remove control chars and stray box glyphs
    - Fix frequent mojibake sequences for Spanish accents and degree symbol
    - Collapse excessive whitespace; trim lines
    """
    import re
    import unicodedata

    s = unicodedata.normalize("NFC", text)

    # Common mojibake fixes (UTF-8 mis-decoded as Latin-1)
    replacements = {
        "\u00C3\u00A1": "\u00E1",  # Ã¡ -> á
        "\u00C3\u00A9": "\u00E9",  # Ã© -> é
        "\u00C3\u00AD": "\u00ED",  # Ã­ -> í
        "\u00C3\u00B3": "\u00F3",  # Ã³ -> ó
        "\u00C3\u00BA": "\u00FA",  # Ãº -> ú
        "\u00C3\u00B1": "\u00F1",  # Ã± -> ñ
        "\u00C3\u0081": "\u00C1",  # Ã -> Á (rare)
        "\u00C3\u0089": "\u00C9",  # -> É
        "\u00C3\u0093": "\u00D3",  # -> Ó
        "\u00C3\u009A": "\u00DA",  # -> Ú
        "\u00C3\u0091": "\u00D1",  # -> Ñ
        "\u00C2\u00B0": "\u00B0",  # Â° -> °
        "\u00C2\u00BA": "\u00BA",  # Âº -> º
        "\u00C2\u00AA": "\u00AA",  # Âª -> ª
    }
    for k, v in replacements.items():
        s = s.replace(k, v)

    # Remove box glyph and control chars (preserve newlines and tabs)
    s = s.replace("\u25A1", "")
    s = re.sub(r"[\u0000-\u0009\u000B\u000C\u000E-\u001F\u007F]", " ", s)

    # Normalize whitespace per line
    lines = []
    for line in s.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)
