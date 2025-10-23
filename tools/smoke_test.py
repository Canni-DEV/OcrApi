import logging
import sys
from pathlib import Path


def main() -> int:
    # Ensure project root is on sys.path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        from app.services.ocr_engine import PaddleOcrEngine
        from app.core.config import get_settings
        import numpy as np
    except Exception as e:
        print("IMPORT-ERROR:", type(e).__name__, e)
        return 2

    settings = get_settings()
    logger = logging.getLogger("SMOKE")
    engine = PaddleOcrEngine(settings=settings, logger=logger)

    img = (np.ones((10, 10, 3), dtype="uint8") * 255)
    try:
        out = engine.recognize_text(img)
        print("OK: OCR ran; length:", len(out))
        return 0
    except Exception as e:
        print("RUN-ERROR:", type(e).__name__, e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
