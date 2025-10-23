import sys
from pathlib import Path
import logging

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import numpy as np
import cv2
from app.services.ocr_engine import PaddleOcrEngine
from app.core.config import get_settings


def main():
    logging.basicConfig(level=logging.INFO)
    s = get_settings()
    eng = PaddleOcrEngine(settings=s, logger=logging.getLogger("INTROSPECT"))
    img = np.ones((200, 800, 3), dtype="uint8") * 255
    cv2.putText(img, "FACTURA 123 ABC", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

    try:
        eng._ensure_initialized()
        res = eng._ocr.ocr(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    except Exception as e:
        print("ERR:", type(e).__name__, e)
        return 1

    print("TYPE:", type(res).__name__)
    print("LEN:", len(res))
    if len(res):
        elem0 = res[0]
        print("ELEM0-TYPE:", type(elem0).__name__)
        print("ELEM0-STR:", str(elem0)[:600])
        if isinstance(elem0, list) and elem0:
            print("ELEM0-0-TYPE:", type(elem0[0]).__name__)
            print("ELEM0-0-STR:", str(elem0[0])[:600])
        if isinstance(elem0, dict):
            print("ELEM0-KEYS:", list(elem0.keys()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

