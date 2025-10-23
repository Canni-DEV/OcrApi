import sys
from pathlib import Path
from fastapi.testclient import TestClient


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    from main import create_application  # type: ignore

    app = create_application()
    # Use context manager to ensure startup/shutdown events run
    with TestClient(app) as client:
        images = [
            ("remito", r"C:\\Temp\\remito.png"),
            ("factura", r"C:\\Temp\\factura.png"),
        ]

        # Health check
        r_health = client.get("/health")
        print("HEALTH:", r_health.status_code, r_health.json())

        import json
        out_dir = Path("artifacts")
        out_dir.mkdir(exist_ok=True)

        for name, image_path in images:
            print(f"--- OCR {name} ---")
            r_ocr = client.post("/ocr", json={"image_path": image_path})
            print("OCR:", r_ocr.status_code)
            try:
                data = r_ocr.json()
                text = data.get("text", "") if isinstance(data, dict) else ""
                elapsed = data.get("elapsed_seconds") if isinstance(data, dict) else None
                (out_dir / f"e2e_{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                print("OCR-TEXT-LEN:", len(text))
                if elapsed is not None:
                    print("OCR-ELAPSED:", elapsed)
            except Exception:
                safe = r_ocr.text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
                print("OCR-RAW(len):", len(safe))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
