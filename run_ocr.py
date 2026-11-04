from __future__ import annotations

import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work"
MODEL_ZIP = WORK_DIR / "model.zip"
MODEL_DIR = WORK_DIR / "model"
IMAGE_FILE = WORK_DIR / "input-image"
MAX_IMAGE_SIZE = 20 * 1024 * 1024


def env_required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def download_model(drive_url: str) -> None:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    print("Downloading model ZIP from Google Drive...")
    subprocess.run(
        ["gdown", "--fuzzy", drive_url, "-O", str(MODEL_ZIP)],
        check=True,
    )

    if not MODEL_ZIP.is_file() or not zipfile.is_zipfile(MODEL_ZIP):
        raise RuntimeError(
            "Could not download a valid ZIP. Set Google Drive sharing to "
            "'Anyone with the link – Viewer' and provide the share URL."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print("Checking ZIP and extracting model...")
    with zipfile.ZipFile(MODEL_ZIP) as archive:
        bad_file = archive.testzip()
        if bad_file:
            raise RuntimeError(f"Corrupt file inside ZIP: {bad_file}")
        archive.extractall(MODEL_DIR)


def locate_recognition_model() -> Path:
    """Find an exported PaddleOCR recognition model inside the extracted ZIP."""
    matches = []
    for config_file in MODEL_DIR.rglob("inference.yml"):
        folder = config_file.parent
        has_params = (folder / "inference.pdiparams").is_file()
        has_model = (
            (folder / "inference.json").is_file()
            or (folder / "inference.pdmodel").is_file()
        )
        if has_params and has_model:
            matches.append(folder)

    if not matches:
        file_names = [str(p.relative_to(MODEL_DIR)) for p in MODEL_DIR.rglob("*") if p.is_file()]
        preview = "\n".join(file_names[:80]) or "(ZIP appears empty)"
        raise RuntimeError(
            "No exported PaddleOCR inference model was found. The ZIP should "
            "contain a directory with inference.yml, inference.pdiparams and "
            "inference.json (or inference.pdmodel). A training checkpoint such "
            "as best_accuracy.pdparams alone is not sufficient; export it first.\n"
            "Files found:\n" + preview
        )

    if len(matches) > 1:
        print("More than one exported model found; using:", matches[0])
    print("Using recognition model:", matches[0])
    return matches[0]


def download_image(url: str) -> Path:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise RuntimeError("Image URL must start with https:// or http://")

    print("Downloading test image...")
    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()

    total = 0
    with IMAGE_FILE.open("wb") as out:
        for chunk in response.iter_content(chunk_size=256 * 1024):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_IMAGE_SIZE:
                raise RuntimeError("Image is larger than the 20 MB test limit.")
            out.write(chunk)

    try:
        with Image.open(IMAGE_FILE) as image:
            image.verify()
    except Exception as exc:
        raise RuntimeError("The URL did not return a valid image file.") from exc

    return IMAGE_FILE


def extract_result_dict(item) -> dict:
    # PaddleOCR 3.x returns result objects exposing a JSON representation.
    value = getattr(item, "json", None)
    if callable(value):
        value = value()
    if value is None and isinstance(item, dict):
        value = item
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict) and isinstance(value.get("res"), dict):
        value = value["res"]
    return value if isinstance(value, dict) else {}


def run_ocr(model_path: Path, image_path: Path) -> dict:
    from paddleocr import PaddleOCR

    print("Initializing PaddleOCR on CPU...")
    # The standard mobile detector locates text boxes. The custom model
    # supplied by the user is used for recognition of each detected text line.
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        text_recognition_model_dir=str(model_path),
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        device="cpu",
    )

    print("Running OCR...")
    predictions = ocr.predict(input=str(image_path))
    lines = []

    for prediction in predictions:
        result = extract_result_dict(prediction)
        texts = result.get("rec_texts") or []
        scores = result.get("rec_scores") or []

        for i, text in enumerate(texts):
            text = str(text).strip()
            if not text:
                continue
            score = float(scores[i]) if i < len(scores) else None
            lines.append({"text": text, "score": score})

    return {
        "engine": "PaddleOCR 3.x",
        "device": "cpu",
        "recognition_model": "custom PP-OCRv5 Mobile",
        "line_count": len(lines),
        "lines": lines,
    }


def main() -> int:
    try:
        download_model(env_required("GOOGLE_DRIVE_URL"))
        model_path = locate_recognition_model()
        image_path = download_image(env_required("IMAGE_URL"))
        result = run_ocr(model_path, image_path)

        (ROOT / "ocr-result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        recognized_text = "\n".join(line["text"] for line in result["lines"])
        if not recognized_text:
            recognized_text = "متنی شناسایی نشد."

        (ROOT / "ocr-result.txt").write_text(
            recognized_text + "\n",
            encoding="utf-8",
        )

        print("\n========== OCR OUTPUT ==========")
        print(recognized_text)
        print("================================")
        print("Detected lines:", result["line_count"])
        return 0

    except Exception as exc:
        print(f"\nOCR FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
