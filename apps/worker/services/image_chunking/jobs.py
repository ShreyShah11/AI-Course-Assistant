from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from apps.worker.services.image_chunking.run_pipeline import run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[4]
IMAGE_PIPELINE_DIR = (
    PROJECT_ROOT
    / "apps"
    / "api"
    / "pipelines"
    / "chunking pipeline"
    / "image pipeline"
)

for import_path in (PROJECT_ROOT, IMAGE_PIPELINE_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def load_ocr_worker_class() -> type:
    ocr_worker_path = IMAGE_PIPELINE_DIR / "ocr_worker.py"
    spec = importlib.util.spec_from_file_location("ocr_worker", ocr_worker_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load OCR worker from {ocr_worker_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ocr_worker", module)
    spec.loader.exec_module(module)
    return module.OCRWorker


OCRWorker = load_ocr_worker_class()


SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}


def validate_file_path(file_path: str | Path) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "Unsupported file type. Supported: "
            + ", ".join(sorted(SUPPORTED_EXTENSIONS))
        )

    return path


def process_image_chunking_request(
    file_path: str,
    course_id: str,
    dpi: int = 300,
    course_name: str = "",
    subject_area: str = "",
) -> dict[str, Any]:
    path = validate_file_path(file_path)
    worker = OCRWorker(debug=False)
    return run_pipeline(
        file_path=path,
        ocr_worker=worker,
        dpi=dpi,
        course_id=course_id,
        course_name=course_name,
        subject_area=subject_area,
    )
