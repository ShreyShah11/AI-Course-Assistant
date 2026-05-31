from __future__ import annotations

import argparse

from apps.worker.core.config import load_environment

load_environment("worker")

from apps.worker.services.audio_chunking.workers import run_worker as run_audio_chunking_worker
from apps.worker.services.document_chunking.worker import run_worker as run_document_chunking_worker
from apps.worker.services.image_chunking.worker import run_worker as run_image_chunking_worker
from apps.worker.services.qna_chunking.worker import run_worker as run_qna_chunking_worker


SERVICE_WORKERS = {
    "audio-chunking": run_audio_chunking_worker,
    "document-chunking": run_document_chunking_worker,
    "image-chunking": run_image_chunking_worker,
    "qna-chunking": run_qna_chunking_worker,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an RQ worker for a service.")
    parser.add_argument(
        "service",
        nargs="?",
        default="image-chunking",
        choices=sorted(SERVICE_WORKERS),
        help="Service queue to consume.",
    )
    args = parser.parse_args()
    SERVICE_WORKERS[args.service]()


if __name__ == "__main__":
    main()
