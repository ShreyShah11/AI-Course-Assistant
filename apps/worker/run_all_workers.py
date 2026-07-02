"""
run_all_workers.py
------------------
Starts ALL RQ workers in a single process using Python threads.
Each worker listens to its own named queue — one thread per queue.

Usage:
    python -m apps.worker.run_all_workers          # all queues
    python -m apps.worker.run_all_workers audio document  # subset only

This is the recommended entry point for the Docker worker container on Render.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.worker.core.config import load_environment

load_environment("worker")

from apps.worker.services.audio_chunking.workers import run_worker as run_audio
from apps.worker.services.document_chunking.worker import run_worker as run_document
from apps.worker.services.handwritten_chunking.worker import run_worker as run_handwritten
from apps.worker.services.image_chunking.worker import run_worker as run_image
from apps.worker.services.qna_chunking.worker import run_worker as run_qna
from apps.worker.services.youtube_chunking.worker import run_worker as run_youtube


ALL_WORKERS: dict[str, callable] = {
    "audio": run_audio,
    "document": run_document,
    "handwritten": run_handwritten,
    "image": run_image,
    "qna": run_qna,
    "youtube": run_youtube,
}


def _start_worker_thread(name: str, fn: callable) -> threading.Thread:
    """Wrap a worker run function in a daemon thread."""
    def _target():
        print(f"[worker:{name}] Starting...", flush=True)
        try:
            fn()
        except Exception as exc:
            print(f"[worker:{name}] CRASHED: {exc}", flush=True)

    t = threading.Thread(target=_target, name=f"worker-{name}", daemon=True)
    t.start()
    print(f"[worker:{name}] Thread launched (id={t.ident})", flush=True)
    return t


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multiple RQ workers in one process (one thread per queue)."
    )
    parser.add_argument(
        "queues",
        nargs="*",
        default=list(ALL_WORKERS.keys()),
        choices=list(ALL_WORKERS.keys()),
        help="Queues to start. Defaults to all queues.",
    )
    args = parser.parse_args()

    selected: dict[str, callable] = {k: ALL_WORKERS[k] for k in args.queues}

    print(
        f"[run_all_workers] Starting {len(selected)} workers: {', '.join(selected)}",
        flush=True,
    )

    threads = [_start_worker_thread(name, fn) for name, fn in selected.items()]

    # Keep the main thread alive — if ALL worker threads die, exit with error.
    try:
        while True:
            alive = [t for t in threads if t.is_alive()]
            if not alive:
                print("[run_all_workers] All worker threads have stopped. Exiting.", flush=True)
                sys.exit(1)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\n[run_all_workers] Received SIGINT. Shutting down.", flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
