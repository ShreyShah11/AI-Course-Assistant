from __future__ import annotations

from redis import Redis
from rq import SimpleWorker

from apps.worker.core.config import load_environment
from apps.worker.core.redis import get_redis_url

load_environment("worker")

QUEUES = [
    "document-chunking",
    "image-chunking",
    "audio-chunking",
    "handwritten-chunking",
    "qna-chunking",
    "youtube-chunking",
]


def main():
    connection = Redis.from_url(get_redis_url(), decode_responses=False)
    worker = SimpleWorker(QUEUES, connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
