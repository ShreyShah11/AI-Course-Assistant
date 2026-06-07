from __future__ import annotations

import os
import socket
import sys
import time
from urllib.parse import urlparse

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from rq import Queue, SimpleWorker
from rq.job import Job
from rq.timeouts import TimerDeathPenalty

from apps.worker.core.redis import get_redis_url
from apps.worker.core.rq import QueueSettings, env_int, get_queue


QUEUE_NAME = os.getenv("IMAGE_CHUNKING_QUEUE", "image-chunking")
JOB_FUNCTION = (
    "apps.worker.services.image_chunking.jobs."
    "process_image_chunking_request"
)


def get_queue_settings() -> QueueSettings:
    return QueueSettings(
        name=QUEUE_NAME,
        timeout_seconds=env_int("IMAGE_CHUNKING_JOB_TIMEOUT", 1800),
        result_ttl_seconds=env_int("IMAGE_CHUNKING_RESULT_TTL", 86400),
        failure_ttl_seconds=env_int("IMAGE_CHUNKING_FAILURE_TTL", 604800),
    )


def get_service_queue() -> Queue:
    return get_queue(get_queue_settings())


def enqueue_job(
    file_path: str,
    course_id: str,
    dpi: int = 300,
    course_name: str = "",
    subject_area: str = "",
) -> Job:
    settings = get_queue_settings()
    queue = get_service_queue()
    return queue.enqueue(
        JOB_FUNCTION,
        kwargs={
            "file_path": file_path,
            "dpi": dpi,
            "course_id": course_id,
            "course_name": course_name,
            "subject_area": subject_area,
        },
        job_timeout=settings.timeout_seconds,
        result_ttl=settings.result_ttl_seconds,
        failure_ttl=settings.failure_ttl_seconds,
        meta={
            "file_path": file_path,
            "dpi": dpi,
            "course_id": course_id,
            "course_name": course_name,
            "subject_area": subject_area,
            "queue": queue.name,
        },
    )


def run_worker() -> None:
    redis_url = get_redis_url()
    parsed_url = urlparse(redis_url)
    redis_host = parsed_url.hostname or "localhost"
    redis_port = parsed_url.port or 6379

    for attempt in range(1, 6):
        try:
            socket.getaddrinfo(redis_host, redis_port)
            connection = Redis.from_url(
                redis_url,
                decode_responses=False,
                single_connection_client=True,
            )
            connection.ping()
            break
        except (OSError, RedisConnectionError) as exc:
            if attempt == 5:
                raise
            wait_seconds = attempt * 2
            print(
                f"Redis connection failed for {redis_host}:{redis_port} ({exc}). "
                f"Retrying in {wait_seconds}s "
                f"({attempt}/5)..."
            )
            time.sleep(wait_seconds)
    worker = SimpleWorker([QUEUE_NAME], connection=connection)
    if sys.platform.startswith("win"):
        worker.death_penalty_class = TimerDeathPenalty
    worker.work()
