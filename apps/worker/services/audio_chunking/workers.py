from __future__ import annotations

import os
import sys

from redis import Redis
from rq import Queue, SimpleWorker
from rq.job import Job
from rq.timeouts import TimerDeathPenalty

from apps.worker.core.redis import get_redis_url
from apps.worker.core.rq import QueueSettings, env_int, get_queue


QUEUE_NAME = os.getenv("AUDIO_CHUNKING_QUEUE", "audio-chunking")
JOB_FUNCTION = (
    "apps.worker.services.audio_chunking.jobs."
    "process_audio_chunking_request"
)


def get_queue_settings() -> QueueSettings:
    return QueueSettings(
        name=QUEUE_NAME,
        timeout_seconds=env_int("AUDIO_CHUNKING_JOB_TIMEOUT", 7200),
        result_ttl_seconds=env_int("AUDIO_CHUNKING_RESULT_TTL", 86400),
        failure_ttl_seconds=env_int("AUDIO_CHUNKING_FAILURE_TTL", 604800),
    )


def get_service_queue() -> Queue:
    return get_queue(get_queue_settings())


def enqueue_job(
    file_path: str,
    course_id: str,
    course_name: str = "",
    lecture_id: str = "",
    lecture_number: int = 0,
    week_number: int = 0,
    lecture_title: str = "",
    professor: str = "",
) -> Job:
    settings = get_queue_settings()
    queue = get_service_queue()
    kwargs = {
        "file_path": file_path,
        "course_id": course_id,
        "course_name": course_name,
        "lecture_id": lecture_id,
        "lecture_number": lecture_number,
        "week_number": week_number,
        "lecture_title": lecture_title,
        "professor": professor,
    }
    return queue.enqueue(
        JOB_FUNCTION,
        kwargs=kwargs,
        job_timeout=settings.timeout_seconds,
        result_ttl=settings.result_ttl_seconds,
        failure_ttl=settings.failure_ttl_seconds,
        meta={**kwargs, "queue": queue.name},
    )


def run_worker() -> None:
    connection = Redis.from_url(get_redis_url(), decode_responses=False)
    worker = SimpleWorker([QUEUE_NAME], connection=connection)
    if sys.platform.startswith("win"):
        worker.death_penalty_class = TimerDeathPenalty
    worker.work()
