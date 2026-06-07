from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, field_validator
from redis.exceptions import RedisError
from rq.job import Job


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apps.worker.core.rq import fetch_job, serialize_job  # noqa: E402
from apps.worker.services.handwritten_chunking.run_pipeline import SUPPORTED_EXTENSIONS  # noqa: E402
from apps.worker.services.handwritten_chunking.worker import (  # noqa: E402
    enqueue_job as enqueue_handwritten_chunking_job,
    get_service_queue,
)


class HandwrittenChunkingJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_path: str = Field(..., description="Path to handwritten PDF/image")
    course_id: str = Field(..., min_length=1)
    course_name: str = ""
    subject: str = ""
    instructor: str = ""
    semester: str = ""
    university: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value):
        if value in (None, "", "string"):
            return []
        if isinstance(value, list):
            return [item for item in value if item and item != "string"]
        return value


router = APIRouter(prefix="/handwritten-chunking/jobs", tags=["handwritten-chunking-jobs"])


def validate_submission_path(file_path: str) -> Path:
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {path}")
    if not path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type. Supported: " + ", ".join(sorted(SUPPORTED_EXTENSIONS)))
    return path


def enqueue_job(request: HandwrittenChunkingJobRequest) -> dict:
    path = validate_submission_path(request.file_path)
    payload = request.model_dump()
    payload["file_path"] = str(path)
    try:
        job = enqueue_handwritten_chunking_job(**payload)
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Could not connect to Redis/RQ: {exc}") from exc
    return serialize_job(job)


def get_job_or_404(job_id: str) -> Job:
    try:
        job = fetch_job(job_id)
    except RedisError as exc:
        raise HTTPException(status_code=503, detail=f"Could not connect to Redis/RQ: {exc}") from exc
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@router.post("")
async def add_handwritten_chunking_job(request: HandwrittenChunkingJobRequest) -> dict:
    return await run_in_threadpool(enqueue_job, request)


@router.get("")
async def list_handwritten_chunking_jobs(limit: Annotated[int, Query(ge=1, le=100)] = 20) -> dict:
    def list_jobs() -> dict:
        try:
            queue = get_service_queue()
            jobs = [Job.fetch(job_id, connection=queue.connection) for job_id in queue.job_ids[:limit]]
            return {"queue": queue.name, "count": len(jobs), "jobs": [serialize_job(job) for job in jobs]}
        except RedisError as exc:
            raise HTTPException(status_code=503, detail=f"Could not connect to Redis/RQ: {exc}") from exc
    return await run_in_threadpool(list_jobs)


@router.get("/{job_id}")
async def get_handwritten_chunking_job(job_id: str) -> dict:
    return serialize_job(await run_in_threadpool(get_job_or_404, job_id))


@router.get("/{job_id}/result")
async def get_handwritten_chunking_job_result(job_id: str) -> dict:
    job = await run_in_threadpool(get_job_or_404, job_id)
    payload = serialize_job(job, include_result=True)
    if payload["status"] != "finished":
        payload["result"] = None
    return payload


@router.delete("/{job_id}")
async def delete_handwritten_chunking_job(job_id: str) -> dict:
    def delete_job() -> dict:
        job = get_job_or_404(job_id)
        if job.get_status(refresh=True) in {"queued", "deferred", "scheduled"}:
            job.cancel()
        job.delete()
        return {"id": job_id, "deleted": True}
    return await run_in_threadpool(delete_job)
