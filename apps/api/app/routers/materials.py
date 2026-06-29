from __future__ import annotations

from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, Depends, File, UploadFile, status
from fastapi.responses import FileResponse
from fastapi import APIRouter, Depends, File, UploadFile, status, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models import Material, User
from app.schemas import MaterialOut, YouTubeMaterialCreate
from app.services.langchain_adapter import enqueue_ingestion
from app.services.permissions import ensure_course_member, ensure_teacher_owns_course
from fastapi.responses import RedirectResponse
from app.services.storage import store_upload, create_download_url, delete_upload


router = APIRouter(prefix="/courses/{course_id}/materials", tags=["materials"])


@router.post("/files", response_model=MaterialOut, status_code=status.HTTP_201_CREATED)
def upload_file(
    course_id: UUID,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Material:
    ensure_teacher_owns_course(db, user, course_id)
    path = store_upload(course_id, file)
    file_type = Path(file.filename or path.name).suffix.lower().lstrip(".") or "file"
    job_id = enqueue_ingestion(file_path=str(path), course_id=str(course_id), file_type=file_type)
    material = Material(
        course_id=course_id,
        file_name=file.filename or path.name,
        file_type=file_type,
        storage_path=str(path),
        ingestion_job_id=job_id,
    )
    db.add(material)
    db.commit()
    db.refresh(material)
    return material


@router.post("/youtube", response_model=MaterialOut, status_code=status.HTTP_201_CREATED)
def upload_youtube(
    course_id: UUID,
    payload: YouTubeMaterialCreate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Material:
    ensure_teacher_owns_course(db, user, course_id)
    job_id = enqueue_ingestion(file_path=payload.url, course_id=str(course_id), file_type="youtube")
    material = Material(
        course_id=course_id,
        file_name=payload.url,
        file_type="youtube",
        storage_path=payload.url,
        ingestion_job_id=job_id,
    )
    db.add(material)
    db.commit()
    db.refresh(material)
    return material


@router.get("", response_model=list[MaterialOut])
def list_materials(course_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[Material]:
    ensure_course_member(db, user, course_id)
    return list(db.scalars(select(Material).where(Material.course_id == course_id).order_by(Material.created_at.desc())))


@router.delete("/{material_id}")
def delete_material(
    course_id: UUID,
    material_id: UUID,
    db: Session = Depends(get_db),
) -> dict[str, bool]:
    material = db.get(Material, material_id)
    if material is not None and material.course_id == course_id:
        if material.file_type != "youtube":
            delete_upload(material.storage_path)
        db.delete(material)
        db.commit()
    return {"deleted": True}

@router.get("/{material_id}/download")
def download_material(
    course_id: UUID,
    material_id: UUID,
    db: Session = Depends(get_db),
):
    material = db.get(Material, material_id)

    if material is None:
        raise HTTPException(status_code=404, detail="Material not found")

    url = create_download_url(material.storage_path)

    return RedirectResponse(url)