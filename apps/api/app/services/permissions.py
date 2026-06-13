from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Course, Enrollment, User, UserRole


def get_course_or_404(db: Session, course_id: UUID) -> Course:
    course = db.get(Course, course_id)
    if course is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Course not found")
    return course


def ensure_teacher_owns_course(db: Session, user: User, course_id: UUID) -> Course:
    course = get_course_or_404(db, course_id)
    if user.role != UserRole.teacher or course.teacher_id != user.id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Teacher access required")
    return course


def ensure_course_member(db: Session, user: User, course_id: UUID) -> Course:
    course = get_course_or_404(db, course_id)
    if user.role == UserRole.teacher and course.teacher_id == user.id:
        return course
    enrolled = db.scalar(
        select(Enrollment).where(Enrollment.course_id == course_id, Enrollment.student_id == user.id)
    )
    if enrolled is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Course enrollment required")
    return course
