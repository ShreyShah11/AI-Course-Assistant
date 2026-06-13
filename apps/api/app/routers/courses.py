from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_role
from app.db.session import get_db
from app.models import Course, Enrollment, User, UserRole
from app.schemas import CourseCreate, CourseOut, CourseUpdate, UserOut
from app.services.permissions import ensure_teacher_owns_course, get_course_or_404


router = APIRouter(prefix="/courses", tags=["courses"])


@router.post("", response_model=CourseOut, status_code=status.HTTP_201_CREATED)
def create_course(
    payload: CourseCreate,
    db: Session = Depends(get_db),
    user: User = Depends(require_role(UserRole.teacher)),
) -> Course:
    course = Course(title=payload.title, description=payload.description, teacher_id=user.id)
    db.add(course)
    db.commit()
    db.refresh(course)
    return course


@router.get("", response_model=list[CourseOut])
def list_courses(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[Course]:
    if user.role == UserRole.teacher:
        return list(db.scalars(select(Course).where(Course.teacher_id == user.id).order_by(Course.created_at.desc())))
    return list(
        db.scalars(
            select(Course)
            .join(Enrollment)
            .where(Enrollment.student_id == user.id)
            .order_by(Course.created_at.desc())
        )
    )


@router.get("/catalog", response_model=list[CourseOut])
def catalog(db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> list[Course]:
    return list(db.scalars(select(Course).order_by(Course.created_at.desc())))


@router.patch("/{course_id}", response_model=CourseOut)
def update_course(
    course_id: UUID,
    payload: CourseUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Course:
    course = ensure_teacher_owns_course(db, user, course_id)
    if payload.title is not None:
        course.title = payload.title
    if payload.description is not None:
        course.description = payload.description
    db.commit()
    db.refresh(course)
    return course


@router.delete("/{course_id}")
def delete_course(course_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict[str, bool]:
    course = ensure_teacher_owns_course(db, user, course_id)
    db.delete(course)
    db.commit()
    return {"deleted": True}


@router.post("/{course_id}/enroll", status_code=status.HTTP_201_CREATED)
def enroll(course_id: UUID, db: Session = Depends(get_db), user: User = Depends(require_role(UserRole.student))) -> dict:
    get_course_or_404(db, course_id)
    existing = db.scalar(select(Enrollment).where(Enrollment.course_id == course_id, Enrollment.student_id == user.id))
    if existing is not None:
        return {"id": existing.id, "course_id": course_id, "student_id": user.id}
    enrollment = Enrollment(course_id=course_id, student_id=user.id)
    db.add(enrollment)
    db.commit()
    db.refresh(enrollment)
    return {"id": enrollment.id, "course_id": course_id, "student_id": user.id}


@router.get("/{course_id}/students", response_model=list[UserOut])
def enrolled_students(course_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[User]:
    ensure_teacher_owns_course(db, user, course_id)
    return list(db.scalars(select(User).join(Enrollment).where(Enrollment.course_id == course_id)))


@router.get("/{course_id}/analytics")
def analytics(course_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    ensure_teacher_owns_course(db, user, course_id)
    enrollments = len(list(db.scalars(select(Enrollment).where(Enrollment.course_id == course_id))))
    from app.models import ChatHistory, Material, Quiz, QuizResult

    materials = len(list(db.scalars(select(Material).where(Material.course_id == course_id))))
    chats = len(list(db.scalars(select(ChatHistory).where(ChatHistory.course_id == course_id))))
    quizzes = list(db.scalars(select(Quiz).where(Quiz.course_id == course_id)))
    results = []
    if quizzes:
        quiz_ids = [quiz.id for quiz in quizzes]
        results = list(db.scalars(select(QuizResult).where(QuizResult.quiz_id.in_(quiz_ids))))
    average_score = round(sum(result.score for result in results) / len(results), 1) if results else 0
    return {
        "enrolled_students": enrollments,
        "materials": materials,
        "chat_messages": chats,
        "quizzes": len(quizzes),
        "average_score": average_score,
    }
