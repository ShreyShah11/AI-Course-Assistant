from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_current_user, require_role
from app.db.session import get_db
from app.models import Quiz, QuizResult, User, UserRole
from app.schemas import QuizGenerateRequest, QuizOut, QuizResultOut, QuizSubmitRequest
from app.services.langchain_adapter import extract_sources
from app.services.permissions import ensure_course_member, ensure_teacher_owns_course
from routes.agentic_retrieval import QuizRetrievalRequest, quiz_controller


router = APIRouter(tags=["quiz"])


@router.post("/courses/{course_id}/quizzes/generate", response_model=QuizOut, status_code=status.HTTP_201_CREATED)
def generate_quiz(
    course_id: UUID,
    payload: QuizGenerateRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Quiz:
    if user.role == UserRole.teacher:
        ensure_teacher_owns_course(db, user, course_id)
    else:
        ensure_course_member(db, user, course_id)
    query = f"{payload.topic}. Create {payload.question_count} questions."
    retrieval = quiz_controller(QuizRetrievalRequest(query=query, course_id=str(course_id), include_answer_prompt=False))
    quiz = Quiz(
        course_id=course_id,
        generated_content={
            "content": retrieval["final_response"],
            "sources": extract_sources(retrieval),
            "response_model": retrieval.get("response_model"),
        },
    )
    db.add(quiz)
    db.commit()
    db.refresh(quiz)
    return quiz


@router.get("/courses/{course_id}/quizzes", response_model=list[QuizOut])
def list_quizzes(course_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[Quiz]:
    ensure_course_member(db, user, course_id)
    return list(db.scalars(select(Quiz).where(Quiz.course_id == course_id).order_by(Quiz.created_at.desc())))


@router.post("/quizzes/{quiz_id}/submit", response_model=QuizResultOut, status_code=status.HTTP_201_CREATED)
def submit_quiz(
    quiz_id: UUID,
    payload: QuizSubmitRequest,
    db: Session = Depends(get_db),
    user: User = Depends(require_role(UserRole.student)),
) -> QuizResult:
    quiz = db.get(Quiz, quiz_id)
    if quiz is None:
        raise HTTPException(status_code=404, detail="Quiz not found")
    ensure_course_member(db, user, quiz.course_id)
    result = QuizResult(student_id=user.id, quiz_id=quiz_id, score=payload.score, answers=payload.answers)
    db.add(result)
    db.commit()
    db.refresh(result)
    return result


@router.get("/quiz-results", response_model=list[QuizResultOut])
def quiz_history(db: Session = Depends(get_db), user: User = Depends(require_role(UserRole.student))) -> list[QuizResult]:
    return list(db.scalars(select(QuizResult).where(QuizResult.student_id == user.id).order_by(QuizResult.created_at.desc())))
