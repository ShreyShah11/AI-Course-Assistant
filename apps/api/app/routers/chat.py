from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import get_current_user
from app.db.session import get_db
from app.models import ChatHistory, User
from app.schemas import ChatRequest, ChatResponse
from app.services.langchain_adapter import extract_sources
from app.services.permissions import ensure_course_member
from routes.agentic_retrieval import AskRetrievalRequest, ask_controller, summarize_controller, SummarizeRetrievalRequest


router = APIRouter(prefix="/courses/{course_id}/chat", tags=["chat"])


@router.post("/ask", response_model=ChatResponse)
def ask_course(
    course_id: UUID,
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    print("CHAT ENDPOINT HIT")
    print("course_id =", course_id)
    print("user =", user.email if user else None)
    ensure_course_member(db, user, course_id)
    recent = list(
        db.scalars(
            select(ChatHistory)
            .where(ChatHistory.course_id == course_id, ChatHistory.user_id == user.id)
            .order_by(ChatHistory.created_at.desc())
            .limit(4)
        )
    )
    memory = "\n".join(f"Student: {row.message}\nAssistant: {row.response[:500]}" for row in reversed(recent))
    query = f"Recent course chat memory:\n{memory}\n\nCurrent question: {payload.message}" if memory else payload.message
    retrieval = ask_controller(AskRetrievalRequest(query=query, course_id=str(course_id), include_answer_prompt=False))
    sources = extract_sources(retrieval)
    history = ChatHistory(
        user_id=user.id,
        course_id=course_id,
        message=payload.message,
        response=retrieval["final_response"],
        sources={"items": sources},
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return ChatResponse(id=history.id, message=history.message, response=history.response, sources=sources)


@router.post("/flashcards", response_model=ChatResponse)
def flashcards(
    course_id: UUID,
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    ensure_course_member(db, user, course_id)
    prompt = f"Generate concise study flashcards as Q/A pairs for: {payload.message}"
    retrieval = summarize_controller(SummarizeRetrievalRequest(query=prompt, course_id=str(course_id), include_answer_prompt=False))
    sources = extract_sources(retrieval)
    history = ChatHistory(
        user_id=user.id,
        course_id=course_id,
        message=f"Flashcards: {payload.message}",
        response=retrieval["final_response"],
        sources={"items": sources},
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return ChatResponse(id=history.id, message=history.message, response=history.response, sources=sources)


@router.post("/summary", response_model=ChatResponse)
def summarize_course(
    course_id: UUID,
    payload: ChatRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatResponse:
    ensure_course_member(db, user, course_id)
    retrieval = summarize_controller(
        SummarizeRetrievalRequest(query=payload.message, course_id=str(course_id), include_answer_prompt=False)
    )
    sources = extract_sources(retrieval)
    history = ChatHistory(
        user_id=user.id,
        course_id=course_id,
        message=f"Summary: {payload.message}",
        response=retrieval["final_response"],
        sources={"items": sources},
    )
    db.add(history)
    db.commit()
    db.refresh(history)
    return ChatResponse(id=history.id, message=history.message, response=history.response, sources=sources)


@router.get("/history", response_model=list[ChatResponse])
def history(course_id: UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[ChatResponse]:
    ensure_course_member(db, user, course_id)
    rows = db.scalars(
        select(ChatHistory)
        .where(ChatHistory.course_id == course_id, ChatHistory.user_id == user.id)
        .order_by(ChatHistory.created_at.desc())
    )
    return [
        ChatResponse(id=row.id, message=row.message, response=row.response, sources=row.sources.get("items", []))
        for row in rows
    ]
