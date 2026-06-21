from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import UserRole


class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=160)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: UserRole


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: EmailStr
    role: UserRole


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class CourseCreate(BaseModel):
    title: str = Field(..., min_length=2, max_length=220)
    description: str = ""


class CourseUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=2, max_length=220)
    description: str | None = None


class CourseOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    description: str
    teacher_id: uuid.UUID
    created_at: datetime


class EnrollmentOut(BaseModel):
    id: uuid.UUID
    student_id: uuid.UUID
    course_id: uuid.UUID
    student: UserOut | None = None


class MaterialOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    course_id: uuid.UUID
    file_name: str
    file_type: str
    storage_path: str
    ingestion_job_id: str | None = None
    created_at: datetime


class YouTubeMaterialCreate(BaseModel):
    url: str = Field(..., min_length=10)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    id: uuid.UUID
    message: str
    response: str
    sources: list[dict[str, Any]]


class QuizGenerateRequest(BaseModel):
    topic: str = Field(default="Generate a balanced quiz from the course materials")
    question_count: int = Field(default=10, ge=1, le=50)


class QuizOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    course_id: uuid.UUID
    generated_content: dict[str, Any]
    created_at: datetime


class QuizSubmitRequest(BaseModel):
    answers: dict[str, Any]
    score: int = Field(..., ge=0)


class QuizResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    student_id: uuid.UUID
    quiz_id: uuid.UUID
    score: int
    answers: dict[str, Any]
    created_at: datetime

class SendOTPRequest(BaseModel):
    email: EmailStr


class VerifyOTPRequest(BaseModel):
    email: EmailStr
    otp: str


class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str
    role: UserRole