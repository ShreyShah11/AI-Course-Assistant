from __future__ import annotations

import random

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from app.db.session import get_db
from app.models import User, EmailOTP
from app.schemas import (
    LoginRequest,
    TokenResponse,
    UserCreate,
    UserOut,
    SendOTPRequest,
    VerifyOTPRequest,
)
from app.services.email_service import send_otp_email

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/send-otp")
def send_otp(
    payload: SendOTPRequest,
    db: Session = Depends(get_db),
):
    otp = str(random.randint(100000, 999999))

    row = EmailOTP(
        email=payload.email.lower(),
        otp=otp,
    )

    db.add(row)
    db.commit()

    send_otp_email(
        payload.email,
        otp,
    )

    return {"message": "OTP sent"}


@router.post(
    "/verify-otp"
)
def verify_otp(
    payload: VerifyOTPRequest,
    db: Session = Depends(get_db),
):
    row = db.scalar(
        select(EmailOTP).where(
            EmailOTP.email == payload.email.lower(),
            EmailOTP.otp == payload.otp,
        )
    )

    if not row:
        raise HTTPException(
            status_code=400,
            detail="Invalid OTP",
        )

    row.verified = True
    db.commit()

    return {"verified": True}


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
)
def register(
    payload: UserCreate,
    db: Session = Depends(get_db),
) -> TokenResponse:

    existing = db.scalar(
        select(User).where(
            User.email == payload.email.lower()
        )
    )

    if existing:
        raise HTTPException(
            status_code=409,
            detail="Email is already registered",
        )

    otp_row = db.scalar(
        select(EmailOTP).where(
            EmailOTP.email == payload.email.lower(),
            EmailOTP.verified == True,
        )
    )

    if not otp_row:
        raise HTTPException(
            status_code=403,
            detail="Email not verified",
        )

    user = User(
        name=payload.name,
        email=payload.email.lower(),
        password_hash=hash_password(payload.password),
        role=payload.role,
    )

    db.add(user)
    db.commit()
    db.refresh(user)

    return TokenResponse(
        access_token=create_access_token(user),
        user=UserOut.model_validate(user),
    )


@router.post(
    "/login",
    response_model=TokenResponse,
)
def login(
    payload: LoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:

    user = db.scalar(
        select(User).where(
            User.email == payload.email.lower()
        )
    )

    if user is None or not verify_password(
        payload.password,
        user.password_hash,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password",
        )

    return TokenResponse(
        access_token=create_access_token(user),
        user=UserOut.model_validate(user),
    )


@router.post("/logout")
def logout() -> dict[str, bool]:
    return {"ok": True}


@router.get(
    "/me",
    response_model=UserOut,
)
def me(
    user: User = Depends(get_current_user),
) -> User:
    return user