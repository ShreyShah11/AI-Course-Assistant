from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / "apps" / "api" / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "CourseGPT"
    database_url: str = Field(
        default="postgresql+psycopg2://coursegpt:coursegpt@localhost:5432/coursegpt",
        validation_alias="DATABASE_URL",
    )
    jwt_secret_key: str = Field(default="change-me-in-production", validation_alias="JWT_SECRET_KEY")
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24
    upload_dir: Path = Field(default=PROJECT_ROOT / "storage" / "materials", validation_alias="UPLOAD_DIR")

    @property
    def public_api_url(self) -> str:
        return os.getenv("NEXT_PUBLIC_API_URL", "http://127.0.0.1:8000")


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
