from __future__ import annotations

import tempfile
from pathlib import Path

from app.core.supabase import supabase

BUCKET = "materials"


def create_download_url(object_key: str) -> str:
    response = supabase.storage.from_(BUCKET).create_signed_url(
        object_key,
        3600,
    )
    return response["signedURL"]


def delete_upload(object_key: str) -> None:
    supabase.storage.from_(BUCKET).remove([object_key])


def download_to_temp(object_key: str) -> Path:
    data = supabase.storage.from_(BUCKET).download(object_key)

    suffix = Path(object_key).suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        return Path(tmp.name)