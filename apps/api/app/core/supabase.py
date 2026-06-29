from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from supabase import Client, create_client

ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

print("Loading:", ENV_PATH)
print("Exists:", ENV_PATH.exists())

load_dotenv(ENV_PATH)

print("SUPABASE_URL =", os.getenv("SUPABASE_URL"))
print("SERVICE_ROLE =", bool(os.getenv("SUPABASE_SERVICE_ROLE_KEY")))

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY"),
)