"""
Settings for the standalone Code Charlie Streamlit app.

Trimmed from KARR-AI's core/config.py — only what Code Charlie needs:
Supabase connection, OpenAI keys, model names, checkpoint pool tuning,
and the password gate + fixed "gate user" UUID.

Loads from .env locally and from Streamlit Cloud secrets in production
(secrets.toml is mirrored to env vars by the Streamlit runtime).
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase (shared with KARR-AI: same project, same compliance_embeddings)
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str

    # PostgreSQL transaction-pooler URL for LangGraph checkpoints.
    # Get from Supabase Dashboard > Connect.
    DATABASE_URL: str
    CHECKPOINT_POOL_MIN_SIZE: int = 0
    CHECKPOINT_POOL_MAX_SIZE: int = 2
    CHECKPOINT_POOL_TIMEOUT_SECONDS: int = 60
    CHECKPOINT_POOL_MAX_IDLE_SECONDS: int = 600
    CHECKPOINT_POOL_MAX_LIFETIME_SECONDS: int = 3600

    # OpenAI
    OPENAI_API_KEY: str
    COMPLIANCE_EMBED_MODEL: str = "text-embedding-3-large"
    COMPLIANCE_EMBED_DIMS: int = 3072
    GEN_MODEL: str = "gpt-5.5"

    # Streamlit gate (single shared password)
    GATE_PASSWORD: str
    # Fixed UUID owning every Streamlit session. Generate once
    # (e.g. python -c "import uuid; print(uuid.uuid4())") and keep it.
    GATE_USER_ID: str

    class Config:
        env_file = ".env"


settings = Settings()
