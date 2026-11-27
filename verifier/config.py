from pydantic_settings import BaseSettings, SettingsConfigDict


class VerifierSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERIFIER_")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:55432/postgres"
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    model: str = "claude-sonnet-5-5"
    backend: str = "cli"  # "cli": local Claude Code login; "api": ANTHROPIC_API_KEY
    top_k: int = 8
    chunk_size: int = 800
    chunk_overlap: int = 120
