from pydantic_settings import BaseSettings, SettingsConfigDict


class VerifierSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VERIFIER_")

    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:55432/postgres"
    ollama_url: str = "http://localhost:11434"
    embed_model: str = "bge-m3"
    model: str = "claude-sonnet-5-5"  # the Claude model ("claude" / "api" backends)
    local_model: str = "qwen3.5:4b"  # the Ollama model ("local" backend), as Ask the Workspace uses
    # "local": reasoning in Ollama on this machine, nothing leaves it (default, private).
    # "cli" (or "claude"): Claude through the Claude Code login. "api": Claude with ANTHROPIC_API_KEY.
    backend: str = "local"

    def model_for(self, backend: str) -> str:
        from verifier.reasoner import normalize_backend

        return self.local_model if normalize_backend(backend) == "local" else self.model
    top_k: int = 8
    chunk_size: int = 800
    chunk_overlap: int = 120
