from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class DatabaseSettings(BaseSettings):
    database_url: str = Field(...)

    @field_validator("database_url", mode="before")
    @classmethod
    def fix_async_scheme(cls, v: str) -> str:
        # Railway (and most PaaS) provide postgres:// or postgresql://
        # asyncpg requires postgresql+asyncpg://
        if v.startswith("postgres://"):
            return "postgresql+asyncpg://" + v[len("postgres://"):]
        if v.startswith("postgresql://"):
            return "postgresql+asyncpg://" + v[len("postgresql://"):]
        return v