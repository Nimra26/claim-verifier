from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class AISettings(BaseSettings):
    openai_key: Optional[str] = Field(default=None)
    ollama_key: Optional[str] = Field(default=None)
    ollama_base_url: Optional[str]= Field(default=None)
    model: Optional[str] = "qwen3.5:4b"  # local by default (2 Nov, strict: was gpt-oss:120b-cloud); 4b from 4 Nov (was qwen3:1.7b)
