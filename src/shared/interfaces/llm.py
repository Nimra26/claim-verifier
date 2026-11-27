from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, AsyncIterator, Callable, Mapping


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class ChatMessage:
    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(slots=True)
class GenerationConfig:
    temperature: float = 0.0
    max_tokens: int | None = None
    top_p: float | None = None
    stop: list[str] | None = None
    timeout_s: float | None = None
    response_format: dict[str, Any] | None = None
    # False turns off hidden reasoning on models that think (Qwen3/3.5). None leaves the model's default.
    think: bool | None = None
    # Called once when the reply's first words arrive (18 Nov): a model on a CPU spends most of a call reading the
    # prompt before it writes anything, and the asker's page says which of the two is happening. A client that can
    # stream the reply (OpenAILLM) streams it to notice this and still returns one whole LLMResponse; any other
    # client ignores it. Never given to a call with tools.
    on_first_token: Callable[[], None] | None = None


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str
    finish_reason: str | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any | None = None


class LLM(ABC):
    """Provider-agnostic LLM contract used across agents and RAG pipeline."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""

    @property
    def supports_tool_calls(self) -> bool:
        return False

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        config: GenerationConfig | None = None,
        tools: list[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        """Return a single full completion."""

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        config: GenerationConfig | None = None,
        tools: list[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        """Yield text chunks as they are generated."""
