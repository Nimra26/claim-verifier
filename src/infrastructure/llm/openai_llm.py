from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Callable, Mapping

import openai
from openai import AsyncOpenAI, DefaultAsyncHttpxClient

from src.shared import resilience

from src.shared.interfaces.llm import (
    ChatMessage,
    GenerationConfig,
    LLM,
    LLMResponse,
    MessageRole,
    TokenUsage,
    ToolCall,
)

log = logging.getLogger(__name__)


class OpenAILLM(LLM):
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str | None = None,
        organization: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        # trust_env=False: httpx would otherwise follow HTTP(S)_PROXY / ALL_PROXY from the environment even for
        # http://localhost, sending the question and passages to a proxy off this machine (strict local mode, 2 Nov).
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            organization=organization,
            max_retries=max_retries,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def supports_tool_calls(self) -> bool:
        return True

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        config: GenerationConfig | None = None,
        tools: list[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        cfg = config or GenerationConfig()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_openai_message(msg) for msg in messages],
            "temperature": cfg.temperature,
        }
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        if cfg.stop:
            payload["stop"] = cfg.stop
        if cfg.timeout_s is not None:
            payload["timeout"] = cfg.timeout_s
        if cfg.response_format is not None:
            payload["response_format"] = cfg.response_format
        if cfg.think is False:
            # Ollama 0.40 ignores the "/no_think" prompt switch; this request field is what turns reasoning off.
            # Without it Qwen spent the whole max_tokens on hidden reasoning (28 Oct: empty and one-word answers).
            payload["extra_body"] = {"reasoning_effort": "none"}
        if tools:
            payload["tools"] = list(tools)
        elif cfg.on_first_token is not None:
            # Streamed only to notice the first words (GenerationConfig.on_first_token); the caller still gets one
            # whole reply, under the same breaker and the same time limit for the whole call.
            return await resilience.call("model server", lambda: self._streamed(payload, cfg.timeout_s,
                                                                                 cfg.on_first_token))

        completion = await resilience.call("model server", lambda: self._client.chat.completions.create(**payload))
        choice = completion.choices[0]
        message = choice.message

        usage = TokenUsage(
            prompt_tokens=(completion.usage.prompt_tokens if completion.usage else 0),
            completion_tokens=(
                completion.usage.completion_tokens if completion.usage else 0
            ),
        )

        return LLMResponse(
            content=message.content or "",
            model=completion.model,
            finish_reason=choice.finish_reason,
            usage=usage,
            tool_calls=self._extract_tool_calls(message),
            raw=completion,
        )

    async def _streamed(self, payload: dict[str, Any], timeout_s: float | None,
                        on_first_token: Callable[[], None]) -> LLMResponse:
        """generate() read as a stream: the same reply, plus on_first_token() when its first words arrive.
        timeout_s bounds the WHOLE call, as it does a reply read in one piece (a stream's own timeout is per read)."""
        payload = {**payload, "stream": True, "stream_options": {"include_usage": True}}

        async def read() -> LLMResponse:
            parts: list[str] = []
            model, finish, usage, told = self._model, None, None, False
            stream = await self._client.chat.completions.create(**payload)
            try:
                async for chunk in stream:
                    model = getattr(chunk, "model", None) or model
                    if getattr(chunk, "usage", None) is not None:
                        usage = chunk.usage
                    if not chunk.choices:
                        continue
                    choice = chunk.choices[0]
                    text = getattr(choice.delta, "content", None)
                    if text:
                        if not told:
                            told = True
                            try:
                                on_first_token()
                            except Exception:  # a listener must never break the answer
                                log.debug("on_first_token listener failed", exc_info=True)
                        parts.append(text)
                    if choice.finish_reason:
                        finish = choice.finish_reason
            finally:
                await stream.close()  # the model server stops writing a reply nobody reads
            return LLMResponse(
                content="".join(parts), model=model, finish_reason=finish,
                usage=TokenUsage(prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                                 completion_tokens=getattr(usage, "completion_tokens", 0) or 0),
                tool_calls=[], raw=None)

        if not timeout_s:
            return await read()
        try:
            async with asyncio.timeout(timeout_s) as limit:
                return await read()
        except TimeoutError:
            if not limit.expired():
                raise
            # The same error a reply read in one piece gives (the answer's fallback path and the breaker know it). No
            # request object: this file talks to the model server only through the openai client (no httpx import).
            raise openai.APITimeoutError(request=None) from None

    async def stream(
        self,
        messages: list[ChatMessage],
        *,
        config: GenerationConfig | None = None,
        tools: list[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        cfg = config or GenerationConfig()
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_openai_message(msg) for msg in messages],
            "temperature": cfg.temperature,
            "stream": True,
        }
        if cfg.max_tokens is not None:
            payload["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            payload["top_p"] = cfg.top_p
        if cfg.stop:
            payload["stop"] = cfg.stop
        if cfg.timeout_s is not None:
            payload["timeout"] = cfg.timeout_s
        if cfg.response_format is not None:
            payload["response_format"] = cfg.response_format
        if cfg.think is False:
            # Ollama 0.40 ignores the "/no_think" prompt switch; this request field is what turns reasoning off.
            # Without it Qwen spent the whole max_tokens on hidden reasoning (28 Oct: empty and one-word answers).
            payload["extra_body"] = {"reasoning_effort": "none"}
        if tools:
            payload["tools"] = list(tools)

        stream = await self._client.chat.completions.create(**payload)
        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                yield token

    @staticmethod
    def _to_openai_message(message: ChatMessage) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role.value,
            "content": message.content,
        }
        if message.name:
            payload["name"] = message.name
        if message.role == MessageRole.TOOL and message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.role == MessageRole.ASSISTANT and message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in message.tool_calls
            ]
        return payload

    @staticmethod
    def _extract_tool_calls(message: Any) -> list[ToolCall]:
        tool_calls: list[ToolCall] = []
        raw_calls = getattr(message, "tool_calls", None) or []
        for call in raw_calls:
            arguments_raw = call.function.arguments or "{}"
            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError:
                arguments = {"_raw": arguments_raw}

            tool_calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=arguments,
                )
            )
        return tool_calls
