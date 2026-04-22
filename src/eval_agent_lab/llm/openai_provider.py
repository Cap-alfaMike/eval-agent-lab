"""OpenAI LLM provider implementation."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from eval_agent_lab.config import CacheConfig, LLMConfig
from eval_agent_lab.exceptions import LLMConnectionError, LLMRateLimitError, LLMResponseError
from eval_agent_lab.llm import BaseLLMProvider, LLMMessage, LLMResponse, StreamChunk


class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible LLM provider (works with OpenAI, Azure, local servers)."""

    def __init__(self, config: LLMConfig, cache_config: Optional[CacheConfig] = None):
        super().__init__(config, cache_config)
        self._client: Optional[Any] = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                kwargs: Dict[str, Any] = {
                    "api_key": self.config.get_api_key(),
                    "timeout": self.config.timeout,
                    "max_retries": self.config.max_retries,
                }
                if self.config.base_url:
                    kwargs["base_url"] = self.config.base_url
                self._client = AsyncOpenAI(**kwargs)
            except ImportError as exc:
                raise LLMConnectionError("openai package not installed") from exc
        return self._client

    def _build_request_kwargs(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Build the common request kwargs dict shared by _call and _stream."""
        request_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": m.role, "content": m.content, **({"name": m.name} if m.name else {}),
                         **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {})} for m in messages],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        return request_kwargs

    async def _call(self, messages: List[LLMMessage],
                    tools: Optional[List[Dict[str, Any]]] = None,
                    **kwargs: Any) -> LLMResponse:
        client = self._get_client()
        request_kwargs = self._build_request_kwargs(messages, tools)

        try:
            response = await client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            exc_str = str(exc)
            if "rate_limit" in exc_str.lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {exc}") from exc
            raise LLMConnectionError(f"OpenAI API error: {exc}") from exc

        choice = response.choices[0]
        tool_calls = []
        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name,
                                "arguments": tc.function.arguments},
                })

        return LLMResponse(
            content=choice.message.content or "",
            tool_calls=tool_calls,
            model=response.model,
            usage={"prompt_tokens": response.usage.prompt_tokens,
                   "completion_tokens": response.usage.completion_tokens},
            finish_reason=choice.finish_reason or "",
        )

    # ------------------------------------------------------------------
    # Streaming support
    # ------------------------------------------------------------------

    async def _stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Yield incremental token chunks from the OpenAI streaming API.

        The presence of this method signals to ``BaseLLMProvider.generate()``
        that streaming is available.  When ``config.stream`` is True,
        ``generate()`` will call ``_collect_stream()`` which consumes this
        iterator and returns a single ``LLMResponse``.
        """
        client = self._get_client()
        request_kwargs = self._build_request_kwargs(messages, tools)
        request_kwargs["stream"] = True

        try:
            stream = await client.chat.completions.create(**request_kwargs)
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                yield StreamChunk(
                    content=delta.content or "",
                    finish_reason=chunk.choices[0].finish_reason,
                    model=chunk.model or "",
                )
        except Exception as exc:
            exc_str = str(exc)
            if "rate_limit" in exc_str.lower():
                raise LLMRateLimitError(f"Rate limit exceeded: {exc}") from exc
            raise LLMConnectionError(f"OpenAI streaming error: {exc}") from exc

    async def stream(
        self,
        messages: List[LLMMessage],
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Public streaming interface — delegates to _stream."""
        async for chunk in self._stream(messages, tools, **kwargs):
            yield chunk

