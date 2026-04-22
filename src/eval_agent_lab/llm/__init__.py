"""LLM provider abstraction layer with caching, batching, and cost tracking."""

from __future__ import annotations

import abc
import hashlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel, Field

from eval_agent_lab.config import CacheConfig, LLMConfig


class LLMMessage(BaseModel):
    """A single message in a conversation."""

    role: str  # system, user, assistant, tool
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class LLMResponse(BaseModel):
    """Structured response from an LLM provider."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    model: str = ""
    usage: dict[str, int] = Field(default_factory=dict)
    latency_ms: float = 0.0
    cached: bool = False
    finish_reason: str = ""
    streamed: bool = False
    raw: dict[str, Any] | None = None


class StreamChunk(BaseModel):
    """A single chunk from a streaming LLM response."""

    content: str = ""
    finish_reason: str | None = None
    model: str = ""


class LLMUsageStats(BaseModel):
    """Aggregated usage statistics for cost tracking."""

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_requests: int = 0
    total_latency_ms: float = 0.0
    cache_hits: int = 0

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    def estimated_cost(
        self, prompt_cost_per_1k: float = 0.00015, completion_cost_per_1k: float = 0.0006
    ) -> float:
        return (
            self.total_prompt_tokens / 1000 * prompt_cost_per_1k
            + self.total_completion_tokens / 1000 * completion_cost_per_1k
        )


class BaseLLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig, cache_config: CacheConfig | None = None):
        self.config = config
        self.stats = LLMUsageStats()
        self._cache: Any | None = None
        if cache_config and cache_config.enabled:
            try:
                import diskcache

                cache_config.directory.mkdir(parents=True, exist_ok=True)
                self._cache = diskcache.Cache(str(cache_config.directory))
            except ImportError:
                pass

    def _cache_key(self, messages: list[LLMMessage], **kwargs: Any) -> str:
        data = json.dumps([m.model_dump() for m in messages], sort_keys=True)
        data += json.dumps(kwargs, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()

    @abc.abstractmethod
    async def _call(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> LLMResponse: ...

    async def generate(
        self, messages: list[LLMMessage], tools: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> LLMResponse:
        """Generate a response with caching and stats tracking.

        When ``config.stream`` is True this method transparently collects
        the full streamed response and returns it as a single LLMResponse.
        """
        cache_key = self._cache_key(messages, tools=tools, **kwargs)

        if self._cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            self.stats.cache_hits += 1
            return LLMResponse(**cached, cached=True)

        # If streaming is enabled and the provider supports it, collect
        # the stream into a single response so callers don't have to change.
        if self.config.stream and hasattr(self, "_stream"):
            return await self._collect_stream(messages, tools, **kwargs)

        start = time.perf_counter()
        response = await self._call(messages, tools, **kwargs)
        response.latency_ms = round((time.perf_counter() - start) * 1000, 2)

        # Update stats
        self.stats.total_requests += 1
        self.stats.total_latency_ms += response.latency_ms
        self.stats.total_prompt_tokens += response.usage.get("prompt_tokens", 0)
        self.stats.total_completion_tokens += response.usage.get("completion_tokens", 0)

        if self._cache:
            self._cache[cache_key] = response.model_dump(exclude={"cached"})

        return response

    async def _collect_stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Consume stream() and aggregate into a single LLMResponse."""
        start = time.perf_counter()
        chunks: list[str] = []
        model = ""
        finish_reason = ""

        async for chunk in self.stream(messages, tools, **kwargs):
            chunks.append(chunk.content)
            if chunk.model:
                model = chunk.model
            if chunk.finish_reason:
                finish_reason = chunk.finish_reason

        content = "".join(chunks)
        latency = round((time.perf_counter() - start) * 1000, 2)

        # Estimate tokens for streamed responses
        prompt_tokens = sum(len(m.content.split()) for m in messages) * 4 // 3
        completion_tokens = len(content.split()) * 4 // 3

        self.stats.total_requests += 1
        self.stats.total_latency_ms += latency
        self.stats.total_prompt_tokens += prompt_tokens
        self.stats.total_completion_tokens += completion_tokens

        return LLMResponse(
            content=content,
            model=model,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            latency_ms=latency,
            finish_reason=finish_reason,
            streamed=True,
        )

    async def stream(
        self,
        messages: list[LLMMessage],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response chunks. Override in providers that support it.

        Default implementation yields the full response as a single chunk.
        """
        response = await self.generate(messages, tools, **kwargs)
        yield StreamChunk(
            content=response.content,
            finish_reason=response.finish_reason,
            model=response.model,
        )

    async def batch_generate(
        self, message_batches: list[list[LLMMessage]], **kwargs: Any
    ) -> list[LLMResponse]:
        """Process multiple requests (sequential; override for parallel)."""
        results = []
        for messages in message_batches:
            results.append(await self.generate(messages, **kwargs))
        return results
