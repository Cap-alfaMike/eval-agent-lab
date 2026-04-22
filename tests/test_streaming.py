"""Unit tests for the streaming LLM layer."""

import pytest

from eval_agent_lab.config import LLMConfig
from eval_agent_lab.llm import BaseLLMProvider, LLMMessage, LLMResponse, StreamChunk


class MockStreamingProvider(BaseLLMProvider):
    """Mock provider that supports streaming."""

    async def _call(self, messages, tools=None, **kwargs):
        content = "Hello, I am a mock response."
        return LLMResponse(
            content=content,
            model="mock-model",
            usage={"prompt_tokens": 10, "completion_tokens": 7},
            finish_reason="stop",
        )

    async def _stream(self, messages, tools=None, **kwargs):
        """Simulate streaming by yielding word-by-word."""
        words = ["Hello, ", "I am ", "a mock ", "response."]
        for i, word in enumerate(words):
            yield StreamChunk(
                content=word,
                finish_reason="stop" if i == len(words) - 1 else None,
                model="mock-model",
            )

    async def stream(self, messages, tools=None, **kwargs):
        async for chunk in self._stream(messages, tools, **kwargs):
            yield chunk


@pytest.mark.unit
class TestStreamChunk:
    def test_stream_chunk_model(self):
        chunk = StreamChunk(content="hello", model="gpt-4o")
        assert chunk.content == "hello"
        assert chunk.model == "gpt-4o"
        assert chunk.finish_reason is None


@pytest.mark.unit
class TestStreamingProvider:
    @pytest.fixture
    def provider(self):
        config = LLMConfig(api_key="fake", stream=False)
        return MockStreamingProvider(config)

    @pytest.fixture
    def streaming_provider(self):
        config = LLMConfig(api_key="fake", stream=True)
        return MockStreamingProvider(config)

    @pytest.mark.asyncio
    async def test_non_streaming_generate(self, provider):
        """Normal generate() works as before."""
        messages = [LLMMessage(role="user", content="hello")]
        response = await provider.generate(messages)
        assert response.content == "Hello, I am a mock response."
        assert not response.streamed

    @pytest.mark.asyncio
    async def test_streaming_generate_collects(self, streaming_provider):
        """When config.stream=True, generate() collects streamed chunks."""
        messages = [LLMMessage(role="user", content="hello")]
        response = await streaming_provider.generate(messages)
        assert response.content == "Hello, I am a mock response."
        assert response.streamed is True
        assert response.model == "mock-model"

    @pytest.mark.asyncio
    async def test_direct_stream(self, provider):
        """Direct stream() iteration works."""
        messages = [LLMMessage(role="user", content="hello")]
        chunks = []
        async for chunk in provider.stream(messages):
            chunks.append(chunk.content)
        full_text = "".join(chunks)
        assert full_text == "Hello, I am a mock response."

    @pytest.mark.asyncio
    async def test_stream_finish_reason(self, provider):
        """Last chunk should have finish_reason='stop'."""
        messages = [LLMMessage(role="user", content="hello")]
        last_chunk = None
        async for chunk in provider.stream(messages):
            last_chunk = chunk
        assert last_chunk is not None
        assert last_chunk.finish_reason == "stop"

    @pytest.mark.asyncio
    async def test_stats_updated_on_stream(self, streaming_provider):
        """Stats should be updated after streaming."""
        messages = [LLMMessage(role="user", content="hello")]
        await streaming_provider.generate(messages)
        assert streaming_provider.stats.total_requests == 1
        assert streaming_provider.stats.total_latency_ms > 0
