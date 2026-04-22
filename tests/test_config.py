"""Unit tests for configuration management."""

import pytest

from eval_agent_lab.config import AppConfig, LLMConfig, LLMProviderType


@pytest.mark.unit
class TestLLMConfig:
    def test_default_config(self):
        config = LLMConfig()
        assert config.provider == LLMProviderType.OPENAI
        assert config.model == "gpt-4o-mini"
        assert config.temperature == 0.0
        assert config.max_tokens == 4096

    def test_api_key_from_config(self):
        config = LLMConfig(api_key="test-key")
        assert config.get_api_key() == "test-key"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "env-key")
        config = LLMConfig()
        assert config.get_api_key() == "env-key"

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        config = LLMConfig()
        with pytest.raises(ValueError, match="No API key"):
            config.get_api_key()


@pytest.mark.unit
class TestAppConfig:
    def test_default_config(self):
        config = AppConfig()
        assert config.llm.model == "gpt-4o-mini"
        assert config.cache.enabled is True
        assert config.pipeline.max_concurrent == 5

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("EAL_MODEL", "gpt-4-turbo")
        monkeypatch.setenv("EAL_TEMPERATURE", "0.5")
        config = AppConfig.from_env()
        assert config.llm.model == "gpt-4-turbo"
        assert config.llm.temperature == 0.5
