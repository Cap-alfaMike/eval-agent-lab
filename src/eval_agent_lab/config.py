"""Global configuration management using Pydantic settings."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class LLMProviderType(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"


class LLMConfig(BaseModel):
    """Configuration for an LLM provider."""

    provider: LLMProviderType = LLMProviderType.OPENAI
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = Field(default=None, description="API key (reads from env if None)")
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: float = 60.0
    max_retries: int = 3
    stream: bool = False

    def get_api_key(self) -> str:
        """Resolve API key from config or environment."""
        if self.api_key:
            return self.api_key
        env_key = os.environ.get("OPENAI_API_KEY", "")
        if not env_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY env var or pass api_key in config."
            )
        return env_key


class CacheConfig(BaseModel):
    """Configuration for the caching layer."""

    enabled: bool = True
    directory: Path = Path(".cache/eval_agent_lab")
    ttl_seconds: int = 86400  # 24 hours


class ObservabilityConfig(BaseModel):
    """Configuration for logging and observability."""

    log_level: LogLevel = LogLevel.INFO
    log_file: Optional[Path] = None
    enable_trace_logging: bool = True
    enable_cost_tracking: bool = True


class PipelineConfig(BaseModel):
    """Configuration for evaluation pipeline execution."""

    max_concurrent: int = 5
    batch_size: int = 10
    timeout_per_item: float = 120.0
    fail_fast: bool = False
    output_dir: Path = Path("outputs")


class AppConfig(BaseModel):
    """Root application configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create configuration from environment variables."""
        return cls(
            llm=LLMConfig(
                model=os.environ.get("EAL_MODEL", "gpt-4o-mini"),
                temperature=float(os.environ.get("EAL_TEMPERATURE", "0.0")),
                max_tokens=int(os.environ.get("EAL_MAX_TOKENS", "4096")),
            ),
            observability=ObservabilityConfig(
                log_level=LogLevel(os.environ.get("EAL_LOG_LEVEL", "INFO")),
            ),
            pipeline=PipelineConfig(
                max_concurrent=int(os.environ.get("EAL_MAX_CONCURRENT", "5")),
            ),
        )
