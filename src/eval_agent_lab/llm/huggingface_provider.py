"""HuggingFace Transformers LLM provider for local/open-source models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from eval_agent_lab.config import CacheConfig, LLMConfig
from eval_agent_lab.llm import BaseLLMProvider, LLMMessage, LLMResponse


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Transformers provider for local inference."""

    def __init__(self, config: LLMConfig, cache_config: Optional[CacheConfig] = None):
        super().__init__(config, cache_config)
        self._pipeline: Optional[Any] = None

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                from transformers import pipeline
                self._pipeline = pipeline(
                    "text-generation",
                    model=self.config.model,
                    max_new_tokens=self.config.max_tokens,
                    temperature=max(self.config.temperature, 0.01),
                )
            except ImportError as exc:
                raise ImportError("Install with: pip install eval-agent-lab[huggingface]") from exc
        return self._pipeline

    async def _call(self, messages: List[LLMMessage],
                    tools: Optional[List[Dict[str, Any]]] = None,
                    **kwargs: Any) -> LLMResponse:
        pipe = self._get_pipeline()
        prompt = "\n".join(f"{m.role}: {m.content}" for m in messages)

        outputs = pipe(prompt, max_new_tokens=self.config.max_tokens,
                       do_sample=self.config.temperature > 0,
                       temperature=max(self.config.temperature, 0.01))

        generated = outputs[0]["generated_text"]
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        # Estimate token counts
        prompt_tokens = len(prompt.split()) * 4 // 3
        completion_tokens = len(generated.split()) * 4 // 3

        return LLMResponse(
            content=generated,
            model=self.config.model,
            usage={"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
            finish_reason="stop",
        )
