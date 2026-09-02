"""Deterministic no-cost provider used by tests and local pipeline checks."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from typing import Any, Optional, Sequence, Type

from pydantic import BaseModel

from .base import LLMProvider, StructuredModel, TransientLLMError


DEFAULT_FAKE_OUTPUT = {
    "sentiment": "positive",
    "sentiment_score": 0.8,
    "emotion": "joy",
    "topics": ["编程"],
    "stance": "support",
    "risk_level": "low",
    "risk_reasons": [],
    "keywords": ["编程", "学习"],
    "summary": "评论者认可该内容。",
}


class FakeLLMProvider(LLMProvider):
    def __init__(
        self,
        output: Optional[dict[str, Any] | BaseModel] = None,
        *,
        model: str = "fake-model-v1",
        provider_name: str = "fake",
        failures_before_success: int = 0,
        failure: Optional[Exception] = None,
        delays: Optional[Sequence[float]] = None,
        failure_call_indices: Optional[set[int]] = None,
    ):
        self.output = deepcopy(DEFAULT_FAKE_OUTPUT) if output is None else output
        self._model = model
        self._provider_name = provider_name
        self.failures_before_success = failures_before_success
        self.failure = failure or TransientLLMError("simulated transient failure")
        self.delays = list(delays or [])
        self.failure_call_indices = set(failure_call_indices or set())
        self.call_count = 0
        self.active_count = 0
        self.max_observed_concurrency = 0

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[StructuredModel],
    ) -> StructuredModel:
        del prompt
        call_index = self.call_count
        self.call_count += 1
        self.active_count += 1
        self.max_observed_concurrency = max(self.max_observed_concurrency, self.active_count)
        try:
            if call_index < len(self.delays) and self.delays[call_index] > 0:
                await asyncio.sleep(self.delays[call_index])
            if call_index < self.failures_before_success or call_index in self.failure_call_indices:
                raise self.failure
            source = self.output.model_dump(mode="python") if isinstance(self.output, BaseModel) else deepcopy(self.output)
            return response_model.model_validate(source)
        finally:
            self.active_count -= 1
