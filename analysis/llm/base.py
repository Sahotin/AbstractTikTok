"""Provider-neutral interfaces and errors for structured LLM generation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type, TypeVar

from pydantic import BaseModel


StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class LLMProviderError(Exception):
    """Base error for provider failures safe to expose by type only."""


class TransientLLMError(LLMProviderError):
    """A retryable rate-limit, server, or network failure."""


class StructuredOutputError(LLMProviderError):
    """The provider response could not be parsed as the requested schema."""


class LLMProvider(ABC):
    """Generate validated output without leaking HTTP or SDK details."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[StructuredModel],
    ) -> StructuredModel:
        raise NotImplementedError
