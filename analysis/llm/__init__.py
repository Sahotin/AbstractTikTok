"""LLM provider abstractions for structured analysis."""

from .base import LLMProvider, LLMProviderError, StructuredOutputError, TransientLLMError
from .config import LLMConfig
from .fake import FakeLLMProvider
from .openai_compatible import OpenAICompatibleProvider


def create_provider(config: LLMConfig) -> LLMProvider:
    if config.provider.lower() in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleProvider(config)
    raise ValueError(f"Unsupported LLM provider: {config.provider}")


__all__ = [
    "FakeLLMProvider",
    "LLMConfig",
    "LLMProvider",
    "LLMProviderError",
    "OpenAICompatibleProvider",
    "StructuredOutputError",
    "TransientLLMError",
    "create_provider",
]
