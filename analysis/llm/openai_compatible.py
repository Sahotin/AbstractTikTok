"""Minimal OpenAI-compatible structured-output provider using httpx."""

from __future__ import annotations

import json
import time
from typing import Any, Type

import httpx
from pydantic import ValidationError

from .base import (
    LLMProvider,
    LLMProviderError,
    StructuredModel,
    StructuredOutputError,
    TransientLLMError,
)
from .config import LLMConfig


class OpenAICompatibleProvider(LLMProvider):
    def __init__(
        self,
        config: LLMConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        config.validate_for_remote_provider()
        self.config = config
        self.transport = transport
        self._request_count = 0
        self._prompt_tokens = 0
        self._prompt_cache_hit_tokens = 0
        self._prompt_cache_miss_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0
        self._latencies_ms: list[float] = []

    @property
    def provider_name(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    def _endpoint(self) -> str:
        base_url = self.config.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return f"{base_url}/chat/completions"

    def usage_snapshot(self) -> dict[str, int | float | None]:
        """Return aggregate non-secret provider telemetry for one process."""

        ordered = sorted(self._latencies_ms)
        return {
            "request_count": self._request_count,
            "prompt_tokens": self._prompt_tokens,
            "prompt_cache_hit_tokens": self._prompt_cache_hit_tokens,
            "prompt_cache_miss_tokens": self._prompt_cache_miss_tokens,
            "completion_tokens": self._completion_tokens,
            "total_tokens": self._total_tokens,
            "latency_avg_ms": round(sum(ordered) / len(ordered), 2) if ordered else None,
            "latency_p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 2) if ordered else None,
        }

    async def generate_structured(
        self,
        prompt: str,
        response_model: Type[StructuredModel],
    ) -> StructuredModel:
        api_key = self.config.api_key
        if api_key is None:
            raise LLMProviderError("API key is not configured")
        request_prompt = prompt
        if self.config.structured_output_mode == "json_object":
            schema = json.dumps(
                response_model.model_json_schema(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            request_prompt = (
                f"{prompt}\n\n必须返回符合以下 JSON Schema 的 JSON 对象：\n{schema}"
            )
            response_format: dict[str, Any] = {"type": "json_object"}
        else:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": response_model.__name__,
                    "strict": True,
                    "schema": response_model.model_json_schema(),
                },
            }
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.config.temperature,
            "messages": [{"role": "user", "content": request_prompt}],
            "response_format": response_format,
        }
        started = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout, transport=self.transport) as client:
                response = await client.post(
                    self._endpoint(),
                    headers={
                        "Authorization": f"Bearer {api_key.get_secret_value()}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TransientLLMError("Provider request timed out") from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            if status_code == 429 or status_code >= 500:
                raise TransientLLMError(f"Provider returned retryable HTTP {status_code}") from exc
            raise LLMProviderError(f"Provider returned HTTP {status_code}") from exc
        except httpx.RequestError as exc:
            raise TransientLLMError("Provider network request failed") from exc

        try:
            body = response.json()
            usage = body.get("usage") or {}
            self._request_count += 1
            self._prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self._prompt_cache_hit_tokens += int(usage.get("prompt_cache_hit_tokens") or 0)
            self._prompt_cache_miss_tokens += int(usage.get("prompt_cache_miss_tokens") or 0)
            self._completion_tokens += int(usage.get("completion_tokens") or 0)
            self._total_tokens += int(usage.get("total_tokens") or 0)
            self._latencies_ms.append((time.perf_counter() - started) * 1000)
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str):
                raise TypeError("message content is not a string")
            decoded = json.loads(content)
            return response_model.model_validate(decoded)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError, ValidationError) as exc:
            raise StructuredOutputError("Provider returned invalid structured output") from exc
