"""OpenAI-compatible chat/tool-call transport with strict response parsing."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Sequence

import httpx

from analysis.llm import LLMConfig
from analysis.llm.base import LLMProviderError, StructuredOutputError, TransientLLMError

from .models import ProviderToolCall, ProviderTurn, ToolDefinition


class ToolCallingProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str: ...

    @property
    @abstractmethod
    def model(self) -> str: ...

    @abstractmethod
    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[ToolDefinition],
    ) -> ProviderTurn: ...


class OpenAICompatibleToolCallingProvider(ToolCallingProvider):
    def __init__(self, config: LLMConfig, *, transport: httpx.AsyncBaseTransport | None = None):
        config.validate_for_remote_provider()
        self.config = config
        self.transport = transport

    @property
    def provider_name(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/chat/completions") else f"{base}/chat/completions"

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[ToolDefinition],
    ) -> ProviderTurn:
        key = self.config.api_key
        if key is None:
            raise LLMProviderError("API key is not configured")
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.config.temperature,
            "messages": list(messages),
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ]
            payload["tool_choice"] = "auto"
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout, transport=self.transport) as client:
                response = await client.post(
                    self._endpoint(),
                    headers={
                        "Authorization": f"Bearer {key.get_secret_value()}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TransientLLMError("Tool-calling request timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                raise TransientLLMError(f"Tool-calling provider returned retryable HTTP {status}") from exc
            raise LLMProviderError(f"Tool-calling provider returned HTTP {status}") from exc
        except httpx.RequestError as exc:
            raise TransientLLMError("Tool-calling provider network request failed") from exc
        try:
            message = response.json()["choices"][0]["message"]
            content = message.get("content")
            if content is not None and not isinstance(content, str):
                raise TypeError("assistant content is not a string")
            parsed_calls = []
            for raw in message.get("tool_calls") or []:
                function = raw["function"]
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise TypeError("tool arguments are not an object")
                parsed_calls.append(ProviderToolCall(
                    id=raw["id"], name=function["name"], arguments=arguments
                ))
            assistant_message: dict[str, Any] = {"role": "assistant"}
            if content is not None:
                assistant_message["content"] = content
            if message.get("tool_calls"):
                assistant_message["tool_calls"] = message["tool_calls"]
            return ProviderTurn(
                content=content,
                tool_calls=parsed_calls,
                assistant_message=assistant_message,
            )
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise StructuredOutputError("Provider returned invalid tool-calling output") from exc
