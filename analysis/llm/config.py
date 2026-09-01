"""Environment-backed LLM settings with secret-safe representations."""

from __future__ import annotations

import os
from typing import Literal, Optional

from pydantic import BaseModel, Field, SecretStr


class LLMConfig(BaseModel):
    provider: str = "openai-compatible"
    model: str = ""
    base_url: str = ""
    api_key: Optional[SecretStr] = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    timeout: float = Field(default=30.0, gt=0.0, le=300.0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_base_delay: float = Field(default=0.25, ge=0.0, le=30.0)
    structured_output_mode: Literal["json_schema", "json_object"] = "json_schema"

    @classmethod
    def from_environment(cls) -> "LLMConfig":
        """Read configuration without ever logging or persisting the key."""

        api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("LLM_BASE_URL", "")
        structured_output_mode = os.getenv("LLM_STRUCTURED_OUTPUT_MODE")
        if structured_output_mode is None:
            structured_output_mode = "json_object" if "api.deepseek.com" in base_url.lower() else "json_schema"
        return cls(
            provider=os.getenv("LLM_PROVIDER", "openai-compatible"),
            model=os.getenv("LLM_MODEL", ""),
            base_url=base_url,
            api_key=SecretStr(api_key) if api_key else None,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            timeout=float(os.getenv("LLM_TIMEOUT", "30")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
            retry_base_delay=float(os.getenv("LLM_RETRY_BASE_DELAY", "0.25")),
            structured_output_mode=structured_output_mode,
        )

    def validate_for_remote_provider(self) -> None:
        if not self.model.strip():
            raise ValueError("LLM_MODEL is required")
        if not self.base_url.strip():
            raise ValueError("LLM_BASE_URL is required")
        if self.api_key is None or not self.api_key.get_secret_value():
            raise ValueError("LLM_API_KEY or OPENAI_API_KEY is required")
        if "api.deepseek.com" in self.base_url.lower() and self.structured_output_mode != "json_object":
            raise ValueError("DeepSeek Chat Completions requires LLM_STRUCTURED_OUTPUT_MODE=json_object")
