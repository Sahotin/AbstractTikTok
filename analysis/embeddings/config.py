"""Secret-safe environment configuration for embedding providers."""

from __future__ import annotations

import os
from typing import Optional

from pydantic import BaseModel, Field, SecretStr


class EmbeddingConfig(BaseModel):
    provider: str = "openai-compatible"
    model: str = ""
    base_url: str = ""
    api_key: Optional[SecretStr] = None
    timeout: float = Field(default=30.0, gt=0, le=300)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_base_delay: float = Field(default=0.25, ge=0, le=30)
    batch_size: int = Field(default=100, ge=1, le=500)
    max_concurrency: int = Field(default=3, ge=1, le=20)
    embedding_version: str = Field(default="embedding_v1", min_length=1, max_length=64)
    fake_dimension: int = Field(default=16, ge=2, le=4096)

    @classmethod
    def from_environment(cls) -> "EmbeddingConfig":
        key = os.getenv("EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
        return cls(
            provider=os.getenv("EMBEDDING_PROVIDER", "openai-compatible"),
            model=os.getenv("EMBEDDING_MODEL", ""),
            base_url=os.getenv("EMBEDDING_BASE_URL", ""),
            api_key=SecretStr(key) if key else None,
            timeout=float(os.getenv("EMBEDDING_TIMEOUT", "30")),
            max_retries=int(os.getenv("EMBEDDING_MAX_RETRIES", "2")),
            retry_base_delay=float(os.getenv("EMBEDDING_RETRY_BASE_DELAY", "0.25")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            max_concurrency=int(os.getenv("EMBEDDING_MAX_CONCURRENCY", "3")),
            embedding_version=os.getenv("EMBEDDING_VERSION", "embedding_v1"),
            fake_dimension=int(os.getenv("EMBEDDING_FAKE_DIMENSION", "16")),
        )

    def validate_for_remote_provider(self) -> None:
        if not self.model.strip():
            raise ValueError("EMBEDDING_MODEL is required")
        if not self.base_url.strip():
            raise ValueError("EMBEDDING_BASE_URL is required")
        if self.api_key is None or not self.api_key.get_secret_value():
            raise ValueError("EMBEDDING_API_KEY or OPENAI_API_KEY is required")

    def safe_snapshot(self) -> dict[str, object]:
        return self.model_dump(mode="python", exclude={"api_key"})
