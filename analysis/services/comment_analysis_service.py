"""Versioned, cached, observable single-comment LLM analysis pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Sequence
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError

from analysis.domain import CommentAnalysis, CommentAnalysisPayload, NormalizedComment
from analysis.llm import LLMConfig, LLMProvider, StructuredOutputError, TransientLLMError
from analysis.preprocessing import compute_text_hash, normalize_text, utc_now
from analysis.repositories import AnalysisRepository
from .cost_estimation import estimate_text_tokens


logger = logging.getLogger("analysis.comment_analysis")
DEFAULT_PROMPT_PATH = Path(__file__).parents[1] / "prompts" / "comment_analysis_v1.txt"
DEFAULT_PROMPT_VERSION = "comment_analysis_v1"
DEFAULT_ANALYSIS_VERSION = "comment_analysis_schema_v1"


class CommentAnalysisError(RuntimeError):
    """Raised after a bounded analysis attempt cannot produce valid output."""


class CommentAnalysisService:
    """Preprocess, cache, call a provider, validate, and persist analyses."""

    def __init__(
        self,
        repository: AnalysisRepository,
        provider: LLMProvider,
        config: LLMConfig,
        *,
        prompt_path: Path | str = DEFAULT_PROMPT_PATH,
        prompt_version: str = DEFAULT_PROMPT_VERSION,
        analysis_version: str = DEFAULT_ANALYSIS_VERSION,
        max_batch_size: int = 50,
    ):
        self.repository = repository
        self.provider = provider
        self.config = config
        self.prompt_path = Path(prompt_path)
        self.prompt_version = prompt_version
        self.analysis_version = analysis_version
        self.max_batch_size = max_batch_size
        self._prompt_template = self.prompt_path.read_text(encoding="utf-8")
        self._cache_key_locks: dict[str, asyncio.Lock] = {}
        self._cache_key_locks_guard = asyncio.Lock()
        if "{{COMMENT_TEXT}}" not in self._prompt_template:
            raise ValueError("Comment analysis prompt is missing {{COMMENT_TEXT}} placeholder")

    def _render_prompt(self, normalized_text: str) -> str:
        encoded_text = json.dumps(normalized_text, ensure_ascii=False)
        return self._prompt_template.replace("{{COMMENT_TEXT}}", encoded_text)

    def estimate_request_input_tokens(self, comment: NormalizedComment) -> int:
        """Estimate the full prompt and structured-output schema sent to the provider."""

        normalized_text = normalize_text(comment.text)
        if not normalized_text:
            return 1
        prompt = self._render_prompt(normalized_text)
        schema = json.dumps(
            CommentAnalysisPayload.model_json_schema(),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return estimate_text_tokens(f"{prompt}\n{schema}")

    def _cache_key(self, comment: NormalizedComment) -> tuple[str, dict[str, str]]:
        normalized_text = normalize_text(comment.text)
        if not normalized_text:
            raise ValueError("Cannot analyze an empty comment")
        cache_key = {
            "comment_id": str(comment.id),
            "input_hash": compute_text_hash(normalized_text),
            "provider": self.provider.provider_name,
            "model": self.provider.model,
            "prompt_version": self.prompt_version,
            "analysis_version": self.analysis_version,
        }
        return normalized_text, cache_key

    @staticmethod
    def _content_cache_key(cache_key: dict[str, str]) -> dict[str, str]:
        """Remove per-comment identity from the reusable content cache key."""

        return {key: value for key, value in cache_key.items() if key != "comment_id"}

    async def get_cached_analysis(self, comment: NormalizedComment) -> CommentAnalysis | None:
        """Expose the exact cache check needed by the batch runner."""

        _normalized_text, cache_key = self._cache_key(comment)
        return await self.repository.get_cached_analysis(**cache_key)

    async def _lock_for_cache_key(self, cache_key: dict[str, str]) -> asyncio.Lock:
        identity = json.dumps(cache_key, ensure_ascii=False, sort_keys=True)
        async with self._cache_key_locks_guard:
            return self._cache_key_locks.setdefault(identity, asyncio.Lock())

    async def analyze_comment(self, comment: NormalizedComment) -> CommentAnalysis:
        started = time.perf_counter()
        normalized_text, cache_key = self._cache_key(comment)
        cached = await self.repository.get_cached_analysis(**cache_key)
        if cached is not None:
            self._log_result(comment, started, cache_kind="exact", success=True)
            return cached

        content_cache_key = self._content_cache_key(cache_key)
        cache_lock = await self._lock_for_cache_key(content_cache_key)
        async with cache_lock:
            cached = await self.repository.get_cached_analysis(**cache_key)
            if cached is not None:
                self._log_result(comment, started, cache_kind="exact", success=True)
                return cached
            reusable = await self.repository.get_cached_analysis_by_content_hash(**content_cache_key)
            if reusable is not None:
                payload = CommentAnalysisPayload.model_validate(
                    reusable.model_dump(
                        mode="python",
                        include=set(CommentAnalysisPayload.model_fields),
                    )
                )
                stored = await self._store_analysis(comment, payload, cache_key)
                self._log_result(comment, started, cache_kind="content", success=True)
                return stored
            prompt = self._render_prompt(normalized_text)
            try:
                payload = await self._generate_with_retry(prompt)
                stored = await self._store_analysis(comment, payload, cache_key)
                self._log_result(comment, started, cache_kind="none", success=True)
                return stored
            except Exception as exc:
                self._log_result(
                    comment,
                    started,
                    cache_kind="none",
                    success=False,
                    error_type=type(exc).__name__,
                )
                raise

    async def _store_analysis(
        self,
        comment: NormalizedComment,
        payload: CommentAnalysisPayload,
        cache_key: dict[str, str],
    ) -> CommentAnalysis:
        identity = json.dumps(cache_key, ensure_ascii=False, sort_keys=True)
        analysis = CommentAnalysis(
            id=uuid5(NAMESPACE_URL, f"mediacrawler:comment-analysis:{identity}"),
            comment_id=comment.id,
            **payload.model_dump(mode="python"),
            model=self.provider.model,
            provider=self.provider.provider_name,
            prompt_version=self.prompt_version,
            analysis_version=self.analysis_version,
            input_hash=cache_key["input_hash"],
            created_at=utc_now(),
        )
        return await self.repository.save_analysis(analysis)

    async def analyze_comments(self, comments: Sequence[NormalizedComment]) -> list[CommentAnalysis]:
        """Sequential bounded orchestration; concurrency belongs to Phase 2B."""

        if len(comments) > self.max_batch_size:
            raise ValueError(f"Phase 2A batch size cannot exceed {self.max_batch_size}")
        results = []
        for comment in comments:
            results.append(await self.analyze_comment(comment))
        return results

    async def _generate_with_retry(self, prompt: str) -> CommentAnalysisPayload:
        retryable_errors = (asyncio.TimeoutError, TransientLLMError, StructuredOutputError, ValidationError)
        for attempt in range(self.config.max_retries + 1):
            try:
                generated = await asyncio.wait_for(
                    self.provider.generate_structured(prompt, CommentAnalysisPayload),
                    timeout=self.config.timeout,
                )
                return CommentAnalysisPayload.model_validate(generated)
            except retryable_errors as exc:
                if attempt >= self.config.max_retries:
                    raise CommentAnalysisError(
                        f"Structured comment analysis failed after {attempt + 1} attempt(s): {type(exc).__name__}"
                    ) from exc
                delay = self.config.retry_base_delay * (2 ** attempt)
                logger.warning(
                    "comment_analysis_retry provider=%s model=%s prompt_version=%s attempt=%s error_type=%s",
                    self.provider.provider_name,
                    self.provider.model,
                    self.prompt_version,
                    attempt + 1,
                    type(exc).__name__,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
        raise AssertionError("unreachable retry state")

    def _log_result(
        self,
        comment: NormalizedComment,
        started: float,
        *,
        cache_kind: str,
        success: bool,
        error_type: str = "none",
    ) -> None:
        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "comment_analysis comment_id=%s provider=%s model=%s prompt_version=%s cache_kind=%s latency_ms=%.2f success=%s error_type=%s",
            comment.id,
            self.provider.provider_name,
            self.provider.model,
            self.prompt_version,
            cache_kind,
            latency_ms,
            success,
            error_type,
        )
