"""Create, inspect, cancel, and derive retry jobs for batch analysis."""

from __future__ import annotations

import os
from typing import Optional, Sequence
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from pydantic import BaseModel, Field

from analysis.domain import (
    AnalysisJob,
    AnalysisJobItem,
    AnalysisJobItemStatus,
    AnalysisJobStatus,
    NormalizedComment,
    Platform,
)
from analysis.preprocessing import utc_now
from analysis.repositories import AnalysisRepository

from .comment_analysis_service import CommentAnalysisService
from .cost_estimation import estimate_comment_costs
from .query_service import AnalysisQueryService


class BatchAnalysisConfig(BaseModel):
    batch_size: int = Field(default=20, ge=1, le=500)
    max_concurrency: int = Field(default=5, ge=1, le=20)
    max_requests_per_minute: Optional[int] = Field(default=None, ge=1)
    estimated_output_tokens_per_comment: int = Field(default=150, ge=1, le=4000)
    input_price_per_1m_tokens: Optional[float] = Field(default=None, ge=0)
    output_price_per_1m_tokens: Optional[float] = Field(default=None, ge=0)

    @classmethod
    def from_environment(cls) -> "BatchAnalysisConfig":
        rpm = os.getenv("LLM_MAX_REQUESTS_PER_MINUTE")
        input_price = os.getenv("LLM_INPUT_PRICE_PER_1M_TOKENS")
        output_price = os.getenv("LLM_OUTPUT_PRICE_PER_1M_TOKENS")
        return cls(
            batch_size=int(os.getenv("LLM_BATCH_SIZE", "20")),
            max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "5")),
            max_requests_per_minute=int(rpm) if rpm else None,
            estimated_output_tokens_per_comment=int(os.getenv("LLM_ESTIMATED_OUTPUT_TOKENS", "150")),
            input_price_per_1m_tokens=float(input_price) if input_price else None,
            output_price_per_1m_tokens=float(output_price) if output_price else None,
        )


class AnalysisJobService:
    def __init__(
        self,
        repository: AnalysisRepository,
        query_service: AnalysisQueryService,
        analysis_service: CommentAnalysisService,
    ):
        self.repository = repository
        self.query_service = query_service
        self.analysis_service = analysis_service

    async def create_job(
        self,
        *,
        platform: Platform | str = Platform.DOUYIN,
        limit: int,
        config: BatchAnalysisConfig,
        run_id: UUID | str | None = None,
        content_id: UUID | str | None = None,
        parent_job_id: UUID | str | None = None,
        comment_ids: Optional[Sequence[UUID | str]] = None,
    ) -> AnalysisJob:
        if limit < 1 or limit > 100_000:
            raise ValueError("limit must be between 1 and 100000")
        await self.repository.initialize()
        comments = await self._load_comments(
            platform=platform,
            limit=limit,
            run_id=run_id,
            content_id=content_id,
            comment_ids=comment_ids,
        )
        if not comments:
            raise ValueError("No comments matched the job filters")
        job_id = uuid4()
        estimate = estimate_comment_costs(
            comments,
            output_tokens_per_comment=config.estimated_output_tokens_per_comment,
            input_price_per_1m_tokens=config.input_price_per_1m_tokens,
            output_price_per_1m_tokens=config.output_price_per_1m_tokens,
            input_token_estimates=[
                self.analysis_service.estimate_request_input_tokens(comment)
                for comment in comments
            ],
        )
        platform_value = platform if isinstance(platform, Platform) else Platform(platform)
        job = AnalysisJob(
            id=job_id,
            parent_job_id=UUID(str(parent_job_id)) if parent_job_id else None,
            status=AnalysisJobStatus.PENDING,
            platform=platform_value,
            run_id=UUID(str(run_id)) if run_id else None,
            content_id=UUID(str(content_id)) if content_id else None,
            requested_limit=limit,
            total_comments=len(comments),
            pending_count=len(comments),
            provider=self.analysis_service.provider.provider_name,
            model=self.analysis_service.provider.model,
            prompt_version=self.analysis_service.prompt_version,
            analysis_version=self.analysis_service.analysis_version,
            max_concurrency=config.max_concurrency,
            batch_size=config.batch_size,
            max_requests_per_minute=config.max_requests_per_minute,
            estimated_input_tokens=estimate.input_tokens,
            estimated_output_tokens=estimate.output_tokens,
            input_price_per_1m_tokens=config.input_price_per_1m_tokens,
            output_price_per_1m_tokens=config.output_price_per_1m_tokens,
            estimated_cost=estimate.cost,
            created_at=utc_now(),
        )
        items = [
            AnalysisJobItem(
                id=uuid5(NAMESPACE_URL, f"mediacrawler:analysis-job-item:{job_id}:{comment.id}"),
                job_id=job_id,
                comment_id=comment.id,
                status=AnalysisJobItemStatus.PENDING,
            )
            for comment in comments
        ]
        return await self.repository.create_analysis_job(job, items)

    async def retry_failed_job(
        self,
        job_id: UUID | str,
        *,
        config: Optional[BatchAnalysisConfig] = None,
    ) -> AnalysisJob:
        parent = await self.get_job(job_id)
        failed_ids = await self.repository.failed_job_comment_ids(str(job_id))
        if not failed_ids:
            raise ValueError("Job has no failed comments to retry")
        retry_config = config or BatchAnalysisConfig(
            batch_size=parent.batch_size,
            max_concurrency=parent.max_concurrency,
            max_requests_per_minute=parent.max_requests_per_minute,
            input_price_per_1m_tokens=parent.input_price_per_1m_tokens,
            output_price_per_1m_tokens=parent.output_price_per_1m_tokens,
        )
        return await self.create_job(
            platform=parent.platform,
            limit=len(failed_ids),
            config=retry_config,
            run_id=parent.run_id,
            content_id=parent.content_id,
            parent_job_id=parent.id,
            comment_ids=failed_ids,
        )

    async def get_job(self, job_id: UUID | str) -> AnalysisJob:
        job = await self.repository.get_analysis_job(str(job_id))
        if job is None:
            raise KeyError(f"Analysis job not found: {job_id}")
        return job

    async def cancel_job(self, job_id: UUID | str) -> AnalysisJob:
        job = await self.repository.cancel_analysis_job(str(job_id), utc_now())
        if job is None:
            raise KeyError(f"Analysis job not found: {job_id}")
        return job

    async def _load_comments(
        self,
        *,
        platform: Platform | str,
        limit: int,
        run_id: UUID | str | None,
        content_id: UUID | str | None,
        comment_ids: Optional[Sequence[UUID | str]],
    ) -> list[NormalizedComment]:
        if comment_ids is not None:
            return await self.repository.get_comments_by_ids([str(comment_id) for comment_id in comment_ids[:limit]])
        comments: list[NormalizedComment] = []
        offset = 0
        while len(comments) < limit:
            page_size = min(1000, limit - len(comments))
            page = await self.query_service.list_comments(
                platform=platform,
                run_id=run_id,
                content_id=content_id,
                limit=page_size,
                offset=offset,
            )
            comments.extend(page.items)
            offset += len(page.items)
            if not page.items or offset >= page.total:
                break
        return comments[:limit]
