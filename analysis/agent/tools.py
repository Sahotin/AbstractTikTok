"""Read-only, schema-validated tools exposed to the public-opinion agent."""

from __future__ import annotations

import asyncio
from typing import Any, Literal, Optional, Type
from uuid import UUID

from pydantic import Field

from analysis.domain import Platform, PublicOpinionQuery, SemanticSearchQuery, TrendQuery
from analysis.domain.models import DomainModel
from analysis.preprocessing import prepare_analysis_text
from analysis.repositories import AnalysisRepository, SemanticRepository
from analysis.services import (
    AnalysisQueryService,
    PublicOpinionService,
    SemanticSearchService,
    TrendAnalysisService,
    present_topic,
)

from .models import AgentRequest, ToolDefinition


class _VideoArgs(DomainModel):
    content_id: Optional[UUID] = None


class _CommentsArgs(DomainModel):
    content_id: Optional[UUID] = None
    limit: int = Field(default=20, ge=1, le=50)


class _ScopeArgs(DomainModel):
    platform: Optional[Platform] = Field(
        default=None, description="Social platform scope."
    )
    run_id: Optional[UUID] = Field(
        default=None,
        description="CollectionRun UUID only; never pass a SemanticAnalysisRun UUID here.",
    )
    content_id: Optional[UUID] = Field(
        default=None, description="NormalizedContent internal UUID."
    )


class _DistributionArgs(_ScopeArgs):
    top_k: int = Field(default=10, ge=1, le=20)


class _TopicsArgs(DomainModel):
    semantic_run_id: Optional[UUID] = Field(
        default=None, description="Completed SemanticAnalysisRun UUID."
    )
    limit: int = Field(default=10, ge=1, le=20)


class _TrendArgs(_ScopeArgs):
    bucket: Literal["hour", "day"] = "day"
    min_bucket_comments: int = Field(default=3, ge=1, le=1000)


class _SearchArgs(DomainModel):
    semantic_run_id: Optional[UUID] = Field(
        default=None, description="Completed SemanticAnalysisRun UUID."
    )
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=10)
    min_similarity: float = Field(default=0.2, ge=-1.0, le=1.0)


class _HighRiskArgs(_ScopeArgs):
    limit: int = Field(default=10, ge=1, le=20)


class AgentToolRegistry:
    """Tool boundary with request-scope enforcement and compact return values."""

    _ARGUMENTS: dict[str, tuple[str, Type[DomainModel]]] = {
        "get_video_info": ("Get normalized video/content metadata by internal content UUID.", _VideoArgs),
        "get_comments": ("Get a bounded list of comments for one video.", _CommentsArgs),
        "get_comment_statistics": ("Get comment counts, reply counts, and time range. run_id means a collection run, never a semantic run.", _ScopeArgs),
        "get_sentiment_distribution": ("Get analyzed sentiment, emotion, stance, risk, topics, and coverage aggregates.", _DistributionArgs),
        "get_topics": ("Get semantic topic clusters from one completed semantic run.", _TopicsArgs),
        "get_trend": ("Get time-bucket sentiment/risk trend and detected inflection points.", _TrendArgs),
        "search_similar_comments": ("Search semantically similar comments in one embedding run.", _SearchArgs),
        "get_high_risk_comments": ("Get a bounded list of comments whose latest analysis has high risk.", _HighRiskArgs),
    }

    def __init__(
        self,
        analysis_repository: AnalysisRepository,
        semantic_repository: SemanticRepository,
        search_service: SemanticSearchService,
        request_scope: AgentRequest,
    ):
        self.analysis_repository = analysis_repository
        self.semantic_repository = semantic_repository
        self.query_service = AnalysisQueryService(analysis_repository)
        self.opinion_service = PublicOpinionService(analysis_repository)
        self.trend_service = TrendAnalysisService(analysis_repository)
        self.search_service = search_service
        self.scope = request_scope

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=name,
                description=description,
                parameters=model.model_json_schema(),
            )
            for name, (description, model) in self._ARGUMENTS.items()
        ]

    async def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        definition = self._ARGUMENTS.get(name)
        if definition is None:
            raise ValueError(f"Unknown agent tool: {name}")
        args = definition[1].model_validate(arguments)
        handler = getattr(self, f"_{name}")
        return await handler(args)

    @staticmethod
    def _scoped(request_value, argument_value, label: str):
        if request_value is not None and argument_value is not None and request_value != argument_value:
            raise ValueError(f"Tool {label} is outside the agent request scope")
        return request_value if request_value is not None else argument_value

    def _scope_values(self, args: _ScopeArgs) -> dict[str, Any]:
        if (
            args.run_id is not None
            and self.scope.semantic_run_id is not None
            and args.run_id == self.scope.semantic_run_id
        ):
            raise ValueError(
                "run_id expects a CollectionRun UUID; the supplied value is the SemanticAnalysisRun UUID"
            )
        return {
            "platform": self._scoped(self.scope.platform, args.platform, "platform"),
            "run_id": self._scoped(self.scope.run_id, args.run_id, "run_id"),
            "content_id": self._scoped(self.scope.content_id, args.content_id, "content_id"),
            "semantic_run_id": self.scope.semantic_run_id,
            "eligible_comments": self.scope.eligible_comments,
            "analysis_snapshot": self.scope.analysis_snapshot,
        }

    async def _get_video_info(self, args: _VideoArgs) -> Any:
        content_id = self._scoped(self.scope.content_id, args.content_id, "content_id")
        if content_id is None:
            raise ValueError("content_id is required for get_video_info")
        content = await self.query_service.get_content(content_id)
        if content is None:
            raise KeyError(f"Content not found: {content_id}")
        return content.model_dump(mode="json", exclude={"raw_payload"})

    async def _get_comments(self, args: _CommentsArgs) -> Any:
        content_id = self._scoped(self.scope.content_id, args.content_id, "content_id")
        if content_id is None:
            raise ValueError("content_id is required for get_comments")
        page = await self.query_service.list_comments(content_id=content_id, limit=args.limit)
        return {
            "total": page.total,
            "items": [
                item.model_dump(mode="json", include={
                    "id", "content_id", "text", "published_at", "like_count",
                    "reply_count", "depth", "ip_location",
                })
                for item in page.items
            ],
        }

    async def _get_comment_statistics(self, args: _ScopeArgs) -> Any:
        values = self._scope_values(args)
        result = await self.query_service.get_comment_statistics(**{
            key: values[key] for key in ("platform", "run_id", "content_id")
        })
        return vars(result)

    async def _get_sentiment_distribution(self, args: _DistributionArgs) -> Any:
        values = self._scope_values(args)
        result = await self.opinion_service.get_public_opinion(
            PublicOpinionQuery(**values, top_k=args.top_k)
        )
        return result.model_dump(mode="json", exclude={"comment_volume_by_date"})

    async def _get_topics(self, args: _TopicsArgs) -> Any:
        run_id = self._scoped(self.scope.semantic_run_id, args.semantic_run_id, "semantic_run_id")
        if run_id is None:
            raise ValueError("semantic_run_id is required for get_topics")
        page = await self.semantic_repository.list_topic_clusters(
            semantic_run_id=str(run_id), limit=args.limit
        )
        representatives = await asyncio.gather(*(
            self.semantic_repository.get_topic_representatives(str(topic.id), limit=3)
            for topic in page.items
        ))
        distributions = await asyncio.gather(*(
            self.semantic_repository.get_topic_analysis_distributions(str(topic.id))
            for topic in page.items
        ))
        return {
            "total": page.total,
            "items": [
                {
                    **topic.model_dump(mode="json", exclude={"centroid"}),
                    "topic_name": present_topic(topic, topic_representatives).name,
                    "topic_summary": present_topic(topic, topic_representatives).summary,
                    "sentiment_distribution": topic_distributions["sentiment"],
                    "risk_distribution": topic_distributions["risk"],
                    "representative_comments": [
                        {
                            "comment_id": str(item.comment_id),
                            "text": prepare_analysis_text(item.text),
                            "like_count": item.like_count,
                            "similarity": item.similarity,
                            "sentiment": item.sentiment,
                            "risk_level": item.risk_level,
                        }
                        for item in topic_representatives
                    ],
                }
                for topic, topic_representatives, topic_distributions in zip(page.items, representatives, distributions)
            ],
        }

    async def _get_trend(self, args: _TrendArgs) -> Any:
        values = self._scope_values(args)
        result = await self.trend_service.get_trend(TrendQuery(
            **values, bucket=args.bucket, min_bucket_comments=args.min_bucket_comments
        ))
        payload = result.model_dump(mode="json")
        payload["point_count"] = len(payload["points"])
        payload["points"] = payload["points"][-48:]
        return payload

    async def _search_similar_comments(self, args: _SearchArgs) -> Any:
        run_id = self._scoped(self.scope.semantic_run_id, args.semantic_run_id, "semantic_run_id")
        if run_id is None:
            raise ValueError("semantic_run_id is required for search_similar_comments")
        result = await self.search_service.search(SemanticSearchQuery(
            semantic_run_id=run_id,
            query=args.query,
            top_k=args.top_k,
            min_similarity=args.min_similarity,
        ))
        return result.model_dump(mode="json")

    async def _get_high_risk_comments(self, args: _HighRiskArgs) -> Any:
        values = self._scope_values(args)
        rows = await self.analysis_repository.list_high_risk_comments(
            platform=values["platform"].value if isinstance(values["platform"], Platform) else values["platform"],
            run_id=str(values["run_id"]) if values["run_id"] else None,
            content_id=str(values["content_id"]) if values["content_id"] else None,
            semantic_run_id=str(values["semantic_run_id"]) if values["semantic_run_id"] else None,
            limit=args.limit,
        )
        return {
            "count": len(rows),
            "items": [
                {
                    "comment_id": str(row.comment.id),
                    "content_id": str(row.comment.content_id),
                    "text": prepare_analysis_text(row.comment.text),
                    "like_count": row.comment.like_count,
                    "published_at": row.comment.published_at,
                    "sentiment": row.sentiment,
                    "sentiment_score": row.sentiment_score,
                    "risk_level": row.risk_level,
                    "risk_reasons": row.risk_reasons,
                    "summary": row.summary,
                }
                for row in rows
            ],
        }
