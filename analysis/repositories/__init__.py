"""Persistence interfaces and implementations for normalized data."""

from .sqlalchemy import (
    AnalysisRepository,
    AggregateValue,
    AuthorQueryStatistics,
    CommentVolumeAggregate,
    CommentQueryStatistics,
    ContentQueryStatistics,
    ImportDatabaseStats,
    HighRiskCommentAggregate,
    SentimentScoreAggregate,
    TopicDimensionAggregate,
    UpsertResult,
)
from .semantic_sqlalchemy import SemanticRepository

__all__ = [
    "AnalysisRepository",
    "AggregateValue",
    "AuthorQueryStatistics",
    "CommentVolumeAggregate",
    "CommentQueryStatistics",
    "ContentQueryStatistics",
    "ImportDatabaseStats",
    "HighRiskCommentAggregate",
    "SentimentScoreAggregate",
    "TopicDimensionAggregate",
    "UpsertResult",
    "SemanticRepository",
]
