"""Create and execute embedding-only semantic analysis runs."""

from __future__ import annotations

import hashlib
import json
from typing import Sequence
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from analysis.domain import (
    NormalizedComment,
    Platform,
    SemanticAnalysisRun,
    SemanticRunItem,
    SemanticRunItemStatus,
    SemanticRunStatus,
    SemanticScopeType,
)
from analysis.embeddings import EmbeddingConfig
from analysis.preprocessing import compute_text_hash, normalize_text, utc_now
from analysis.repositories import AnalysisRepository, SemanticRepository

from .embedding_service import EmbeddingReport, EmbeddingService
from .query_service import AnalysisQueryService


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class SemanticRunService:
    def __init__(
        self,
        analysis_repository: AnalysisRepository,
        semantic_repository: SemanticRepository,
        embedding_service: EmbeddingService,
        config: EmbeddingConfig,
    ):
        self.analysis_repository = analysis_repository
        self.semantic_repository = semantic_repository
        self.embedding_service = embedding_service
        self.config = config
        self.query_service = AnalysisQueryService(analysis_repository)

    async def create_run(
        self,
        *,
        platform: Platform | str = Platform.DOUYIN,
        limit: int,
        run_id: UUID | str | None = None,
        content_id: UUID | str | None = None,
    ) -> SemanticAnalysisRun:
        if not 1 <= limit <= 100_000:
            raise ValueError("limit must be between 1 and 100000")
        platform_value = platform if isinstance(platform, Platform) else Platform(platform)
        comments = await self._load_comments(platform_value, limit, run_id, content_id)
        if not comments:
            raise ValueError("No comments matched the semantic run scope")
        scope_type = SemanticScopeType.CONTENT if content_id else SemanticScopeType.RUN if run_id else SemanticScopeType.PLATFORM
        scope = {
            "scope_type": scope_type.value,
            "platform": platform_value.value,
            "run_id": str(run_id) if run_id else None,
            "content_id": str(content_id) if content_id else None,
            "limit": limit,
        }
        snapshot = self._input_snapshot(comments)
        safe_config = self.config.safe_snapshot()
        safe_config.update({
            "provider": self.embedding_service.provider.provider_name,
            "model": self.embedding_service.provider.model,
        })
        algorithm_config: dict[str, object] = {}
        configuration_snapshot = {
            "embedding": safe_config,
            "algorithm": "none",
            "algorithm_version": "phase3b1",
            "algorithm_config": algorithm_config,
        }
        run = SemanticAnalysisRun(
            id=uuid4(),
            status=SemanticRunStatus.PENDING,
            scope_type=scope_type,
            platform=platform_value,
            run_id=UUID(str(run_id)) if run_id else None,
            content_id=UUID(str(content_id)) if content_id else None,
            requested_limit=limit,
            scope_hash=_sha256_json(scope),
            input_snapshot_hash=snapshot,
            embedding_provider=self.embedding_service.provider.provider_name,
            embedding_model=self.embedding_service.provider.model,
            embedding_version=self.config.embedding_version,
            embedding_config=safe_config,
            algorithm_config=algorithm_config,
            config_hash=_sha256_json(configuration_snapshot),
            total_comments=len(comments),
            analyzed_comments=(
                await self.analysis_repository.get_public_opinion_counts(
                    platform=platform_value.value,
                    run_id=str(run_id) if run_id else None,
                    content_id=str(content_id) if content_id else None,
                )
            )[1],
            created_at=utc_now(),
        )
        return await self.semantic_repository.create_semantic_run(run)

    async def execute_run(self, run_id: UUID | str) -> tuple[SemanticAnalysisRun, EmbeddingReport]:
        run = await self.get_run(run_id)
        if run.status in {SemanticRunStatus.COMPLETED, SemanticRunStatus.CANCELLED}:
            raise ValueError(f"Cannot execute a {run.status.value} semantic run")
        try:
            self._validate_runtime(run)
            comments = await self._load_comments(run.platform, run.requested_limit, run.run_id, run.content_id)
            if self._input_snapshot(comments) != run.input_snapshot_hash:
                raise ValueError("Semantic run input snapshot no longer matches normalized comments")
            await self.semantic_repository.update_semantic_run(
                str(run.id), status=SemanticRunStatus.RUNNING, started_at=utc_now(), error_message=None
            )
            report = await self.embedding_service.embed_comments(comments)
            await self._persist_fixed_snapshot(run, comments)
            final_status = SemanticRunStatus.COMPLETED if report.embedded_comments else SemanticRunStatus.FAILED
            stored = await self.semantic_repository.update_semantic_run(
                str(run.id),
                status=final_status,
                embedded_comments=report.embedded_comments,
                failed_comments=report.failed_comments,
                embedding_dimension=report.dimension,
                finished_at=utc_now(),
                error_message=(f"{report.failed_comments} comment embedding(s) failed" if report.failed_comments else None),
            )
            if stored is None:
                raise RuntimeError("Semantic run final status was not persisted")
            return stored, report
        except Exception as exc:
            await self.semantic_repository.update_semantic_run(
                str(run.id),
                status=SemanticRunStatus.FAILED,
                finished_at=utc_now(),
                error_message=f"{type(exc).__name__}: {str(exc)[:500]}",
            )
            raise

    async def _persist_fixed_snapshot(
        self,
        run: SemanticAnalysisRun,
        comments: Sequence[NormalizedComment],
    ) -> None:
        key = {
            "provider": run.embedding_provider,
            "model": run.embedding_model,
            "embedding_version": run.embedding_version,
        }
        records = await self.semantic_repository.get_embeddings_for_comments(
            [str(comment.id) for comment in comments], **key
        )
        hashes = {
            str(comment.id): compute_text_hash(normalize_text(comment.text))
            for comment in comments
        }
        embeddings = {
            str(record.comment_id): record
            for record in records
            if hashes.get(str(record.comment_id)) == record.input_hash
        }
        analyses = await self.semantic_repository.get_latest_analyses_for_comments(
            [str(comment.id) for comment in comments]
        )
        items: list[SemanticRunItem] = []
        analysis_snapshot = []
        for comment in sorted(comments, key=lambda item: str(item.id)):
            comment_key = str(comment.id)
            embedding = embeddings.get(comment_key)
            analysis = analyses.get(comment_key)
            if analysis is not None and analysis.input_hash != hashes[comment_key]:
                analysis = None
            analysis_snapshot.append((comment_key, str(analysis.id) if analysis else None))
            items.append(
                SemanticRunItem(
                    id=uuid5(NAMESPACE_URL, f"semantic-run-item:{run.id}:{comment.id}"),
                    semantic_run_id=run.id,
                    comment_id=comment.id,
                    embedding_id=embedding.id if embedding else None,
                    analysis_id=analysis.id if analysis else None,
                    status=(
                        SemanticRunItemStatus.EMBEDDED
                        if embedding is not None
                        else SemanticRunItemStatus.MISSING_EMBEDDING
                    ),
                    error_message=None if embedding is not None else "missing_embedding",
                )
            )
        await self.semantic_repository.replace_semantic_run_items(
            str(run.id),
            items,
            analysis_snapshot_hash=_sha256_json(analysis_snapshot),
            analyzed_comments=sum(item.analysis_id is not None for item in items),
        )

    async def get_run(self, run_id: UUID | str) -> SemanticAnalysisRun:
        run = await self.semantic_repository.get_semantic_run(str(run_id))
        if run is None:
            raise KeyError(f"Semantic analysis run not found: {run_id}")
        return run

    def _validate_runtime(self, run: SemanticAnalysisRun) -> None:
        expected = (
            self.embedding_service.provider.provider_name,
            self.embedding_service.provider.model,
            self.config.embedding_version,
        )
        actual = (run.embedding_provider, run.embedding_model, run.embedding_version)
        if expected != actual:
            raise ValueError("Current embedding configuration does not match persisted semantic run")

    async def _load_comments(
        self,
        platform: Platform,
        limit: int,
        run_id: UUID | str | None,
        content_id: UUID | str | None,
    ) -> list[NormalizedComment]:
        comments: list[NormalizedComment] = []
        offset = 0
        while len(comments) < limit:
            page = await self.query_service.list_comments(
                platform=platform,
                run_id=run_id,
                content_id=content_id,
                limit=min(1000, limit - len(comments)),
                offset=offset,
            )
            comments.extend(page.items)
            offset += len(page.items)
            if not page.items or offset >= page.total:
                break
        return comments[:limit]

    @staticmethod
    def _input_snapshot(comments: Sequence[NormalizedComment]) -> str:
        rows = [
            (str(comment.id), compute_text_hash(normalize_text(comment.text)))
            for comment in comments
        ]
        return _sha256_json(sorted(rows))
