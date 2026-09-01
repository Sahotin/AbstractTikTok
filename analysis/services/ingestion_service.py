"""Orchestrate file reading, normalization, validation, and persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Callable, Optional, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, Field, ValidationError

from analysis.domain import CollectionRun, CollectionRunStatus, Platform
from analysis.io import InputSource, discover_input_sources, iter_records
from analysis.normalization import DouyinNormalizer
from analysis.preprocessing import (
    DataQualityIssue,
    utc_now,
    validate_author,
    validate_comment,
    validate_content,
)
from analysis.repositories import AnalysisRepository


ProgressCallback = Callable[[str], None]


class IngestionReport(BaseModel):
    """Machine-readable and CLI-friendly Phase 1A acceptance report."""

    run_id: UUID
    database_path: str
    input_files: list[str]
    records_read: int = 0
    contents_read: int = 0
    comments_read: int = 0
    authors_read: int = 0
    contents_inserted: int = 0
    content_duplicates: int = 0
    comments_inserted: int = 0
    comment_duplicates: int = 0
    authors_inserted: int = 0
    author_duplicates: int = 0
    invalid_count: int = 0
    warning_count: int = 0
    issues: list[DataQualityIssue] = Field(default_factory=list)
    database_content_count: int = 0
    database_comment_count: int = 0
    database_author_count: int = 0
    first_level_comment_count: int = 0
    second_level_comment_count: int = 0
    video_count: int = 0
    earliest_comment_at: Optional[str] = None
    latest_comment_at: Optional[str] = None


class IngestionService:
    """Import Douyin crawler output into normalized SQLite tables."""

    def __init__(
        self,
        repository: AnalysisRepository,
        batch_size: int = 500,
        progress: Optional[ProgressCallback] = None,
        issue_sample_limit: int = 100,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.repository = repository
        self.batch_size = batch_size
        self.progress = progress or (lambda _message: None)
        self.issue_sample_limit = issue_sample_limit
        self.normalizer = DouyinNormalizer()

    async def ingest(
        self,
        inputs: Sequence[Path | str],
        platform: str = "douyin",
        crawler_type: str = "import",
        query: Optional[str] = None,
        specified_ids: Optional[list[str]] = None,
    ) -> IngestionReport:
        """Import JSON/JSONL sources in deterministic, idempotent batches."""

        if platform not in {"douyin", "dy"}:
            raise ValueError("Phase 1A only supports platform=douyin")

        sources = await discover_input_sources(inputs)
        run_id, fingerprints = await self._stable_run_id(
            sources=sources,
            crawler_type=crawler_type,
            query=query,
            specified_ids=specified_ids or [],
        )
        report = IngestionReport(
            run_id=run_id,
            database_path=str(self.repository.database_path),
            input_files=[str(source.path) for source in sources],
        )
        started_at = utc_now()
        run = CollectionRun(
            id=run_id,
            platform=Platform.DOUYIN,
            crawler_type=crawler_type,
            query=query,
            specified_ids=specified_ids or [],
            status=CollectionRunStatus.PROCESSING,
            started_at=started_at,
            config_snapshot={
                "batch_size": self.batch_size,
                "input_fingerprints": fingerprints,
                "json_note": "JSON arrays are materialized; JSONL is streamed line-by-line.",
            },
            output_location=[str(source.path) for source in sources],
        )

        await self.repository.initialize()
        await self.repository.upsert_collection_run(run)
        self.progress(f"Run {run_id} started with {len(sources)} input file(s)")
        if any(source.path.suffix.lower() == ".json" for source in sources):
            self.progress("JSON arrays are loaded as whole files; use JSONL for line-by-line streaming on large datasets")

        try:
            for source in sources:
                self.progress(f"Reading {source.kind}: {source.path}")
                if source.kind == "content":
                    await self._ingest_content_source(source, report, run_id)
                elif source.kind == "comment":
                    await self._ingest_comment_source(source, report, run_id)
                else:
                    await self._ingest_author_source(source, report)

            stats = await self.repository.stats_for_run(str(run_id))
            self._apply_database_stats(report, stats)
            completed_run = run.model_copy(
                update={
                    "status": CollectionRunStatus.COMPLETED,
                    "finished_at": utc_now(),
                    "content_count": stats.content_count,
                    "comment_count": stats.comment_count,
                }
            )
            await self.repository.upsert_collection_run(completed_run)
            self.progress(f"Run {run_id} completed")
            return report
        except Exception as exc:
            failed_run = run.model_copy(
                update={
                    "status": CollectionRunStatus.FAILED,
                    "finished_at": utc_now(),
                    "error_message": str(exc),
                }
            )
            await self.repository.upsert_collection_run(failed_run)
            raise

    async def _ingest_content_source(self, source: InputSource, report: IngestionReport, run_id: UUID) -> None:
        batch: list[tuple[int, dict]] = []
        row_number = 0
        async for raw in iter_records(source.path):
            row_number += 1
            report.records_read += 1
            report.contents_read += 1
            batch.append((row_number, raw))
            if len(batch) >= self.batch_size:
                await self._flush_content_batch(source, batch, report, run_id)
                batch = []
                self.progress(f"Processed {row_number} content rows from {source.path.name}")
        if batch:
            await self._flush_content_batch(source, batch, report, run_id)

    async def _flush_content_batch(
        self,
        source: InputSource,
        batch: list[tuple[int, dict]],
        report: IngestionReport,
        run_id: UUID,
    ) -> None:
        contents = []
        authors = []
        for row_number, raw in batch:
            try:
                content = self.normalizer.normalize_content(raw, run_id)
                issues = validate_content(content)
                if self._record_issues(issues, source, row_number, report):
                    continue
                author = self.normalizer.normalize_author(raw)
                if author:
                    author_issues = validate_author(author)
                    if not self._record_issues(author_issues, source, row_number, report):
                        authors.append(author)
                contents.append(content)
            except (ValueError, TypeError, ValidationError) as exc:
                self._record_invalid("content", str(exc), source, row_number, report, raw.get("aweme_id"))

        author_result = await self.repository.upsert_authors(authors)
        content_result = await self.repository.upsert_contents(contents)
        report.authors_inserted += author_result.inserted
        report.author_duplicates += author_result.duplicates
        report.contents_inserted += content_result.inserted
        report.content_duplicates += content_result.duplicates

    async def _ingest_comment_source(self, source: InputSource, report: IngestionReport, run_id: UUID) -> None:
        batch: list[tuple[int, dict]] = []
        row_number = 0
        async for raw in iter_records(source.path):
            row_number += 1
            report.records_read += 1
            report.comments_read += 1
            batch.append((row_number, raw))
            if len(batch) >= self.batch_size:
                await self._flush_comment_batch(source, batch, report, run_id)
                batch = []
                self.progress(f"Processed {row_number} comment rows from {source.path.name}")
        if batch:
            await self._flush_comment_batch(source, batch, report, run_id)

    async def _flush_comment_batch(
        self,
        source: InputSource,
        batch: list[tuple[int, dict]],
        report: IngestionReport,
        run_id: UUID,
    ) -> None:
        candidates = []
        for row_number, raw in batch:
            try:
                comment = self.normalizer.normalize_comment(raw, run_id)
                author = self.normalizer.normalize_author(raw)
                candidates.append((row_number, raw, comment, author))
            except (ValueError, TypeError, ValidationError) as exc:
                self._record_invalid("comment", str(exc), source, row_number, report, raw.get("comment_id"))

        existing_content_ids = await self.repository.existing_content_ids(
            "douyin", (candidate[2].native_content_id for candidate in candidates)
        )
        comments = []
        authors = []
        for row_number, _raw, comment, author in candidates:
            issues = validate_comment(comment, existing_content_ids)
            if self._record_issues(issues, source, row_number, report):
                continue
            if author:
                author_issues = validate_author(author)
                if not self._record_issues(author_issues, source, row_number, report):
                    authors.append(author)
            comments.append(comment)

        author_result = await self.repository.upsert_authors(authors)
        comment_result = await self.repository.upsert_comments(comments)
        report.authors_inserted += author_result.inserted
        report.author_duplicates += author_result.duplicates
        report.comments_inserted += comment_result.inserted
        report.comment_duplicates += comment_result.duplicates

    async def _ingest_author_source(self, source: InputSource, report: IngestionReport) -> None:
        batch = []
        row_number = 0
        async for raw in iter_records(source.path):
            row_number += 1
            report.records_read += 1
            report.authors_read += 1
            try:
                author = self.normalizer.normalize_author(raw)
                if author is None:
                    self._record_invalid("author", "user_id is required", source, row_number, report, None)
                    continue
                if self._record_issues(validate_author(author), source, row_number, report):
                    continue
                batch.append(author)
            except (ValueError, TypeError, ValidationError) as exc:
                self._record_invalid("author", str(exc), source, row_number, report, raw.get("user_id"))
            if len(batch) >= self.batch_size:
                result = await self.repository.upsert_authors(batch)
                report.authors_inserted += result.inserted
                report.author_duplicates += result.duplicates
                batch = []
        if batch:
            result = await self.repository.upsert_authors(batch)
            report.authors_inserted += result.inserted
            report.author_duplicates += result.duplicates

    def _record_issues(
        self,
        issues: list[DataQualityIssue],
        source: InputSource,
        row_number: int,
        report: IngestionReport,
    ) -> bool:
        has_error = False
        for issue in issues:
            enriched = issue.model_copy(update={"source": str(source.path), "row_number": row_number})
            if issue.severity == "error":
                has_error = True
            else:
                report.warning_count += 1
            if len(report.issues) < self.issue_sample_limit:
                report.issues.append(enriched)
        if has_error:
            report.invalid_count += 1
        return has_error

    def _record_invalid(
        self,
        entity_type: str,
        message: str,
        source: InputSource,
        row_number: int,
        report: IngestionReport,
        native_id: object,
    ) -> None:
        report.invalid_count += 1
        issue = DataQualityIssue(
            severity="error",
            entity_type=entity_type,
            message=message,
            source=str(source.path),
            row_number=row_number,
            native_id=str(native_id) if native_id is not None else None,
        )
        if len(report.issues) < self.issue_sample_limit:
            report.issues.append(issue)

    @staticmethod
    def _apply_database_stats(report: IngestionReport, stats) -> None:
        report.database_content_count = stats.total_database_contents
        report.database_comment_count = stats.total_database_comments
        report.database_author_count = stats.total_database_authors
        report.first_level_comment_count = stats.first_level_comment_count
        report.second_level_comment_count = stats.second_level_comment_count
        report.video_count = stats.video_count
        report.earliest_comment_at = stats.earliest_comment_at.isoformat() if stats.earliest_comment_at else None
        report.latest_comment_at = stats.latest_comment_at.isoformat() if stats.latest_comment_at else None

    async def _stable_run_id(
        self,
        sources: Sequence[InputSource],
        crawler_type: str,
        query: Optional[str],
        specified_ids: list[str],
    ) -> tuple[UUID, list[dict[str, str]]]:
        fingerprints = []
        for source in sources:
            digest = await asyncio.to_thread(self._hash_file, source.path)
            fingerprints.append({
                "path": str(source.path),
                "kind": source.kind,
                "sha256": digest,
            })
        identity = json.dumps(
            {
                "platform": "douyin",
                "crawler_type": crawler_type,
                "query": query,
                "specified_ids": sorted(specified_ids),
                "inputs": fingerprints,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return uuid5(NAMESPACE_URL, identity), fingerprints

    @staticmethod
    def _hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
