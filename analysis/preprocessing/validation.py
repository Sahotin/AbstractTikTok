"""Explicit data-quality checks applied after Pydantic validation."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from analysis.domain import NormalizedAuthor, NormalizedComment, NormalizedContent


class DataQualityIssue(BaseModel):
    """One actionable validation warning or error."""

    severity: Literal["warning", "error"]
    entity_type: Literal["content", "comment", "author", "file"]
    message: str
    source: Optional[str] = None
    row_number: Optional[int] = None
    native_id: Optional[str] = None


def validate_content(content: NormalizedContent) -> list[DataQualityIssue]:
    """Validate content fields not fully expressed by Pydantic types."""

    issues: list[DataQualityIssue] = []
    if not content.native_content_id.strip():
        issues.append(DataQualityIssue(severity="error", entity_type="content", message="Content ID is empty"))
    if content.published_at is None:
        issues.append(DataQualityIssue(severity="warning", entity_type="content", message="Publish time is missing", native_id=content.native_content_id))
    if not (content.title or content.body):
        issues.append(DataQualityIssue(severity="warning", entity_type="content", message="Both title and body are empty", native_id=content.native_content_id))
    return issues


def validate_comment(
    comment: NormalizedComment,
    known_content_ids: Optional[set[str]] = None,
) -> list[DataQualityIssue]:
    """Validate comment identity, text, parent metadata, and content linkage."""

    issues: list[DataQualityIssue] = []
    if not comment.native_comment_id.strip():
        issues.append(DataQualityIssue(severity="error", entity_type="comment", message="Comment ID is empty"))
    if not comment.text.strip():
        issues.append(DataQualityIssue(severity="error", entity_type="comment", message="Comment text is empty", native_id=comment.native_comment_id))
    if not comment.native_content_id.strip():
        issues.append(DataQualityIssue(severity="error", entity_type="comment", message="Content ID is empty", native_id=comment.native_comment_id))
    if known_content_ids is not None and comment.native_content_id not in known_content_ids:
        issues.append(DataQualityIssue(severity="error", entity_type="comment", message=f"Referenced content does not exist: {comment.native_content_id}", native_id=comment.native_comment_id))
    if comment.parent_comment_id == comment.native_comment_id:
        issues.append(DataQualityIssue(severity="error", entity_type="comment", message="Comment cannot be its own parent", native_id=comment.native_comment_id))
    if comment.depth == 0 and comment.parent_comment_id:
        issues.append(DataQualityIssue(severity="error", entity_type="comment", message="Root comment unexpectedly has a parent", native_id=comment.native_comment_id))
    if comment.depth > 0 and not comment.parent_comment_id:
        issues.append(DataQualityIssue(severity="error", entity_type="comment", message="Reply comment has no parent", native_id=comment.native_comment_id))
    if comment.published_at is None:
        issues.append(DataQualityIssue(severity="warning", entity_type="comment", message="Publish time is missing", native_id=comment.native_comment_id))
    return issues


def validate_author(author: NormalizedAuthor) -> list[DataQualityIssue]:
    """Validate author identity and useful profile fields."""

    issues: list[DataQualityIssue] = []
    if not author.native_author_id.strip():
        issues.append(DataQualityIssue(severity="error", entity_type="author", message="Author ID is empty"))
    if not author.nickname:
        issues.append(DataQualityIssue(severity="warning", entity_type="author", message="Author nickname is missing", native_id=author.native_author_id))
    return issues

