"""Pure conversion of MediaCrawler Douyin records into domain models."""

from __future__ import annotations

from typing import Any, Mapping, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from analysis.domain import NormalizedAuthor, NormalizedComment, NormalizedContent, Platform
from analysis.preprocessing import (
    compute_text_hash,
    detect_language,
    normalize_text,
    parse_datetime,
    utc_now,
)


def _optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _required_str(value: Any, field_name: str) -> str:
    result = _optional_str(value)
    if result is None:
        raise ValueError(f"{field_name} is required")
    return result


def _non_negative_int(value: Any, field_name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer: {value!r}") from exc
    if result < 0:
        raise ValueError(f"{field_name} cannot be negative: {result}")
    return result


def _split_urls(*values: Any) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not value:
            continue
        candidates = value if isinstance(value, list) else str(value).split(",")
        for candidate in candidates:
            url = str(candidate).strip()
            if url and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


class DouyinNormalizer:
    """Convert already-persisted Douyin crawler output without database access."""

    platform = Platform.DOUYIN

    @staticmethod
    def content_id(native_content_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, f"mediacrawler:douyin:content:{native_content_id}")

    @staticmethod
    def comment_id(native_comment_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, f"mediacrawler:douyin:comment:{native_comment_id}")

    @staticmethod
    def author_id(native_author_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, f"mediacrawler:douyin:author:{native_author_id}")

    def normalize_author(self, raw: Mapping[str, Any]) -> Optional[NormalizedAuthor]:
        """Extract the author projection embedded in a content/comment record."""

        native_author_id = _optional_str(raw.get("user_id"))
        if native_author_id is None:
            return None

        sec_uid = _optional_str(raw.get("sec_uid"))
        attributes = {
            key: value
            for key, value in {
                "sec_uid": sec_uid,
                "short_user_id": _optional_str(raw.get("short_user_id")),
                "user_unique_id": _optional_str(raw.get("user_unique_id")),
            }.items()
            if value is not None
        }
        raw_author = {
            key: raw.get(key)
            for key in (
                "user_id",
                "sec_uid",
                "short_user_id",
                "user_unique_id",
                "user_signature",
                "nickname",
                "avatar",
                "ip_location",
            )
            if key in raw
        }
        collected_at = parse_datetime(raw.get("last_modify_ts")) or utc_now()

        return NormalizedAuthor(
            id=self.author_id(native_author_id),
            platform=self.platform,
            native_author_id=native_author_id,
            nickname=_optional_str(raw.get("nickname")),
            profile_url=f"https://www.douyin.com/user/{sec_uid}" if sec_uid else None,
            avatar_url=_optional_str(raw.get("avatar")),
            description=_optional_str(raw.get("user_signature")),
            gender=_optional_str(raw.get("gender")),
            ip_location=_optional_str(raw.get("ip_location")),
            following_count=_non_negative_int(raw.get("follows"), "follows"),
            follower_count=_non_negative_int(raw.get("fans"), "fans"),
            content_count=_non_negative_int(raw.get("videos_count"), "videos_count"),
            verified=None,
            attributes=attributes,
            raw_payload=raw_author,
            collected_at=collected_at,
        )

    def normalize_content(self, raw: Mapping[str, Any], run_id: UUID) -> NormalizedContent:
        """Convert one MediaCrawler Douyin content record."""

        native_content_id = _required_str(raw.get("aweme_id"), "aweme_id")
        native_author_id = _optional_str(raw.get("user_id"))
        video_url = _optional_str(raw.get("video_download_url"))
        image_urls = _optional_str(raw.get("note_download_url"))
        raw_aweme_type = _optional_str(raw.get("aweme_type")) or "unknown"
        if video_url:
            content_type = "video"
        elif image_urls:
            content_type = "image"
        else:
            content_type = f"aweme_{raw_aweme_type}"

        return NormalizedContent(
            id=self.content_id(native_content_id),
            run_id=run_id,
            platform=self.platform,
            native_content_id=native_content_id,
            content_type=content_type,
            title=_optional_str(raw.get("title")),
            body=_optional_str(raw.get("desc")),
            url=_optional_str(raw.get("aweme_url")),
            author_id=self.author_id(native_author_id) if native_author_id else None,
            native_author_id=native_author_id,
            published_at=parse_datetime(raw.get("create_time")),
            like_count=_non_negative_int(raw.get("liked_count"), "liked_count"),
            comment_count=_non_negative_int(raw.get("comment_count"), "comment_count"),
            share_count=_non_negative_int(raw.get("share_count"), "share_count"),
            favorite_count=_non_negative_int(raw.get("collected_count"), "collected_count"),
            view_count=None,
            source_keyword=_optional_str(raw.get("source_keyword")),
            media_urls=_split_urls(
                raw.get("cover_url"),
                raw.get("video_download_url"),
                raw.get("music_download_url"),
                raw.get("note_download_url"),
            ),
            tags=[],
            raw_payload=dict(raw),
            collected_at=parse_datetime(raw.get("last_modify_ts")) or utc_now(),
        )

    def normalize_comment(self, raw: Mapping[str, Any], run_id: UUID) -> NormalizedComment:
        """Convert one MediaCrawler Douyin comment or reply record."""

        native_comment_id = _required_str(raw.get("comment_id"), "comment_id")
        native_content_id = _required_str(raw.get("aweme_id"), "aweme_id")
        native_author_id = _optional_str(raw.get("user_id"))
        raw_parent_id = _optional_str(raw.get("parent_comment_id"))
        parent_comment_id = None if raw_parent_id in (None, "0") else raw_parent_id
        # Douyin's persisted reply_id points to the root first-level comment.
        # The audited real dataset contains only depth 0 and depth 1 records.
        depth = 1 if parent_comment_id else 0
        text = normalize_text(str(raw.get("content") or ""))

        return NormalizedComment(
            id=self.comment_id(native_comment_id),
            run_id=run_id,
            platform=self.platform,
            native_comment_id=native_comment_id,
            content_id=self.content_id(native_content_id),
            native_content_id=native_content_id,
            author_id=self.author_id(native_author_id) if native_author_id else None,
            native_author_id=native_author_id,
            parent_comment_id=parent_comment_id,
            root_comment_id=parent_comment_id,
            depth=depth,
            text=text,
            published_at=parse_datetime(raw.get("create_time")),
            like_count=_non_negative_int(raw.get("like_count"), "like_count"),
            reply_count=_non_negative_int(raw.get("sub_comment_count"), "sub_comment_count"),
            ip_location=_optional_str(raw.get("ip_location")),
            media_urls=_split_urls(raw.get("pictures")),
            text_hash=compute_text_hash(text),
            language=detect_language(text),
            raw_payload=dict(raw),
            collected_at=parse_datetime(raw.get("last_modify_ts")) or utc_now(),
        )

