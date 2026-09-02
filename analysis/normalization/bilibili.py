"""Pure conversion of MediaCrawler Bilibili records into domain models."""

from __future__ import annotations

from typing import Any, Mapping, Optional
from uuid import NAMESPACE_URL, UUID, uuid5

from analysis.domain import NormalizedAuthor, NormalizedComment, NormalizedContent, Platform
from analysis.preprocessing import compute_text_hash, detect_language, normalize_text, parse_datetime, utc_now


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
    text = str(value).strip().replace(",", "")
    try:
        number = float(text)
        if not number.is_integer():
            raise ValueError
        result = int(number)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer: {value!r}") from exc
    if result < 0:
        raise ValueError(f"{field_name} cannot be negative: {result}")
    return result


class BilibiliNormalizer:
    """Convert persisted Bilibili video, comment, and creator rows."""

    platform = Platform.BILIBILI

    @staticmethod
    def content_id(native_content_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, f"mediacrawler:bilibili:content:{native_content_id}")

    @staticmethod
    def comment_id(native_comment_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, f"mediacrawler:bilibili:comment:{native_comment_id}")

    @staticmethod
    def author_id(native_author_id: str) -> UUID:
        return uuid5(NAMESPACE_URL, f"mediacrawler:bilibili:author:{native_author_id}")

    def normalize_author(self, raw: Mapping[str, Any]) -> Optional[NormalizedAuthor]:
        native_author_id = _optional_str(raw.get("user_id"))
        if native_author_id is None:
            return None

        official = _optional_str(raw.get("is_official"))
        verified = None
        if official is not None:
            try:
                verified = int(float(official)) >= 0
            except ValueError:
                verified = None

        return NormalizedAuthor(
            id=self.author_id(native_author_id),
            platform=self.platform,
            native_author_id=native_author_id,
            nickname=_optional_str(raw.get("nickname")),
            profile_url=f"https://space.bilibili.com/{native_author_id}",
            avatar_url=_optional_str(raw.get("avatar")),
            description=_optional_str(raw.get("sign")),
            gender=_optional_str(raw.get("sex")),
            ip_location=None,
            following_count=None,
            follower_count=_non_negative_int(raw.get("total_fans"), "total_fans"),
            content_count=None,
            verified=verified,
            attributes={
                key: value
                for key, value in {
                    "total_liked": _non_negative_int(raw.get("total_liked"), "total_liked"),
                    "user_rank": _non_negative_int(raw.get("user_rank"), "user_rank"),
                }.items()
                if value is not None
            },
            raw_payload=dict(raw),
            collected_at=parse_datetime(raw.get("last_modify_ts")) or utc_now(),
        )

    def normalize_content(self, raw: Mapping[str, Any], run_id: UUID) -> NormalizedContent:
        native_content_id = _required_str(raw.get("video_id"), "video_id")
        native_author_id = _optional_str(raw.get("user_id"))
        cover = _optional_str(raw.get("video_cover_url"))

        return NormalizedContent(
            id=self.content_id(native_content_id),
            run_id=run_id,
            platform=self.platform,
            native_content_id=native_content_id,
            content_type=_optional_str(raw.get("video_type")) or "video",
            title=_optional_str(raw.get("title")),
            body=_optional_str(raw.get("desc")),
            url=_optional_str(raw.get("video_url")) or f"https://www.bilibili.com/video/av{native_content_id}",
            author_id=self.author_id(native_author_id) if native_author_id else None,
            native_author_id=native_author_id,
            published_at=parse_datetime(raw.get("create_time")),
            like_count=_non_negative_int(raw.get("liked_count"), "liked_count"),
            comment_count=_non_negative_int(raw.get("video_comment"), "video_comment"),
            share_count=_non_negative_int(raw.get("video_share_count"), "video_share_count"),
            favorite_count=_non_negative_int(raw.get("video_favorite_count"), "video_favorite_count"),
            view_count=_non_negative_int(raw.get("video_play_count"), "video_play_count"),
            source_keyword=_optional_str(raw.get("source_keyword")),
            media_urls=[cover] if cover else [],
            tags=[],
            raw_payload=dict(raw),
            collected_at=parse_datetime(raw.get("last_modify_ts")) or utc_now(),
        )

    def normalize_comment(self, raw: Mapping[str, Any], run_id: UUID) -> NormalizedComment:
        native_comment_id = _required_str(raw.get("comment_id"), "comment_id")
        native_content_id = _required_str(raw.get("video_id"), "video_id")
        native_author_id = _optional_str(raw.get("user_id"))
        raw_parent_id = _optional_str(raw.get("parent_comment_id"))
        parent_comment_id = None if raw_parent_id in (None, "0") else raw_parent_id
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
            depth=1 if parent_comment_id else 0,
            text=text,
            published_at=parse_datetime(raw.get("create_time")),
            like_count=_non_negative_int(raw.get("like_count"), "like_count"),
            reply_count=_non_negative_int(raw.get("sub_comment_count"), "sub_comment_count"),
            ip_location=None,
            media_urls=[],
            text_hash=compute_text_hash(text),
            language=detect_language(text),
            raw_payload=dict(raw),
            collected_at=parse_datetime(raw.get("last_modify_ts")) or utc_now(),
        )
