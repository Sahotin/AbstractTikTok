"""Deterministic text normalization used by deduplication and future caches."""

from __future__ import annotations

import hashlib
import html
import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_URL_RE = re.compile(r"(?i)\b(?:https?://|www\.)[^\s<>\]\[\u3000]+")
_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")
_REPLY_PREFIX_RE = re.compile(r"^\s*回复\s*@?[^:：\r\n]{1,64}\s*[:：]\s*")
_MENTION_RE = re.compile(r"@[A-Za-z0-9_\-\u3400-\u9fff]{1,32}(?=[\s,，。:：!！?？]|$)")
_QUERY_GARBAGE_RE = re.compile(
    r"(?i)(?:^|[?&\s])(?:share_source|share_medium|spm_id_from|spmid|timestamp|ts|from_source|source_from)=[^&\s]*"
)
_UUID_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{8}-?[0-9a-f]{4}-?[1-5][0-9a-f]{3}-?[89ab][0-9a-f]{3}-?[0-9a-f]{12}(?![0-9a-f])")
_LONG_ID_RE = re.compile(r"(?<![A-Za-z0-9])\d{7,}(?![A-Za-z0-9])")
_PURE_NUMBER_RE = re.compile(r"^\d+$")
_REPEATED_CHAR_RE = re.compile(r"([^\W\d_])\1{3,}", re.UNICODE)
_PLATFORM_TEMPLATE_RE = re.compile(
    r"(?:(?<=^)|(?<=[\s,，。;；]))(?:展开回复|查看回复|分享链接|复制链接)(?=$|[\s,，。;；])"
)
_EMPTY_PUNCTUATION_RE = re.compile(r"^[\s,，。:：;；|/\\\-_<>()（）\[\]{}]+$")


def normalize_text(text: str) -> str:
    """Normalize Unicode and collapse whitespace without changing meaning."""

    normalized = unicodedata.normalize("NFKC", text or "")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def prepare_analysis_text(text: str) -> str:
    """Create a clean AI input while leaving stored/raw comment text untouched."""

    cleaned = html.unescape(text or "")
    # Some crawler exports contain entities encoded more than once.
    cleaned = html.unescape(cleaned)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = _HTML_TAG_RE.sub(" ", cleaned)
    cleaned = _REPLY_PREFIX_RE.sub("", cleaned)
    cleaned = _URL_RE.sub(" ", cleaned)
    cleaned = _QUERY_GARBAGE_RE.sub(" ", cleaned)
    cleaned = _UUID_RE.sub(" ", cleaned)
    cleaned = _LONG_ID_RE.sub(" ", cleaned)
    cleaned = _MENTION_RE.sub(" ", cleaned)
    cleaned = _PLATFORM_TEMPLATE_RE.sub(" ", cleaned)
    cleaned = _REPEATED_CHAR_RE.sub(lambda match: match.group(1) * 3, cleaned)
    cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip(" ,，。:：;；|/\\-_<> {}")
    if _EMPTY_PUNCTUATION_RE.fullmatch(cleaned) or _PURE_NUMBER_RE.fullmatch(cleaned):
        return ""
    return cleaned


def compute_text_hash(text: str) -> str:
    """Return SHA-256 of normalized text for cache and deduplication keys."""

    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    """Provide a lightweight language hint without introducing a new model."""

    if _CJK_RE.search(text):
        return "zh"
    if _LATIN_RE.search(text):
        return "en"
    return "unknown"
