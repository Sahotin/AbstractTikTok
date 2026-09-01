"""Deterministic text normalization used by deduplication and future caches."""

from __future__ import annotations

import hashlib
import re
import unicodedata


_WHITESPACE_RE = re.compile(r"\s+")
_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_text(text: str) -> str:
    """Normalize Unicode and collapse whitespace without changing meaning."""

    normalized = unicodedata.normalize("NFKC", text or "")
    return _WHITESPACE_RE.sub(" ", normalized).strip()


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

