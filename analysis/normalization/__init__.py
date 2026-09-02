"""Platform-specific adapters into the normalized domain model."""

from .bilibili import BilibiliNormalizer
from .douyin import DouyinNormalizer

__all__ = ["BilibiliNormalizer", "DouyinNormalizer"]
