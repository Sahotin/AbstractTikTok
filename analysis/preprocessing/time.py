"""Timestamp conversion helpers with a single UTC policy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def parse_datetime(value: Any) -> Optional[datetime]:
    """Parse seconds, milliseconds, or an ISO datetime string as UTC.

    ``None`` and empty strings represent missing data. Invalid non-empty values
    raise ``ValueError`` so ingestion can report them rather than silently
    dropping malformed timestamps.
    """

    if value is None or value == "":
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        numeric_value: Optional[float] = None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric_value = float(value)
        elif isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                numeric_value = float(stripped)
            except ValueError:
                try:
                    parsed = datetime.fromisoformat(stripped.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise ValueError(f"Invalid datetime value: {value!r}") from exc
        else:
            raise ValueError(f"Unsupported datetime value: {value!r}")

        if numeric_value is not None:
            # Current millisecond Unix timestamps are around 1e12. The lower
            # threshold also handles historical/future dates without guessing
            # based on string length.
            if abs(numeric_value) >= 100_000_000_000:
                numeric_value /= 1000.0
            try:
                parsed = datetime.fromtimestamp(numeric_value, tz=timezone.utc)
            except (OverflowError, OSError, ValueError) as exc:
                raise ValueError(f"Invalid Unix timestamp: {value!r}") from exc

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

