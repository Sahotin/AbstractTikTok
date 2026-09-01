"""JSON and streaming JSONL readers with record-type discovery."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Sequence

import aiofiles


RecordKind = Literal["content", "comment", "author"]


@dataclass(frozen=True)
class InputSource:
    path: Path
    kind: RecordKind


async def iter_records(path: Path) -> AsyncIterator[dict[str, Any]]:
    """Yield records from JSONL line-by-line or from a JSON array.

    JSONL is fully streaming. Existing MediaCrawler JSON output is one JSON
    array, so the standard library must materialize that array; the CLI prints
    this limitation and JSONL remains the recommended large-file format.
    """

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        async with aiofiles.open(path, "r", encoding="utf-8-sig") as file:
            line_number = 0
            async for line in file:
                line_number += 1
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                yield record
        return

    if suffix == ".json":
        async with aiofiles.open(path, "r", encoding="utf-8-sig") as file:
            payload = json.loads(await file.read())
        if isinstance(payload, dict):
            yield payload
            return
        if not isinstance(payload, list):
            raise ValueError(f"Expected a JSON object or array in {path}")
        for index, record in enumerate(payload, start=1):
            if not isinstance(record, dict):
                raise ValueError(f"Expected an object at {path} record {index}")
            yield record
        return

    raise ValueError(f"Unsupported input format: {path.suffix}. Expected .json or .jsonl")


def _kind_from_name(path: Path) -> RecordKind | None:
    name = path.name.lower()
    if "comment" in name:
        return "comment"
    if "content" in name or "aweme" in name or "video" in name:
        return "content"
    if "creator" in name or "author" in name:
        return "author"
    return None


def _kind_from_record(record: dict[str, Any]) -> RecordKind:
    if "comment_id" in record:
        return "comment"
    if "aweme_id" in record:
        return "content"
    if "user_id" in record:
        return "author"
    raise ValueError("Cannot infer record type; expected comment_id, aweme_id, or user_id")


async def discover_input_sources(inputs: Sequence[Path | str]) -> list[InputSource]:
    """Expand files/directories, validate formats, and infer record types."""

    paths: set[Path] = set()
    for raw_input in inputs:
        path = Path(raw_input).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input does not exist: {path}")
        if path.is_dir():
            paths.update(candidate for candidate in path.rglob("*") if candidate.suffix.lower() in {".json", ".jsonl"})
        elif path.suffix.lower() in {".json", ".jsonl"}:
            paths.add(path)
        else:
            raise ValueError(f"Unsupported input format: {path}")

    if not paths:
        raise ValueError("No JSON or JSONL input files were found")

    sources: list[InputSource] = []
    for path in sorted(paths):
        kind = _kind_from_name(path)
        if kind is None:
            first_record = None
            async for record in iter_records(path):
                first_record = record
                break
            if first_record is None:
                raise ValueError(f"Input file is empty: {path}")
            kind = _kind_from_record(first_record)
        sources.append(InputSource(path=path, kind=kind))

    order = {"author": 0, "content": 1, "comment": 2}
    return sorted(sources, key=lambda source: (order[source.kind], str(source.path)))

