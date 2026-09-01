"""Shared subprocess environment for uv-backed local project commands."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_uv_environment() -> dict[str, str]:
    """Keep uv cache and managed Python inside the project when not overridden."""

    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    environment.setdefault("UV_CACHE_DIR", str(PROJECT_ROOT / ".uv-cache"))
    environment.setdefault("UV_PYTHON_INSTALL_DIR", str(PROJECT_ROOT / ".uv-python"))
    return environment
