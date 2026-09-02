"""Shared subprocess environment for uv-backed local project commands."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_uv_environment() -> dict[str, str]:
    """Keep project-owned uv subprocesses independent of user-level uv paths.

    The WebUI can be started from shells that already define ``UV_CACHE_DIR``.
    Reusing that setting would send child ``uv`` commands back to a location the
    current user may not be allowed to create or modify.  These subprocesses
    therefore always use the repository-local runtime directories.
    """

    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    environment["UV_CACHE_DIR"] = str(PROJECT_ROOT / ".uv-cache")
    environment["UV_PYTHON_INSTALL_DIR"] = str(PROJECT_ROOT / ".uv-python")
    return environment
