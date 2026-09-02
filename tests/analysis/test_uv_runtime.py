from __future__ import annotations

from api.runtime_env import PROJECT_ROOT, build_uv_environment


def test_uv_environment_defaults_to_project_local_runtime(monkeypatch) -> None:
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    monkeypatch.delenv("UV_PYTHON_INSTALL_DIR", raising=False)

    environment = build_uv_environment()

    assert environment["UV_CACHE_DIR"] == str(PROJECT_ROOT / ".uv-cache")
    assert environment["UV_PYTHON_INSTALL_DIR"] == str(PROJECT_ROOT / ".uv-python")
    assert environment["PYTHONUNBUFFERED"] == "1"


def test_uv_environment_overrides_inaccessible_parent_paths(monkeypatch) -> None:
    monkeypatch.setenv("UV_CACHE_DIR", "D:/custom-cache")
    monkeypatch.setenv("UV_PYTHON_INSTALL_DIR", "D:/custom-python")

    environment = build_uv_environment()

    assert environment["UV_CACHE_DIR"] == str(PROJECT_ROOT / ".uv-cache")
    assert environment["UV_PYTHON_INSTALL_DIR"] == str(PROJECT_ROOT / ".uv-python")
