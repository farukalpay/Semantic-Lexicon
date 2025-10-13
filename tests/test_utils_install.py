# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

from __future__ import annotations

import os
from pathlib import Path

import pytest

from semantic_lexicon.utils import resolve_package_installation_failure


def _offline(_: str, __: float) -> bool:
    return False


def _online(_: str, __: float) -> bool:
    return True


def test_resolver_prefers_online_install(tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    commands: list[list[str]] = []

    def runner(args: list[str]) -> None:
        commands.append(list(args))

    env: dict[str, str] = {}
    success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_online,
        pip_runner=runner,
        env=env,
    )
    assert success is True
    assert commands == [
        ["install", "--upgrade", "pip"],
        ["install", "-e", str(tmp_path)],
    ]
    assert "PYTHONPATH" not in env


def test_resolver_uses_cached_wheel_when_offline(tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel_file = wheel_dir / "setuptools-68.2.0-py3-none-any.whl"
    wheel_file.write_text("cache")
    commands: list[list[str]] = []

    def runner(args: list[str]) -> None:
        commands.append(list(args))

    success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=runner,
        wheel_dir=wheel_dir,
        env={},
    )
    assert success is True
    assert commands == [
        ["install", str(wheel_file)],
        ["install", "--no-build-isolation", "--no-deps", "-e", str(tmp_path)],
    ]


def test_resolver_sets_pythonpath_as_last_resort(tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    env = {"PYTHONPATH": "/existing"}
    commands: list[list[str]] = []

    def runner(args: list[str]) -> None:
        commands.append(list(args))

    success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=runner,
        wheel_dir=tmp_path / "missing-wheels",
        env=env,
    )
    assert success is False
    assert not commands
    expected_prefix = str(project_src)
    pythonpath = env["PYTHONPATH"]
    assert pythonpath.split(os.pathsep)[0] == expected_prefix
    assert pythonpath.endswith("/existing")


def test_resolver_requires_sufficient_wheel_version(tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    (wheel_dir / "setuptools-60.0.0-py3-none-any.whl").write_text("cache")
    env: dict[str, str] = {}

    success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=lambda args: None,
        wheel_dir=wheel_dir,
        env=env,
    )
    assert success is False
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(project_src)


def test_resolver_validates_required_version(tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()

    with pytest.raises(ValueError):
        resolve_package_installation_failure(
            tmp_path,
            required_build_dep_version=0,
            connection_checker=_offline,
            pip_runner=lambda args: None,
            wheel_dir=tmp_path,
            env={},
        )
