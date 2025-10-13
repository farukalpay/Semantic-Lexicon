# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Utilities for handling offline package installation fallbacks."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import urllib.request
from collections.abc import Iterable, MutableMapping, Sequence
from pathlib import Path
from typing import Callable

DEFAULT_INDEX_URL = "https://pypi.org/simple"
DEFAULT_WHEEL_DIR = Path.home() / ".cache" / "pip" / "wheels"


ConnectionChecker = Callable[[str, float], bool]
PipRunner = Callable[[Sequence[str]], None]


def _default_connection_checker(url: str, timeout: float) -> bool:
    """Return ``True`` when ``url`` is reachable within ``timeout`` seconds."""

    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            status = getattr(response, "status", None)
            if status is None:
                status = response.getcode()
    except (OSError, urllib.error.URLError, ValueError, socket.timeout):
        return False
    return status is None or 200 <= status < 400


def _default_pip_runner(args: Sequence[str]) -> None:
    """Execute ``pip`` with ``args`` using the current Python interpreter."""

    command = [sys.executable, "-m", "pip", *args]
    subprocess.run(command, check=True)


def _normalise_requirement_version(required: float) -> tuple[int, ...]:
    """Convert ``required`` into a version tuple for lexicographic comparison."""

    if required <= 0:
        raise ValueError("required build dependency version must be positive")
    text = f"{required}"
    if "e" in text.lower():
        text = f"{required:.12f}"
    numbers = [int(part) for part in re.findall(r"\d+", text)]
    while len(numbers) > 1 and numbers[-1] == 0:
        numbers.pop()
    return tuple(numbers)


def _extract_version_components(filename: str) -> tuple[int, ...] | None:
    """Extract a numeric version tuple from a wheel ``filename``."""

    numbers = [int(part) for part in re.findall(r"\d+", filename)]
    if not numbers:
        return None
    return tuple(numbers)


def _meets_version(candidate: tuple[int, ...], required: tuple[int, ...]) -> bool:
    """Return ``True`` when ``candidate`` is greater than or equal to ``required``."""

    max_len = max(len(candidate), len(required))
    padded_candidate = candidate + (0,) * (max_len - len(candidate))
    padded_required = required + (0,) * (max_len - len(required))
    return padded_candidate >= padded_required


def _find_local_wheel(
    package_name: str, wheel_dir: Path, required_version: tuple[int, ...]
) -> Path | None:
    """Locate a cached wheel for ``package_name`` that satisfies ``required_version``."""

    if not wheel_dir.exists():
        return None

    canonical_name = package_name.lower().replace("_", "-")
    for wheel_path in sorted(wheel_dir.glob("*.whl")):
        filename = wheel_path.name
        name_without_suffix = wheel_path.stem
        parts = name_without_suffix.split("-")
        if not parts:
            continue
        candidate_name = parts[0].lower().replace("_", "-")
        if candidate_name != canonical_name:
            continue
        candidate_version = _extract_version_components(filename)
        if candidate_version and _meets_version(candidate_version, required_version):
            return wheel_path
    return None


def _augment_pythonpath(env: MutableMapping[str, str], new_path: str) -> None:
    """Ensure ``new_path`` appears at the front of ``PYTHONPATH`` inside ``env``."""

    existing = env.get("PYTHONPATH")
    if existing:
        paths = [part for part in existing.split(os.pathsep) if part]
        if new_path not in paths:
            env["PYTHONPATH"] = os.pathsep.join([new_path, *paths])
    else:
        env["PYTHONPATH"] = new_path


def resolve_package_installation_failure(
    project_path: Path,
    required_build_dep_version: float,
    *,
    index_url: str = DEFAULT_INDEX_URL,
    wheel_dir: Path | None = None,
    connection_checker: ConnectionChecker | None = None,
    pip_runner: PipRunner | None = None,
    env: MutableMapping[str, str] | None = None,
    connection_timeout: float = 3.0,
) -> bool:
    """Attempt to install ``project_path`` even without remote index access.

    The strategy follows these steps:

    1. Try a normal editable installation when the package index is reachable.
    2. If offline, fall back to a cached ``setuptools`` wheel when it satisfies
       ``required_build_dep_version`` and install without build isolation.
    3. As a last resort, expose the project sources by updating ``PYTHONPATH``.

    Parameters
    ----------
    project_path:
        Path to the project root that should be installed.
    required_build_dep_version:
        Minimal acceptable version for build-time dependencies such as
        ``setuptools``.
    index_url:
        Package index URL used to probe connectivity.
    wheel_dir:
        Directory containing cached wheels. Defaults to the standard pip cache.
    connection_checker:
        Optional override for the connectivity test; primarily useful for tests.
    pip_runner:
        Function invoked to execute ``pip`` commands. Defaults to launching the
        interpreter's ``pip`` module.
    env:
        Environment mapping updated when falling back to direct source imports.
    connection_timeout:
        Timeout in seconds applied to the connectivity probe.
    """

    project_root = Path(project_path)
    if wheel_dir is None:
        wheel_dir = DEFAULT_WHEEL_DIR
    if connection_checker is None:
        connection_checker = _default_connection_checker
    if pip_runner is None:
        pip_runner = _default_pip_runner
    if env is None:
        env = os.environ

    required_version = _normalise_requirement_version(required_build_dep_version)

    if connection_checker(index_url, connection_timeout):
        pip_runner(["install", "--upgrade", "pip"])
        pip_runner(["install", "-e", str(project_root)])
        return True

    local_wheel = _find_local_wheel("setuptools", wheel_dir, required_version)
    if local_wheel is not None:
        pip_runner(["install", str(local_wheel)])
        pip_runner(["install", "--no-build-isolation", "--no-deps", "-e", str(project_root)])
        return True

    project_src = str(project_root / "src")
    _augment_pythonpath(env, project_src)
    return False


__all__: Sequence[str] = ["resolve_package_installation_failure"]
