# --- TRADEMARK NOTICE ---
# Lightcap (EUIPO. Reg. 019172085) — Contact: alpay@lightcap.ai
# Do not remove this notice from source distributions.

"""Utilities for handling offline package installation fallbacks."""

from __future__ import annotations

import base64
import hashlib
import importlib
import os
import re
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import urllib.request
from collections.abc import Iterable, MutableMapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Callable

tomllib: Any
try:  # Python 3.11+
    tomllib = importlib.import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - fallback for older interpreters
    tomllib = importlib.import_module("tomli")

TOMLDecodeError = getattr(tomllib, "TOMLDecodeError", ValueError)

DEFAULT_INDEX_URL = "https://pypi.org/simple"
DEFAULT_WHEEL_DIR = Path.home() / ".cache" / "pip" / "wheels"


ConnectionChecker = Callable[[str, float], bool]
PipRunner = Callable[[Sequence[str]], None]


class InstallMode(str, Enum):
    """Resolution modes returned by :func:`resolve_package_installation_failure`."""

    INSTALLED = "Installed"
    WHEEL_INSTALLED = "WheelInstalled"
    LINKED = "Linked"
    DIRECT_INSTALLED = "DirectInstalled"
    SOURCED = "Sourced"


def _canonical_name(value: str) -> str:
    return value.lower().replace("_", "-")


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


def _normalise_requirement_version(required: int | float | Sequence[int]) -> tuple[int, ...]:
    """Convert ``required`` into a version tuple for lexicographic comparison."""

    if isinstance(required, Sequence) and not isinstance(required, (str, bytes)):
        numbers: list[int] = []
        for item in required:
            if not isinstance(item, int):
                raise TypeError("version components must be integers")
            if item < 0:
                raise ValueError("version components must be non-negative")
            numbers.append(int(item))
        if not numbers:
            raise ValueError("required build dependency version must be positive")
        return tuple(numbers)

    if isinstance(required, (int, float)):
        if required <= 0:
            raise ValueError("required build dependency version must be positive")
        text = f"{required}"
        if "e" in text.lower():
            text = f"{required:.12f}"
        numbers = [int(part) for part in re.findall(r"\d+", text)]
        while len(numbers) > 1 and numbers[-1] == 0:
            numbers.pop()
        return tuple(numbers)

    raise TypeError("unsupported version specification type")


def _extract_version_components(segment: str) -> tuple[int, ...] | None:
    numbers = [int(part) for part in re.findall(r"\d+", segment)]
    if not numbers:
        return None
    return tuple(numbers)


def _meets_version(candidate: tuple[int, ...], required: tuple[int, ...]) -> bool:
    max_len = max(len(candidate), len(required))
    padded_candidate = candidate + (0,) * (max_len - len(candidate))
    padded_required = required + (0,) * (max_len - len(required))
    return padded_candidate >= padded_required


def _find_local_wheel(
    package_name: str, wheel_dir: Path, required_version: tuple[int, ...]
) -> Path | None:
    if not wheel_dir.exists():
        return None

    canonical_name = _canonical_name(package_name)
    for wheel_path in sorted(wheel_dir.rglob("*.whl")):
        parts = wheel_path.name.split("-")
        if not parts:
            continue
        candidate_name = _canonical_name(parts[0])
        if candidate_name != canonical_name or len(parts) < 2:
            continue
        candidate_version = _extract_version_components(parts[1])
        if candidate_version and _meets_version(candidate_version, required_version):
            return wheel_path
    return None


def _find_project_wheels(wheel_dir: Path, project_name: str) -> list[Path]:
    if not wheel_dir.exists():
        return []

    canonical_prefix = f"{_canonical_name(project_name)}-"
    matches: list[Path] = []
    for wheel_path in wheel_dir.rglob("*.whl"):
        canonical_filename = _canonical_name(wheel_path.stem)
        if canonical_filename.startswith(canonical_prefix):
            matches.append(wheel_path)
    return matches


def _select_latest_wheel(wheels: Iterable[Path]) -> Path | None:
    best: Path | None = None
    best_version: tuple[int, ...] = ()
    for wheel in wheels:
        parts = wheel.name.split("-")
        if len(parts) < 2:
            continue
        candidate_version = _extract_version_components(parts[1])
        if candidate_version is None:
            continue
        if best is None or _meets_version(candidate_version, best_version):
            best = wheel
            best_version = candidate_version
    return best


def _atomically_write(path: Path, data: str, *, mode: str = "w", encoding: str = "utf-8") -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, mode, encoding=encoding) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _prepend_pythonpath(env: MutableMapping[str, str], new_path: str) -> None:
    existing = env.get("PYTHONPATH")
    if existing:
        parts = [part for part in existing.split(os.pathsep) if part]
        if new_path in parts:
            parts.remove(new_path)
        env["PYTHONPATH"] = os.pathsep.join([new_path, *parts])
    else:
        env["PYTHONPATH"] = new_path


def _ensure_console_script(script_path: Path) -> None:
    if script_path.exists():
        return
    body = (
        f"#!{sys.executable}\n"
        "import runpy; runpy.run_module('semantic_lexicon.cli', run_name='__main__')\n"
    )
    _atomically_write(script_path, body)
    script_path.chmod(0o755)


def _sha256_b64url(file_path: Path) -> str:
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return base64.urlsafe_b64encode(digest.digest()).rstrip(b"=").decode("ascii")


def _iter_package_files(package_root: Path) -> Iterable[Path]:
    for entry in package_root.rglob("*"):
        if entry.is_file():
            yield entry


def _parse_project_name_version(pyproject_path: Path) -> tuple[str, str]:
    default_name = pyproject_path.parent.name
    default_version = "0"

    try:
        raw_text = pyproject_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return default_name, default_version

    try:
        data = tomllib.loads(raw_text)
    except TOMLDecodeError:
        return default_name, default_version

    project = data.get("project") or {}
    tool = data.get("tool") or {}
    setuptools_cfg = tool.get("setuptools") or {}

    name = project.get("name") or setuptools_cfg.get("name") or default_name
    version = project.get("version") or setuptools_cfg.get("version") or default_version

    return str(name), str(version)


def resolve_package_installation_failure(
    project_path: Path,
    required_build_dep_version: int | float | Sequence[int],
    *,
    index_url: str = DEFAULT_INDEX_URL,
    wheel_dir: Path | None = None,
    connection_checker: ConnectionChecker | None = None,
    pip_runner: PipRunner | None = None,
    env: MutableMapping[str, str] | None = None,
    connection_timeout: float = 3.0,
) -> tuple[InstallMode, bool]:
    """Attempt to install ``project_path`` even without remote index access."""

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

    # Tier 0 — online fast path
    if connection_checker(index_url, connection_timeout):
        pip_runner(["install", "--upgrade", "pip"])
        pip_runner(["install", "-e", str(project_root)])
        return InstallMode.INSTALLED, True

    # Tier 1 — cached build deps + editable (no build isolation)
    local_wheel = _find_local_wheel("setuptools", wheel_dir, required_version)
    if local_wheel is not None:
        pip_runner(["install", str(local_wheel)])
        pip_runner(["install", "--no-build-isolation", "--no-deps", "-e", str(project_root)])
        return InstallMode.INSTALLED, True

    # Tier 1.5 — cached project wheel (no index)
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        project_name, project_version = _parse_project_name_version(pyproject_path)
    else:
        project_name = project_root.name
        project_version = "0"

    cached_project_wheels = _find_project_wheels(wheel_dir, project_name)
    latest_project_wheel = _select_latest_wheel(cached_project_wheels)
    if latest_project_wheel is not None:
        pip_runner(["install", "--no-index", str(latest_project_wheel)])
        return InstallMode.WHEEL_INSTALLED, True

    # Tier 2 — durable editable via .pth (no setuptools required)
    src_path = project_root / "src"
    try:
        site_paths = sysconfig.get_paths()
        site_pure = Path(site_paths["purelib"])
        site_scripts = Path(site_paths["scripts"])
    except KeyError:
        site_pure = Path(sysconfig.get_path("purelib"))
        site_scripts = Path(sysconfig.get_path("scripts"))

    try:
        offline_pth = site_pure / f"{_canonical_name(project_name)}-offline.pth"
        _atomically_write(offline_pth, str(src_path) + "\n")
        _prepend_pythonpath(env, str(src_path))
        console_script = site_scripts / project_name
        _ensure_console_script(console_script)
        return InstallMode.LINKED, True
    except OSError:
        pass

    # Tier 3 — direct PEP 376 install (copy + .dist-info), still offline
    try:
        package_src = src_path / "semantic_lexicon"
        if not package_src.exists():
            raise FileNotFoundError("source package missing")

        package_dst = site_pure / "semantic_lexicon"
        if package_dst.exists():
            shutil.rmtree(package_dst)
        shutil.copytree(package_src, package_dst)

        dist_info = site_pure / f"{_canonical_name(project_name)}-{project_version}.dist-info"
        if dist_info.exists():
            shutil.rmtree(dist_info)
        dist_info.mkdir(parents=True, exist_ok=True)

        metadata = (
            "Metadata-Version: 2.1\n"
            f"Name: {project_name}\n"
            f"Version: {project_version}\n"
            "Summary: Offline install\n"
        )
        _atomically_write(dist_info / "METADATA", metadata)
        _atomically_write(dist_info / "INSTALLER", "offline-resolver\n")

        record_lines: list[str] = []
        for file_path in _iter_package_files(package_dst):
            rel_path = file_path.relative_to(site_pure)
            digest = _sha256_b64url(file_path)
            size = file_path.stat().st_size
            record_lines.append(f"{rel_path},sha256={digest},{size}")
        for info_file in ("METADATA", "INSTALLER"):
            file_path = dist_info / info_file
            rel_path = file_path.relative_to(site_pure)
            digest = _sha256_b64url(file_path)
            size = file_path.stat().st_size
            record_lines.append(f"{rel_path},sha256={digest},{size}")
        record_lines.append(f"{(dist_info / 'RECORD').relative_to(site_pure)},,")
        _atomically_write(dist_info / "RECORD", "\n".join(record_lines) + "\n")

        console_script = site_scripts / project_name
        try:
            _ensure_console_script(console_script)
        except OSError:
            pass

        return InstallMode.DIRECT_INSTALLED, True
    except OSError:
        pass

    # Tier 4 — last resort: source import only (what you already do)
    _prepend_pythonpath(env, str(src_path))
    return InstallMode.SOURCED, True


__all__: Sequence[str] = ["resolve_package_installation_failure", "InstallMode"]
