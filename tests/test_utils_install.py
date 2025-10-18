import os
import sysconfig
from pathlib import Path

import pytest

from semantic_lexicon.utils import InstallMode, resolve_package_installation_failure


def _write_pyproject(root: Path, *, name: str = "semantic-lexicon", version: str = "0.0.1") -> None:
    (root / "pyproject.toml").write_text(
        """
[project]
name = "{name}"
version = "{version}"
""".strip().format(name=name, version=version)
    )


def _offline(_: str, __: float) -> bool:
    return False


def _online(_: str, __: float) -> bool:
    return True


def _configure_site_paths(monkeypatch: pytest.MonkeyPatch, root: Path) -> tuple[Path, Path]:
    purelib = root / "purelib"
    scripts = root / "scripts"
    purelib.mkdir(parents=True)
    scripts.mkdir(parents=True)

    def fake_get_paths() -> dict[str, str]:
        return {"purelib": str(purelib), "scripts": str(scripts)}

    monkeypatch.setattr(sysconfig, "get_paths", fake_get_paths)
    monkeypatch.setattr(sysconfig, "get_path", lambda key: fake_get_paths()[key])
    return purelib, scripts


def test_resolver_prefers_online_install(tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    _write_pyproject(tmp_path)
    commands: list[list[str]] = []

    def runner(args: list[str]) -> None:
        commands.append(list(args))

    env: dict[str, str] = {}
    mode, success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_online,
        pip_runner=runner,
        env=env,
    )
    assert success is True
    assert mode is InstallMode.INSTALLED
    assert commands == [
        ["install", "--upgrade", "pip"],
        ["install", "-e", str(tmp_path)],
    ]
    assert "PYTHONPATH" not in env


def test_resolver_uses_cached_wheel_when_offline(tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    _write_pyproject(tmp_path)
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    wheel_file = wheel_dir / "setuptools-68.2.0-py3-none-any.whl"
    wheel_file.write_text("cache")
    commands: list[list[str]] = []

    def runner(args: list[str]) -> None:
        commands.append(list(args))

    mode, success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=runner,
        wheel_dir=wheel_dir,
        env={},
    )
    assert success is True
    assert mode is InstallMode.INSTALLED
    assert commands == [
        ["install", str(wheel_file)],
        ["install", "--no-build-isolation", "--no-deps", "-e", str(tmp_path)],
    ]


def test_resolver_uses_cached_project_wheel(tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    _write_pyproject(tmp_path)
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    project_wheel = wheel_dir / "semantic_lexicon-1.2.3-py3-none-any.whl"
    project_wheel.write_text("wheel")
    commands: list[list[str]] = []

    def runner(args: list[str]) -> None:
        commands.append(list(args))

    mode, success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=runner,
        wheel_dir=wheel_dir,
        env={},
    )
    assert success is True
    assert mode is InstallMode.WHEEL_INSTALLED
    assert commands == [["install", "--no-index", str(project_wheel)]]


def test_resolver_links_project_sources(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    _write_pyproject(tmp_path)
    (project_src / "semantic_lexicon").mkdir()
    env = {"PYTHONPATH": "/existing"}
    commands: list[list[str]] = []

    def runner(args: list[str]) -> None:
        commands.append(list(args))

    purelib, scripts = _configure_site_paths(monkeypatch, tmp_path / "site")

    mode, success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=runner,
        wheel_dir=tmp_path / "missing-wheels",
        env=env,
    )
    assert success is True
    assert mode is InstallMode.LINKED
    assert not commands
    expected_prefix = str(project_src)
    pythonpath = env["PYTHONPATH"]
    assert pythonpath.split(os.pathsep)[0] == expected_prefix
    assert pythonpath.endswith("/existing")
    pth_files = list(purelib.glob("*.pth"))
    assert len(pth_files) == 1
    assert pth_files[0].read_text().strip() == expected_prefix
    script = scripts / "semantic-lexicon"
    assert script.exists()


def test_resolver_handles_invalid_pyproject(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_src = tmp_path / "src"
    package = project_src / "semantic_lexicon"
    package.mkdir(parents=True)

    # Invalid TOML content that should fall back to directory defaults
    (tmp_path / "pyproject.toml").write_text("[project\nname = 'broken'", encoding="utf8")

    purelib, scripts = _configure_site_paths(monkeypatch, tmp_path / "site")

    env: dict[str, str] = {}
    mode, success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=lambda args: None,
        wheel_dir=tmp_path / "missing-wheels",
        env=env,
    )

    assert success is True
    assert mode is InstallMode.LINKED
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(project_src)
    pth_files = list(purelib.glob("*.pth"))
    assert len(pth_files) == 1
    assert scripts.joinpath(tmp_path.name).exists()


def test_resolver_direct_installation_when_linking_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    _write_pyproject(tmp_path)
    package = project_src / "semantic_lexicon"
    package.mkdir()
    (package / "__init__.py").write_text("__version__='0.0'")

    purelib, scripts = _configure_site_paths(monkeypatch, tmp_path / "site")

    from semantic_lexicon.utils import install as install_mod

    real_atomically_write = install_mod._atomically_write

    def flaky_atomically_write(
        path: Path,
        data: str,
        *,
        mode: str = "w",
        encoding: str = "utf-8",
    ) -> None:
        if str(path).endswith("offline.pth"):
            raise OSError("nope")
        real_atomically_write(path, data, mode=mode, encoding=encoding)

    monkeypatch.setattr(install_mod, "_atomically_write", flaky_atomically_write)

    mode, success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=lambda args: None,
        wheel_dir=tmp_path / "missing-wheels",
        env={},
    )
    assert success is True
    assert mode is InstallMode.DIRECT_INSTALLED
    installed_package = purelib / "semantic_lexicon" / "__init__.py"
    assert installed_package.exists()
    dist_info = next(purelib.glob("semantic-lexicon-*.dist-info"))
    record = dist_info / "RECORD"
    assert record.exists()
    assert "semantic_lexicon/__init__.py" in record.read_text()
    script = scripts / "semantic-lexicon"
    assert script.exists()


def test_resolver_sources_when_copy_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    _write_pyproject(tmp_path)
    (project_src / "semantic_lexicon").mkdir()

    from semantic_lexicon.utils import install as install_mod

    _configure_site_paths(monkeypatch, tmp_path / "site")

    real_atomically_write = install_mod._atomically_write

    def flaky_atomically_write(
        path: Path,
        data: str,
        *,
        mode: str = "w",
        encoding: str = "utf-8",
    ) -> None:
        if str(path).endswith("offline.pth"):
            raise OSError("nope")
        real_atomically_write(path, data, mode=mode, encoding=encoding)

    monkeypatch.setattr(install_mod, "_atomically_write", flaky_atomically_write)

    def boom(*_args, **_kwargs):
        raise OSError("boom")

    monkeypatch.setattr(install_mod.shutil, "copytree", boom)

    env: dict[str, str] = {}
    mode, success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=lambda args: None,
        wheel_dir=tmp_path / "missing-wheels",
        env=env,
    )
    assert success is True
    assert mode is InstallMode.SOURCED
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(project_src)


def test_resolver_requires_sufficient_wheel_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    _write_pyproject(tmp_path)
    (project_src / "semantic_lexicon").mkdir()
    wheel_dir = tmp_path / "wheels"
    wheel_dir.mkdir()
    (wheel_dir / "setuptools-60.0.0-py3-none-any.whl").write_text("cache")

    purelib, _ = _configure_site_paths(monkeypatch, tmp_path / "site")

    env: dict[str, str] = {}

    mode, success = resolve_package_installation_failure(
        tmp_path,
        required_build_dep_version=68,
        connection_checker=_offline,
        pip_runner=lambda args: None,
        wheel_dir=wheel_dir,
        env=env,
    )
    assert success is True
    assert mode is InstallMode.LINKED
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(project_src)
    pth_files = list(purelib.glob("*.pth"))
    assert len(pth_files) == 1


def test_resolver_validates_required_version(tmp_path: Path) -> None:
    project_src = tmp_path / "src"
    project_src.mkdir()
    _write_pyproject(tmp_path)

    with pytest.raises(ValueError):
        resolve_package_installation_failure(
            tmp_path,
            required_build_dep_version=0,
            connection_checker=_offline,
            pip_runner=lambda args: None,
            wheel_dir=tmp_path,
            env={},
        )
