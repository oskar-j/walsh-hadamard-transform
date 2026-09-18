"""The requirements files must say what pyproject.toml says.

They exist for contributors who set up without uv, and they drifted for four
releases: ``click`` never reached ``requirements.txt`` after 0.1.3 made it a
runtime dependency, and ``pytest-cov`` never reached ``requirements-dev.txt``,
so the documented non-uv setup could not run the coverage gate CI enforces.
Nothing read these files, so nothing noticed. This does.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib", reason="tomllib is stdlib from 3.11; CI runs 3.11+ too")


def _requirements(root: Path, name: str) -> set[str]:
    """Every requirement line in a file, following ``-r`` includes."""
    lines: set[str] = set()
    for raw in (root / name).read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-r "):
            lines |= _requirements(root, line[3:].strip())
        else:
            lines.add(line)
    return lines


def _normalise(spec: str) -> str:
    return re.sub(r"\s+", "", spec).lower()


def test_runtime_dependencies_are_mirrored(root: Path) -> None:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    expected = {_normalise(s) for s in project["dependencies"]}
    assert {_normalise(s) for s in _requirements(root, "requirements.txt")} == expected


def test_demo_extra_is_mirrored(root: Path) -> None:
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    expected = {_normalise(s) for s in project["dependencies"]}
    expected |= {_normalise(s) for s in project["optional-dependencies"]["demo"]}
    assert {_normalise(s) for s in _requirements(root, "requirements-demo.txt")} == expected


def test_dev_group_is_mirrored(root: Path) -> None:
    data = tomllib.loads((root / "pyproject.toml").read_text())
    expected = {_normalise(s) for s in data["project"]["dependencies"]}
    expected |= {_normalise(s) for s in data["project"]["optional-dependencies"]["demo"]}
    expected |= {_normalise(s) for s in data["dependency-groups"]["dev"]}
    assert {_normalise(s) for s in _requirements(root, "requirements-dev.txt")} == expected
