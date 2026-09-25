"""examples/roundtrip.py must keep working, and must leave ``data/`` alone (#67)."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

from conftest import Sample


def _load_example(root: Path) -> ModuleType:
    pytest.importorskip("matplotlib")
    pytest.importorskip("PIL")
    path = root / "examples" / "roundtrip.py"
    if not path.exists():  # pragma: no cover - a distribution without examples
        pytest.skip(f"example missing: {path}")
    spec = importlib.util.spec_from_file_location("roundtrip", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _snapshot(folder: Path) -> dict[str, tuple[int, str]]:
    """Every file below ``folder`` with its modification time and digest."""
    return {
        str(path.relative_to(folder)): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def test_the_roundtrip_example_writes_nothing_into_data(
    root: Path,
    sample: Sample,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """It used to write its output over ``data/bmp/recreated.bmp``, the
    reference ``test_golden.py`` compares the codec against. Run after a
    change to the codec, it regenerated that reference from the changed code,
    and the BMP comparisons then passed against the codec's own output. It
    works in a temporary directory now, and ``data/`` is not written to at
    all: not a new file, and not an old one rewritten with the same bytes."""
    source = sample("image.bmp")
    example = _load_example(root)
    assert example.SOURCE.resolve() == source.resolve()

    before = _snapshot(root / "data")
    shown: list[int] = []
    monkeypatch.setattr(example.plt, "show", lambda: shown.append(len(example.plt.get_fignums())))
    try:
        example.main()
    finally:
        example.plt.close("all")

    assert shown == [1], "the example should draw exactly one figure and show it"
    assert "of the original" in capsys.readouterr().out
    assert _snapshot(root / "data") == before
