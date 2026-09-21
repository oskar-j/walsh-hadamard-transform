"""The stream helpers: stdin/stdout fallback, and the staged atomic write."""

from __future__ import annotations

import io
import os
import stat
import sys
from pathlib import Path

import pytest

from walsh.image._io import align, open_binary, open_binary_read, open_binary_write


def test_open_binary_read_and_write_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    with open_binary_write(str(path)) as handle:
        handle.write(b"payload")
    with open_binary_read(str(path)) as handle:
        assert handle.read() == b"payload"


def test_none_selects_the_standard_streams(monkeypatch: pytest.MonkeyPatch) -> None:
    """The streams are yielded as-is, and deliberately not closed."""
    stdin, stdout = io.BytesIO(b"in"), io.BytesIO()
    monkeypatch.setattr(sys, "stdin", type("S", (), {"buffer": stdin})())
    monkeypatch.setattr(sys, "stdout", type("S", (), {"buffer": stdout})())

    with open_binary_read(None) as handle:
        assert handle is stdin
    with open_binary_write(None) as handle:
        assert handle is stdout
    assert not stdin.closed
    assert not stdout.closed


@pytest.mark.parametrize(("mode", "expected"), [("rb", b"payload"), ("r", b"payload")])
def test_legacy_open_binary_still_reads(tmp_path: Path, mode: str, expected: bytes) -> None:
    path = tmp_path / "legacy.bin"
    path.write_bytes(b"payload")
    with open_binary(str(path), mode) as handle:
        assert handle.read() == expected


def test_legacy_open_binary_still_writes(tmp_path: Path) -> None:
    path = tmp_path / "legacy.bin"
    with open_binary(str(path), "wb") as handle:
        handle.write(b"written")
    assert path.read_bytes() == b"written"


# -- staging ---------------------------------------------------------------


def _staging_files(directory: Path) -> list[Path]:
    """Every leftover staging file in `directory`."""
    return [p for p in directory.iterdir() if p.name.startswith(".walsh-")]


def test_a_failed_write_leaves_the_previous_file_untouched(tmp_path: Path) -> None:
    """The point of the whole exercise: a partial write must not destroy data."""
    path = tmp_path / "archive.bin"
    path.write_bytes(b"PREVIOUS CONTENTS")

    with pytest.raises(ValueError, match="halfway"), open_binary_write(str(path)) as handle:
        handle.write(b"new bytes, then a failure")
        raise ValueError("halfway through")

    assert path.read_bytes() == b"PREVIOUS CONTENTS"
    assert _staging_files(tmp_path) == []


def test_a_failed_write_creates_nothing_when_there_was_no_file(tmp_path: Path) -> None:
    path = tmp_path / "never.bin"

    with pytest.raises(ValueError), open_binary_write(str(path)) as handle:
        handle.write(b"partial")
        raise ValueError

    assert not path.exists()
    assert _staging_files(tmp_path) == []


def test_an_interrupt_also_cleans_up(tmp_path: Path) -> None:
    """KeyboardInterrupt is a BaseException, so it needs the same handling."""
    path = tmp_path / "interrupted.bin"
    path.write_bytes(b"safe")

    with pytest.raises(KeyboardInterrupt), open_binary_write(str(path)) as handle:
        handle.write(b"doomed")
        raise KeyboardInterrupt

    assert path.read_bytes() == b"safe"
    assert _staging_files(tmp_path) == []


def test_a_successful_write_replaces_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "replaced.bin"
    path.write_bytes(b"old contents, longer than the new ones")

    with open_binary_write(str(path)) as handle:
        handle.write(b"new")

    assert path.read_bytes() == b"new"
    assert _staging_files(tmp_path) == []


def test_the_staging_file_is_gone_before_the_caller_sees_the_result(tmp_path: Path) -> None:
    """Nothing outside this module should ever observe the temporary name."""
    path = tmp_path / "clean.bin"
    with open_binary_write(str(path)) as handle:
        handle.write(b"x")
        assert len(_staging_files(tmp_path)) == 1, "written through a staging file"
    assert _staging_files(tmp_path) == []


def test_an_existing_file_keeps_its_permissions(tmp_path: Path) -> None:
    """Replacing a file must not silently change who can read it."""
    path = tmp_path / "private.bin"
    path.write_bytes(b"old")
    path.chmod(0o604)

    with open_binary_write(str(path)) as handle:
        handle.write(b"new")

    assert stat.S_IMODE(path.stat().st_mode) == 0o604


def test_a_new_file_gets_the_usual_permissions_not_the_temporary_ones(tmp_path: Path) -> None:
    """mkstemp creates 0600; a file the user made with `>` would be 0666 & ~umask."""
    path = tmp_path / "fresh.bin"
    with open_binary_write(str(path)) as handle:
        handle.write(b"x")

    umask = os.umask(0)
    os.umask(umask)
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask


def test_a_symlinked_destination_stays_a_symlink(tmp_path: Path) -> None:
    """Writing through a link is what open() did; replacing the link is not."""
    real = tmp_path / "real.bin"
    real.write_bytes(b"old")
    link = tmp_path / "link.bin"
    link.symlink_to(real)

    with open_binary_write(str(link)) as handle:
        handle.write(b"new")

    assert link.is_symlink()
    assert real.read_bytes() == b"new"


def test_a_destination_that_cannot_be_replaced_is_written_directly(tmp_path: Path) -> None:
    """os.replace onto a character device would fail, so /dev/null is opened as-is."""
    with open_binary_write("/dev/null") as handle:
        handle.write(b"discarded")
    assert _staging_files(tmp_path) == []


def test_writing_into_a_missing_directory_still_raises_oserror(tmp_path: Path) -> None:
    """The staging file cannot be created either, and the error must not change kind."""
    missing = tmp_path / "nope" / "out.bin"
    with pytest.raises(OSError) as exc_info, open_binary_write(str(missing)):
        pass  # pragma: no cover - the context manager raises on entry
    assert exc_info.value.filename == str(missing)
    assert str(missing) in str(exc_info.value).replace("\\\\", "\\")


def test_align_rounds_up() -> None:
    assert align(1, 4) == 4
    assert align(4, 4) == 4
    assert align(5, 4) == 8


@pytest.mark.parametrize(
    ("value", "alignment", "expected"),
    [(1, 4, 4), (4, 4, 4), (5, 4, 8), (12, 4, 12), (13, 4, 16)],
)
def test_align(value: int, alignment: int, expected: int) -> None:
    assert align(value, alignment) == expected
