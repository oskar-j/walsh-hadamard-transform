"""`Task.compress` with a picture as its output: the codec without the `.cim` file.

    Task().compress(input="earth.ppm", output="earth_compressed.ppm").run()

What is written is the picture's lossy reconstruction, and the claim these
tests hold it to is that it is exactly what compressing to a `.cim` and
extracting that would have written. In 0.5.1 the output must be the file type
of the input; anything else is a `NotImplementedError` that names 0.6.0.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from conftest import (
    gradient_pixels,
    write_bmp,
    write_npy,
    write_pam,
    write_png,
    write_ppm,
    write_tiff,
)
from walsh import Task, UnsupportedFileFormatError, reader_for
from walsh.task import CROSS_FORMAT_ISSUE

WIDTH, HEIGHT = 40, 24


def _write_pkl(path: Path, width: int, height: int, pixels: list[tuple[int, int, int]]) -> Path:
    rows = [pixels[y * width : (y + 1) * width] for y in range(height)]
    path.write_bytes(pickle.dumps(rows, protocol=4))
    return path


WRITERS: dict[str, Callable[..., Path]] = {
    ".bmp": write_bmp,
    ".ppm": write_ppm,
    ".pam": write_pam,
    ".tif": write_tiff,
    ".npy": write_npy,
    ".pkl": _write_pkl,
    ".png": write_png,
}


def _source(tmp_path: Path, suffix: str, name: str = "source") -> Path:
    return WRITERS[suffix](tmp_path / f"{name}{suffix}", WIDTH, HEIGHT, _picture())


def _picture() -> list[tuple[int, int, int]]:
    """A gradient with some texture, so the codec's loss is visible in it."""
    rng = np.random.default_rng(41)
    noise = rng.integers(0, 24, size=(WIDTH * HEIGHT, 3))
    return [
        (int(min(255, r + n[0])), int(min(255, g + n[1])), int(min(255, b + n[2])))
        for (r, g, b), n in zip(gradient_pixels(WIDTH, HEIGHT), noise, strict=True)
    ]


def _pixels(path: Path) -> np.ndarray:
    image = reader_for(str(path))
    image.load(str(path))
    return image.get_array()


def _two_steps(task: Callable[[], Task], source: Path, target: Path) -> Path:
    """The route the direct one must match: a .cim on disk in between."""
    container = target.with_suffix(".cim")
    task().compress(input=str(source), output=str(container)).run()
    task().extract(input=str(container), output=str(target)).run()
    return target


# --- the same picture as the two-step route -----------------------------------


@pytest.mark.parametrize("suffix", sorted(WRITERS))
def test_the_direct_output_is_what_compress_then_extract_writes(
    tmp_path: Path, suffix: str
) -> None:
    source = _source(tmp_path, suffix)
    direct = tmp_path / f"direct{suffix}"
    Task().compress(input=str(source), output=str(direct)).run()

    expected = _two_steps(Task, source, tmp_path / f"expected{suffix}")
    assert np.array_equal(_pixels(direct), _pixels(expected))
    if suffix != ".png":  # a PNG is pinned by its pixels; see test_png.py
        assert direct.read_bytes() == expected.read_bytes()


def test_the_output_is_a_lossy_picture_and_not_the_compressed_data(tmp_path: Path) -> None:
    """It is the *result* of the compression: a whole picture of the input's
    size and dimensions, close to the input and not equal to it."""
    source = _source(tmp_path, ".ppm")
    direct = tmp_path / "direct.ppm"
    Task().compress(input=str(source), output=str(direct)).run()

    before, after = _pixels(source).astype(int), _pixels(direct).astype(int)
    assert direct.stat().st_size == source.stat().st_size
    assert before.shape == after.shape
    assert 0 < np.abs(before - after).mean() < 25


def test_nothing_but_the_output_is_written(tmp_path: Path) -> None:
    """No .cim and no staging file: the container only ever exists in memory."""
    source = _source(tmp_path, ".ppm")
    out = tmp_path / "out"
    out.mkdir()
    Task().compress(input=str(source), output=str(out / "direct.ppm")).run()
    assert sorted(path.name for path in out.iterdir()) == ["direct.ppm"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out", "source.ppm"]


@pytest.mark.parametrize(
    "configure",
    [
        lambda: Task(packed_block_size=2),
        lambda: Task(y_block_size=16, cb_block_size=32, cr_block_size=32, packed_block_size=6),
        lambda: Task().with_coeff_removal(40.0),
        lambda: Task(transform="dct"),
        lambda: Task(transform="haar", packed_block_size=3).with_coeff_removal(5.0),
    ],
    ids=["packed-2", "larger-blocks", "coefficient-removal", "dct", "haar-packed-3-removal"],
)
def test_every_setting_reaches_the_direct_route(
    tmp_path: Path, configure: Callable[[], Task]
) -> None:
    """The DCT and Haar are not bit-exact across machines, but this compares
    two runs on one machine, which do the same arithmetic in the same order."""
    source = _source(tmp_path, ".ppm")
    direct = tmp_path / "direct.ppm"
    configure().compress(input=str(source), output=str(direct)).run()

    expected = _two_steps(configure, source, tmp_path / "expected.ppm")
    assert direct.read_bytes() == expected.read_bytes()

    default = tmp_path / "default.ppm"
    Task().compress(input=str(source), output=str(default)).run()
    assert direct.read_bytes() != default.read_bytes(), "the setting changed nothing"


def test_a_declared_size_is_honoured_and_verified(tmp_path: Path) -> None:
    """A flat pickled list carries no size, so the direct route needs the
    declaration exactly as the two-step one does."""
    flat = tmp_path / "flat.pkl"
    flat.write_bytes(pickle.dumps(_picture(), protocol=4))
    direct = tmp_path / "direct.pkl"

    Task().with_input_size(WIDTH, HEIGHT).compress(input=str(flat), output=str(direct)).run()
    assert _pixels(direct).shape == (HEIGHT, WIDTH, 3)

    with pytest.raises(ValueError, match=f"is {WIDTH}x{HEIGHT}, not the 10x96 declared"):
        Task().with_input_size(10, 96).compress(
            input=str(_source(tmp_path, ".ppm")), output=str(tmp_path / "wrong.ppm")
        ).run()
    assert not (tmp_path / "wrong.ppm").exists()


# --- one file type at a time, for now -----------------------------------------


@pytest.mark.parametrize(
    ("source", "target"),
    [(".ppm", ".png"), (".png", ".ppm"), (".bmp", ".tif"), (".npy", ".pkl"), (".pam", ".ppm")],
)
def test_another_file_type_is_not_implemented_yet(tmp_path: Path, source: str, target: str) -> None:
    path = _source(tmp_path, source)
    output = tmp_path / f"crossed{target}"
    task = Task()
    with pytest.raises(NotImplementedError) as caught:
        task.compress(input=str(path), output=str(output))

    message = str(caught.value)
    assert "planned for 0.6.0" in message
    assert CROSS_FORMAT_ISSUE in message and CROSS_FORMAT_ISSUE.endswith("/issues/51")
    assert f"compress to a .cim and extract that to {target}" in message
    assert str(path) in message and str(output) in message

    assert not output.exists()
    with pytest.raises(ValueError, match="nothing to run"):
        task.run()  # the refused call left the task as it was


def test_the_route_the_message_recommends_works(tmp_path: Path) -> None:
    source = _source(tmp_path, ".ppm")
    crossed = _two_steps(Task, source, tmp_path / "crossed.png")
    same = tmp_path / "same.ppm"
    Task().compress(input=str(source), output=str(same)).run()
    assert np.array_equal(_pixels(crossed), _pixels(same))


@pytest.mark.parametrize(
    ("source", "target"),
    [(".tif", ".tiff"), (".ppm", ".pnm"), (".pkl", ".pickle"), (".ppm", ".PPM"), (".png", ".Png")],
)
def test_two_spellings_of_one_file_type_are_one_file_type(
    tmp_path: Path, source: str, target: str
) -> None:
    """The type is the reader class, so suffixes that share one do not cross."""
    path = _source(tmp_path, source)
    output = tmp_path / f"respelled{target}"
    Task().compress(input=str(path), output=str(output)).run()
    assert np.array_equal(_pixels(output), _pixels(_two_steps(Task, path, tmp_path / f"e{source}")))


def test_standard_input_counts_as_a_bmp() -> None:
    """A stream has no suffix, and `reader_for(None)` has always meant BMP."""
    Task().compress(input=None, output="fine.bmp")
    with pytest.raises(NotImplementedError, match=r"planned for 0\.6\.0"):
        Task().compress(input=None, output="crossed.ppm")


def test_an_input_this_package_cannot_read_is_refused_when_the_output_is_a_picture() -> None:
    with pytest.raises(UnsupportedFileFormatError, match=r"unsupported image format '\.jpg'"):
        Task().compress(input="photo.jpg", output="photo_compressed.ppm")


# --- anything that is not a picture is still the container --------------------


@pytest.mark.parametrize("name", ["out.cim", "out.CIM", "out", "out.bin", "archive.tar"])
def test_any_other_output_name_gets_the_container(tmp_path: Path, name: str) -> None:
    source = _source(tmp_path, ".ppm")
    reference = tmp_path / "reference.cim"
    Task().compress(input=str(source), output=str(reference)).run()

    output = tmp_path / name
    Task().compress(input=str(source), output=str(output)).run()
    assert output.read_bytes() == reference.read_bytes()


def test_the_container_still_goes_to_standard_output(
    tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    source = _source(tmp_path, ".ppm")
    reference = tmp_path / "reference.cim"
    Task().compress(input=str(source), output=str(reference)).run()

    Task().compress(input=str(source), output=None).run()
    assert capfdbinary.readouterr().out == reference.read_bytes()


def test_extract_takes_its_input_and_output_the_same_way(tmp_path: Path) -> None:
    source = _source(tmp_path, ".ppm")
    container = tmp_path / "c.cim"
    Task().compress(str(source), str(container)).run()  # positionally, too
    Task().extract(input=str(container), output=str(tmp_path / "back.png")).run()
    assert _pixels(tmp_path / "back.png").shape == (HEIGHT, WIDTH, 3)


def test_a_path_object_is_as_good_as_a_string(tmp_path: Path) -> None:
    source = _source(tmp_path, ".ppm")
    Task().compress(input=source, output=tmp_path / "direct.ppm").run()
    assert (tmp_path / "direct.ppm").stat().st_size == source.stat().st_size


def test_one_task_can_be_told_to_do_something_else(tmp_path: Path) -> None:
    """compress() and extract() replace the plan; the settings stay."""
    source = _source(tmp_path, ".ppm")
    task = Task(packed_block_size=2)
    task.compress(input=str(source), output=str(tmp_path / "a.cim")).run()
    task.extract(input=str(tmp_path / "a.cim"), output=str(tmp_path / "a.ppm")).run()
    task.compress(input=str(source), output=str(tmp_path / "b.ppm")).run()
    assert (tmp_path / "a.ppm").read_bytes() == (tmp_path / "b.ppm").read_bytes()
