"""`Codec.compress` with a picture as its output: the codec without the `.cim` file.

    Codec().compress(input="earth.ppm", output="earth_compressed.ppm").run()

What is written is the picture's lossy reconstruction, and the claim these
tests hold it to is that it is exactly what compressing to a `.cim` and
extracting that would have written. Since 0.5.2 (#51) the output may be any
supported format, whatever the input is: 0.5.1 had required the two to match.
"""

from __future__ import annotations

import io
import pickle
import sys
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
from walsh import Codec, UnsupportedFileFormatError, reader_for

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


def _two_steps(codec: Callable[[], Codec], source: Path, target: Path) -> Path:
    """The route the direct one must match: a .cim on disk in between."""
    container = target.with_suffix(".cim")
    codec().compress(input=str(source), output=str(container)).run()
    codec().extract(input=str(container), output=str(target)).run()
    return target


# --- the same picture as the two-step route -----------------------------------


@pytest.mark.parametrize("suffix", sorted(WRITERS))
def test_the_direct_output_is_what_compress_then_extract_writes(
    tmp_path: Path, suffix: str
) -> None:
    source = _source(tmp_path, suffix)
    direct = tmp_path / f"direct{suffix}"
    Codec().compress(input=str(source), output=str(direct)).run()

    expected = _two_steps(Codec, source, tmp_path / f"expected{suffix}")
    assert np.array_equal(_pixels(direct), _pixels(expected))
    if suffix != ".png":  # a PNG is pinned by its pixels; see test_png.py
        assert direct.read_bytes() == expected.read_bytes()


def test_the_output_is_a_lossy_picture_and_not_the_compressed_data(tmp_path: Path) -> None:
    """It is the *result* of the compression: a whole picture of the input's
    size and dimensions, close to the input and not equal to it."""
    source = _source(tmp_path, ".ppm")
    direct = tmp_path / "direct.ppm"
    Codec().compress(input=str(source), output=str(direct)).run()

    before, after = _pixels(source).astype(int), _pixels(direct).astype(int)
    assert direct.stat().st_size == source.stat().st_size
    assert before.shape == after.shape
    assert 0 < np.abs(before - after).mean() < 25


def test_nothing_but_the_output_is_written(tmp_path: Path) -> None:
    """No .cim and no staging file: the container only ever exists in memory."""
    source = _source(tmp_path, ".ppm")
    out = tmp_path / "out"
    out.mkdir()
    Codec().compress(input=str(source), output=str(out / "direct.ppm")).run()
    assert sorted(path.name for path in out.iterdir()) == ["direct.ppm"]
    assert sorted(path.name for path in tmp_path.iterdir()) == ["out", "source.ppm"]


@pytest.mark.parametrize(
    "configure",
    [
        lambda: Codec(packed_block_size=2),
        lambda: Codec(y_block_size=16, cb_block_size=32, cr_block_size=32, packed_block_size=6),
        lambda: Codec().with_coeff_removal(40.0),
        lambda: Codec(transform="dct"),
        lambda: Codec(transform="haar", packed_block_size=3).with_coeff_removal(5.0),
    ],
    ids=["packed-2", "larger-blocks", "coefficient-removal", "dct", "haar-packed-3-removal"],
)
def test_every_setting_reaches_the_direct_route(
    tmp_path: Path, configure: Callable[[], Codec]
) -> None:
    """The DCT and Haar are not bit-exact across machines, but this compares
    two runs on one machine, which do the same arithmetic in the same order."""
    source = _source(tmp_path, ".ppm")
    direct = tmp_path / "direct.ppm"
    configure().compress(input=str(source), output=str(direct)).run()

    expected = _two_steps(configure, source, tmp_path / "expected.ppm")
    assert direct.read_bytes() == expected.read_bytes()

    default = tmp_path / "default.ppm"
    Codec().compress(input=str(source), output=str(default)).run()
    assert direct.read_bytes() != default.read_bytes(), "the setting changed nothing"


def test_a_declared_size_is_honoured_and_verified(tmp_path: Path) -> None:
    """A flat pickled list carries no size, so the direct route needs the
    declaration exactly as the two-step one does."""
    flat = tmp_path / "flat.pkl"
    flat.write_bytes(pickle.dumps(_picture(), protocol=4))
    direct = tmp_path / "direct.pkl"

    Codec().with_input_size(WIDTH, HEIGHT).compress(input=str(flat), output=str(direct)).run()
    assert _pixels(direct).shape == (HEIGHT, WIDTH, 3)

    with pytest.raises(ValueError, match=f"is {WIDTH}x{HEIGHT}, not the 10x96 declared"):
        Codec().with_input_size(10, 96).compress(
            input=str(_source(tmp_path, ".ppm")), output=str(tmp_path / "wrong.ppm")
        ).run()
    assert not (tmp_path / "wrong.ppm").exists()


# --- any format in, any format out (#51) --------------------------------------


@pytest.mark.parametrize("target", sorted(WRITERS))
@pytest.mark.parametrize("source", sorted(WRITERS))
def test_every_format_can_be_written_from_every_other(
    tmp_path: Path, source: str, target: str
) -> None:
    """All 49 ordered pairs, the seven same-type ones among them. The codec
    works on pixels, which no format owns, so the pair must not matter: each
    gives what the two-step route gives for that target."""
    path = _source(tmp_path, source)
    direct = tmp_path / f"direct{target}"
    Codec().compress(input=str(path), output=str(direct)).run()

    expected = _two_steps(Codec, path, tmp_path / f"expected{target}")
    assert np.array_equal(_pixels(direct), _pixels(expected))
    if target != ".png":  # a PNG is pinned by its pixels; see test_png.py
        assert direct.read_bytes() == expected.read_bytes()


def test_the_source_format_leaves_no_trace_in_the_output(tmp_path: Path) -> None:
    """One picture in seven containers, each written straight to a PPM: seven
    identical files. This is the contract test_codec.py states for the .cim,
    carried through to the picture."""
    outputs = set()
    for suffix in sorted(WRITERS):
        output = tmp_path / f"from{suffix.lstrip('.')}.ppm"
        Codec().compress(input=str(_source(tmp_path, suffix)), output=str(output)).run()
        outputs.add(output.read_bytes())
    assert len(outputs) == 1


@pytest.mark.parametrize(
    ("source", "target"),
    [(".tif", ".tiff"), (".ppm", ".pnm"), (".pkl", ".pickle"), (".ppm", ".PPM"), (".bmp", ".Png")],
)
def test_the_output_suffix_may_be_spelled_any_way_the_format_is(
    tmp_path: Path, source: str, target: str
) -> None:
    path = _source(tmp_path, source)
    output = tmp_path / f"respelled{target}"
    Codec().compress(input=str(path), output=str(output)).run()
    assert np.array_equal(
        _pixels(output), _pixels(_two_steps(Codec, path, tmp_path / f"e{source}"))
    )


def test_a_declared_size_crosses_formats_too(tmp_path: Path) -> None:
    """The one input that cannot say its size, written as something else."""
    flat = tmp_path / "flat.pkl"
    flat.write_bytes(pickle.dumps(_picture(), protocol=4))
    output = tmp_path / "from_flat.png"
    Codec().with_input_size(WIDTH, HEIGHT).compress(input=str(flat), output=str(output)).run()

    expected = tmp_path / "expected.png"
    Codec().compress(input=str(_source(tmp_path, ".ppm")), output=str(expected)).run()
    assert np.array_equal(_pixels(output), _pixels(expected))

    with pytest.raises(ValueError, match="declared"):
        Codec().with_input_size(HEIGHT, WIDTH).compress(
            input=str(_source(tmp_path, ".bmp")), output=str(tmp_path / "wrong.tif")
        ).run()
    assert not (tmp_path / "wrong.tif").exists()


def test_standard_input_is_a_bmp_and_can_be_written_as_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stream has no suffix, and `reader_for(None)` has always meant BMP."""
    bmp = _source(tmp_path, ".bmp")
    stream = io.BytesIO(bmp.read_bytes())
    monkeypatch.setattr(sys, "stdin", type("Stdin", (), {"buffer": stream})())

    output = tmp_path / "from_stdin.ppm"
    Codec().compress(input=None, output=str(output)).run()
    expected = _two_steps(Codec, bmp, tmp_path / "expected.ppm")
    assert output.read_bytes() == expected.read_bytes()


def test_compress_only_plans_so_a_bad_name_is_reported_by_run(tmp_path: Path) -> None:
    """compress() reads and checks nothing, on either route. 0.5.1 had to look
    up the input's reader to compare file types, and so refused an unknown
    suffix early, for a picture output only. Both routes now say so from
    run(), and neither leaves a file behind."""
    for name in ("photo_compressed.ppm", "photo.cim"):
        output = tmp_path / name
        codec = Codec().compress(input=str(tmp_path / "photo.jpg"), output=str(output))
        with pytest.raises(UnsupportedFileFormatError, match=r"unsupported image format '\.jpg'"):
            codec.run()
        assert not output.exists()


# --- anything that is not a picture is still the container --------------------


@pytest.mark.parametrize("name", ["out.cim", "out.CIM", "out", "out.bin", "archive.tar"])
def test_any_other_output_name_gets_the_container(tmp_path: Path, name: str) -> None:
    source = _source(tmp_path, ".ppm")
    reference = tmp_path / "reference.cim"
    Codec().compress(input=str(source), output=str(reference)).run()

    output = tmp_path / name
    Codec().compress(input=str(source), output=str(output)).run()
    assert output.read_bytes() == reference.read_bytes()


def test_the_container_still_goes_to_standard_output(
    tmp_path: Path, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    source = _source(tmp_path, ".ppm")
    reference = tmp_path / "reference.cim"
    Codec().compress(input=str(source), output=str(reference)).run()

    Codec().compress(input=str(source), output=None).run()
    assert capfdbinary.readouterr().out == reference.read_bytes()


def test_extract_takes_its_input_and_output_the_same_way(tmp_path: Path) -> None:
    source = _source(tmp_path, ".ppm")
    container = tmp_path / "c.cim"
    Codec().compress(str(source), str(container)).run()  # positionally, too
    Codec().extract(input=str(container), output=str(tmp_path / "back.png")).run()
    assert _pixels(tmp_path / "back.png").shape == (HEIGHT, WIDTH, 3)


def test_a_path_object_is_as_good_as_a_string(tmp_path: Path) -> None:
    source = _source(tmp_path, ".ppm")
    Codec().compress(input=source, output=tmp_path / "direct.ppm").run()
    assert (tmp_path / "direct.ppm").stat().st_size == source.stat().st_size


def test_one_codec_can_be_told_to_do_something_else(tmp_path: Path) -> None:
    """compress() and extract() replace the plan; the settings stay."""
    source = _source(tmp_path, ".ppm")
    codec = Codec(packed_block_size=2)
    codec.compress(input=str(source), output=str(tmp_path / "a.cim")).run()
    codec.extract(input=str(tmp_path / "a.cim"), output=str(tmp_path / "a.ppm")).run()
    codec.compress(input=str(source), output=str(tmp_path / "b.ppm")).run()
    assert (tmp_path / "a.ppm").read_bytes() == (tmp_path / "b.ppm").read_bytes()
