"""Pickled arrays and pixel lists as images, read without executing anything (0.4.15)."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from conftest import gradient_pixels
from walsh import PickleImage, Task
from walsh.cli import main
from walsh.exceptions import UnsupportedFileFormatError
from walsh.image import PICKLE_PROTOCOL, pixels_from_object, reader_for, safe_loads
from walsh.image.arrays import pkl as pkl_module

WIDTH, HEIGHT = 5, 4


@pytest.fixture
def expected() -> np.ndarray:
    return np.array(gradient_pixels(WIDTH, HEIGHT), dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)


def _dump(path: Path, value: object, protocol: int = PICKLE_PROTOCOL) -> Path:
    with path.open("wb") as file:
        pickle.dump(value, file, protocol=protocol)
    return path


def _load(path: Path, size: tuple[int, int] | None = None) -> np.ndarray:
    image = PickleImage()
    if size is not None:
        image.declare_size(*size)
    image.load(str(path))
    assert image.get_dimensions() == (WIDTH, HEIGHT)
    return image.get_array()


def _rows(array: np.ndarray) -> list[list[tuple[int, ...]]]:
    return [list(map(tuple, row)) for row in array.tolist()]


def _flat(array: np.ndarray) -> list[tuple[int, ...]]:
    return [pixel for row in _rows(array) for pixel in row]


# --- what is accepted ---------------------------------------------------------


@pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4, 5])
@pytest.mark.parametrize("layout", ["c", "fortran", "view"])
def test_a_pickled_array_loads_at_every_protocol_and_memory_layout(
    protocol: int, layout: str, expected: np.ndarray, tmp_path: Path
) -> None:
    """Protocols 0-2 carry bytes through _codecs.encode, 3-4 through
    _reconstruct, and 5 through _frombuffer for contiguous arrays."""
    wide = np.concatenate([expected, expected], axis=1)
    array = {"c": expected, "fortran": np.asfortranarray(expected), "view": wide[:, :WIDTH]}[layout]
    np.testing.assert_array_equal(_load(_dump(tmp_path / "a.pkl", array, protocol)), expected)


def test_numpy_load_reads_the_same_file(expected: np.ndarray, tmp_path: Path) -> None:
    """The request was 'pickle files through numpy.load()': same files, read safely."""
    path = _dump(tmp_path / "a.pkl", expected)
    np.testing.assert_array_equal(np.load(path, allow_pickle=True), _load(path))


def test_pickles_from_numpy_1_and_numpy_2_both_load(expected: np.ndarray, tmp_path: Path) -> None:
    """NumPy 1 writes numpy.core and NumPy 2 numpy._core. numpy.load cannot
    cross that line in the old direction; the allowlist maps both spellings."""
    data = pickle.dumps(expected, protocol=3)
    assert b"numpy._core.multiarray" in data or b"numpy.core.multiarray" in data
    as_numpy_2 = data.replace(b"numpy.core.multiarray", b"numpy._core.multiarray")
    as_numpy_1 = as_numpy_2.replace(b"numpy._core.multiarray", b"numpy.core.multiarray")
    assert as_numpy_1 != as_numpy_2
    for name, payload in (("one.pkl", as_numpy_1), ("two.pkl", as_numpy_2)):
        (tmp_path / name).write_bytes(payload)
        np.testing.assert_array_equal(_load(tmp_path / name), expected)


@pytest.mark.parametrize(
    "build",
    [
        lambda a: _rows(a),
        lambda a: a.tolist(),
        lambda a: tuple(tuple(row) for row in _rows(a)),
        lambda a: [tuple(list(p) for p in row) for row in _rows(a)],
        lambda a: [[tuple(p) for p in row] for row in a],
        lambda a: {"width": WIDTH, "height": HEIGHT, "pixels": _flat(a)},
        lambda a: {"width": WIDTH, "height": HEIGHT, "pixels": [list(p) for p in _flat(a)]},
        lambda a: {"width": WIDTH, "height": HEIGHT, "pixels": _rows(a)},
        lambda a: {"width": WIDTH, "height": HEIGHT, "pixels": a},
        lambda a: {"width": np.int64(WIDTH), "height": np.int64(HEIGHT), "pixels": _flat(a)},
        lambda a: np.array([*_rows(a), None], dtype=object)[:-1],
    ],
    ids=[
        "rows of tuples",
        "rows of lists",
        "tuples all the way down",
        "lists and tuples mixed",
        "numpy scalars as samples",
        "dict around a flat list of tuples",
        "dict around a flat list of lists",
        "dict around rows",
        "dict around an array",
        "dict with numpy integer sizes",
        "object array of rows",
    ],
)
def test_every_self_describing_form_gives_the_same_picture(
    build: Any, expected: np.ndarray, tmp_path: Path
) -> None:
    np.testing.assert_array_equal(_load(_dump(tmp_path / "p.pkl", build(expected))), expected)


@pytest.mark.parametrize("pixel", [tuple, list], ids=["tuples", "lists"])
def test_a_flat_list_takes_its_size_from_the_declaration(
    pixel: Any, expected: np.ndarray, tmp_path: Path
) -> None:
    flat = [pixel(p) for p in _flat(expected)]
    path = _dump(tmp_path / "flat.pkl", flat)
    np.testing.assert_array_equal(_load(path, (WIDTH, HEIGHT)), expected)


def test_greyscale_and_opaque_rgba_follow_the_array_rules(
    expected: np.ndarray, tmp_path: Path
) -> None:
    grey = expected[:, :, 0]
    for name, value in (("2d", grey), ("one channel", grey[:, :, None])):
        loaded = _load(_dump(tmp_path / f"{name}.pkl", value))
        np.testing.assert_array_equal(loaded, np.repeat(grey[:, :, None], 3, axis=2))

    opaque = np.dstack([expected, np.full((HEIGHT, WIDTH), 255, dtype=np.uint8)])
    np.testing.assert_array_equal(_load(_dump(tmp_path / "rgba.pkl", opaque)), expected)
    np.testing.assert_array_equal(_load(_dump(tmp_path / "rgba-rows.pkl", _rows(opaque))), expected)


def test_a_declared_size_is_checked_against_input_that_has_its_own(
    expected: np.ndarray, tmp_path: Path
) -> None:
    for value in (
        expected,
        _rows(expected),
        {"width": WIDTH, "height": HEIGHT, "pixels": _flat(expected)},
    ):
        path = _dump(tmp_path / "p.pkl", value)
        np.testing.assert_array_equal(_load(path, (WIDTH, HEIGHT)), expected)
        with pytest.raises(UnsupportedFileFormatError, match=r"not the 4x5 (declared|asked for)"):
            _load(path, (HEIGHT, WIDTH))


# --- what is refused, by name -------------------------------------------------


def _translucent(a: np.ndarray) -> np.ndarray:
    return np.dstack([a, np.full(a.shape[:2], 128, dtype=np.uint8)])


@pytest.mark.parametrize(
    ("build", "size", "match"),
    [
        (
            lambda a: _flat(a),
            None,
            r"flat list of 20 pixels does not say how wide.*--width and --height",
        ),
        (lambda a: _flat(a), (3, 3), r"20 pixels cannot fill the declared 3x3 \(9 pixels\)"),
        (lambda a: _flat(a)[:-1], (WIDTH, HEIGHT), r"19 pixels cannot fill the declared 5x4"),
        (
            lambda a: {"width": WIDTH, "pixels": _flat(a)},
            None,
            r"exactly the keys 'width', 'height' and 'pixels', found 'pixels', 'width'",
        ),
        (lambda a: {}, None, r"exactly the keys .* found none"),
        (
            lambda a: {"width": 0, "height": HEIGHT, "pixels": _flat(a)},
            None,
            r"invalid pickle width 0",
        ),
        (
            lambda a: {"width": WIDTH, "height": -4, "pixels": _flat(a)},
            None,
            r"invalid pickle height -4",
        ),
        (
            lambda a: {"width": True, "height": HEIGHT, "pixels": _flat(a)},
            None,
            r"invalid pickle width True",
        ),
        (
            lambda a: {"width": 5.0, "height": HEIGHT, "pixels": _flat(a)},
            None,
            r"invalid pickle width 5.0",
        ),
        (
            lambda a: {"width": "5", "height": HEIGHT, "pixels": _flat(a)},
            None,
            r"invalid pickle width '5'",
        ),
        (
            lambda a: {
                "width": WIDTH,
                "height": HEIGHT,
                "pixels": {"width": 1, "height": 1, "pixels": []},
            },
            None,
            r"'pixels' is another dict",
        ),
        (
            lambda a: {"width": 2, "height": 10, "pixels": _rows(a)},
            None,
            r"holds a 5x4 picture, not the 2x10 declared",
        ),
        (
            lambda a: [*_rows(a)[:-1], _rows(a)[-1][:-1]],
            None,
            r"every row must have the same, non-zero length, found \[4, 5\]",
        ),
        (
            lambda a: [[(1, 2, 3), (4, 5)]],
            None,
            r"every pixel must have the same, non-zero length, found \[2, 3\]",
        ),
        (
            lambda a: [[(1, 2, 3)], "abc"],
            None,
            r"every row must be a list or tuple, found list, str",
        ),
        (
            lambda a: [[(1, 2, 3), 7]],
            None,
            r"every pixel must be a list or tuple, found int, tuple",
        ),
        (lambda a: [], None, r"the pixel list is empty"),
        (lambda a: [[]], None, r"found a list of list"),
        (lambda a: [1, 2, 3], None, r"found a list of int"),
        (lambda a: [[(0.5, 0.5, 0.5)]], None, r"samples of type float: expected integers in 0-255"),
        (lambda a: [[(True, False, True)]], None, r"samples of type bool"),
        (lambda a: [[("1", "2", "3")]], None, r"samples of type str"),
        (lambda a: [[(0, 0, 256)]], None, r"sample 256: expected integers in 0-255"),
        (lambda a: [[(0, -1, 0)]], None, r"sample -1: expected integers in 0-255"),
        (
            lambda a: [[(1, 2)]],
            None,
            r"2 channels; expected 1 \(greyscale\), 3 \(RGB\) or 4 \(RGBA\)",
        ),
        (lambda a: [[(1, 2, 3, 4, 5)]], None, r"5 channels"),
        (lambda a: _translucent(a), None, r"fourth channel is not fully opaque"),
        (lambda a: _rows(_translucent(a)), None, r"fourth channel is not fully opaque"),
        (lambda a: a.astype(np.float64) / 255, None, r"unsupported pickle dtype float64"),
        (lambda a: a.astype(np.uint16), None, r"unsupported pickle dtype uint16"),
        (lambda a: a[None], None, r"unsupported pickle shape \(1, 4, 5, 3\)"),
        (lambda a: "a picture, honest", None, r"it holds a str, not an image"),
        (lambda a: None, None, r"it holds a NoneType, not an image"),
        (lambda a: {1, 2, 3}, None, r"it holds a set, not an image"),
    ],
)
def test_everything_else_is_refused_by_name(
    build: Any, size: tuple[int, int] | None, match: str, expected: np.ndarray, tmp_path: Path
) -> None:
    path = _dump(tmp_path / "bad.pkl", build(expected))
    image = PickleImage()
    if size is not None:
        image.declare_size(*size)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        image.load(str(path))


# --- nothing in a pickle is executed ------------------------------------------


class _RunsACommand:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        return os.system, (f"echo pwned > {self.marker}",)


def test_a_pickle_that_would_run_a_command_is_refused_and_runs_nothing(tmp_path: Path) -> None:
    """The reason this reader exists. pickle.load on this file runs the command."""
    marker = tmp_path / "pwned"
    path = _dump(tmp_path / "evil.pkl", [[_RunsACommand(marker)]])

    with pytest.raises(
        UnsupportedFileFormatError, match=r"it refers to \w+\.system.*ever executed"
    ):
        PickleImage().load(str(path))
    assert not marker.exists()

    with path.open("rb") as file:  # and this is what the allowlist prevented
        pickle.load(file)
    assert marker.exists()


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        (b"\x80\x02cbuiltins\neval\n(X\x05\x00\x00\x001 + 1tR.", r"refers to builtins\.eval"),
        (b"\x80\x02c__builtin__\neval\n(X\x05\x00\x00\x001 + 1tR.", r"refers to __builtin__\.eval"),
        (b"\x80\x02cnumpy\nload\n.", r"refers to numpy\.load"),
        (
            b"\x80\x02cnumpy.core.multiarray\nfromfile\n.",
            r"refers to numpy\.core\.multiarray\.fromfile",
        ),
        # numpy.ndarray resolves to an inert token, so it cannot be called to allocate.
        (b"\x80\x02cnumpy\nndarray\n(J\xff\xff\xff\x7ftR.", r"not a readable pickle: TypeError"),
        # _reconstruct only builds the empty array NumPy itself starts from.
        (
            b"\x80\x03cnumpy._core.multiarray\n_reconstruct\n(cnumpy\nndarray\n(J\xff\xff\xff\x7ftC\x01btR.",
            r"an array reconstruction that NumPy would not have written",
        ),
        (
            b"\x80\x02c_codecs\nencode\n(X\x01\x00\x00\x00aX\x05\x00\x00\x00utf-8tR.",
            r"bytes encoded as 'utf-8', which pickle never writes",
        ),
        (b"\x80\x02X\x01\x00\x00\x00aQ.", r"not a readable pickle: UnpicklingError"),
        (b"not a pickle at all", r"not a readable pickle: UnpicklingError"),
        (b"", r"not a readable pickle: EOFError"),
        (pickle.dumps([[(1, 2, 3)]])[:-3], r"not a readable pickle"),
        (pickle.dumps([[(1, 2, 3)]]) + b"\x00", r"data follows the first pickle"),
        (pickle.dumps([[(1, 2, 3)]]) * 2, r"data follows the first pickle"),
    ],
    ids=[
        "eval",
        "python 2 eval",
        "numpy.load",
        "numpy fromfile",
        "ndarray called directly",
        "reconstruct with a shape",
        "encode with another codec",
        "persistent id",
        "garbage",
        "empty",
        "truncated",
        "trailing byte",
        "two pickles",
    ],
)
def test_hostile_and_malformed_pickles_are_refused(
    payload: bytes, match: str, tmp_path: Path
) -> None:
    path = tmp_path / "bad.pkl"
    path.write_bytes(payload)
    with pytest.raises(UnsupportedFileFormatError, match=match):
        PickleImage().load(str(path))


def test_running_out_of_memory_is_not_relabelled_as_a_bad_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def exhausted(self: object) -> object:
        raise MemoryError

    monkeypatch.setattr(pkl_module._RestrictedUnpickler, "load", exhausted)
    with pytest.raises(MemoryError):
        safe_loads(pickle.dumps([1]))


def test_safe_loads_returns_plain_data_and_names_its_source() -> None:
    value = {"a": [1, 2.5, "x", b"y", None, True], "b": (1, 2)}
    assert safe_loads(pickle.dumps(value)) == value
    with pytest.raises(UnsupportedFileFormatError, match=r"unsupported thing: it refers to"):
        safe_loads(pickle.dumps(print), "thing")


def test_the_allowlist_survives_a_numpy_without_frombuffer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pkl_module, "_NUMPY_FROMBUFFER", None)
    names = {name for _, name in pkl_module._allowed_globals()}
    assert names == {"ndarray", "dtype", "encode", "_reconstruct", "scalar"}


def test_pixels_from_object_is_usable_on_its_own(expected: np.ndarray) -> None:
    np.testing.assert_array_equal(pixels_from_object(_flat(expected), (WIDTH, HEIGHT)), expected)
    assert pixels_from_object(_rows(expected)).flags.c_contiguous


# --- the writer ---------------------------------------------------------------


def test_the_writer_makes_rows_of_int_tuples_that_need_no_numpy(
    expected: np.ndarray, tmp_path: Path
) -> None:
    image = PickleImage()
    image.set_array(expected)
    path = tmp_path / "out.pkl"
    image.save(str(path))

    data = path.read_bytes()
    assert data[:2] == bytes([0x80, PICKLE_PROTOCOL])
    assert b"numpy" not in data
    with path.open("rb") as file:
        rows = pickle.load(file)
    assert rows == _rows(expected)
    assert type(rows[0][0]) is tuple and type(rows[0][0][0]) is int

    np.testing.assert_array_equal(_load(path), expected)
    again = tmp_path / "again.pkl"
    image.save(str(again))
    assert again.read_bytes() == data, "the bytes are deterministic"


@pytest.mark.parametrize("name", ["p.pkl", "p.pickle", "P.PKL"])
def test_the_suffixes_select_the_pickle_reader(name: str) -> None:
    assert isinstance(reader_for(name), PickleImage)


# --- the declared size, through Task and the command line ---------------------


def test_every_form_compresses_to_the_same_cim_as_the_ppm(
    expected: np.ndarray, gradient_ppm: Path, tmp_path: Path
) -> None:
    from conftest import write_ppm

    source = write_ppm(tmp_path / "g.ppm", WIDTH, HEIGHT, gradient_pixels(WIDTH, HEIGHT))
    Task().with_action("compress").with_input(str(source)).with_output(
        str(tmp_path / "ref.cim")
    ).run()
    reference = (tmp_path / "ref.cim").read_bytes()

    forms: list[tuple[object, tuple[int | None, int | None]]] = [
        (expected, (None, None)),
        (_rows(expected), (None, None)),
        ({"width": WIDTH, "height": HEIGHT, "pixels": _flat(expected)}, (None, None)),
        (_flat(expected), (WIDTH, HEIGHT)),
    ]
    for index, (value, size) in enumerate(forms):
        path = _dump(tmp_path / f"{index}.pkl", value)
        out = tmp_path / f"{index}.cim"
        Task().with_input_size(*size).with_action("compress").with_input(str(path)).with_output(
            str(out)
        ).run()
        assert out.read_bytes() == reference


def test_extract_writes_a_pickle_that_round_trips(gradient_ppm: Path, tmp_path: Path) -> None:
    Task().with_action("compress").with_input(str(gradient_ppm)).with_output(
        str(tmp_path / "g.cim")
    ).run()
    for target in ("back.pkl", "back.ppm"):
        Task().with_action("extract").with_input(str(tmp_path / "g.cim")).with_output(
            str(tmp_path / target)
        ).run()
    via_pickle, via_ppm = reader_for("x.pkl"), reader_for("x.ppm")
    via_pickle.load(str(tmp_path / "back.pkl"))
    via_ppm.load(str(tmp_path / "back.ppm"))
    np.testing.assert_array_equal(via_pickle.get_array(), via_ppm.get_array())


@pytest.mark.parametrize(
    ("width", "height", "match"),
    [
        (4, None, "declared together"),
        (None, 4, "declared together"),
        (0, 4, "input width must be a positive integer, got 0"),
        (4, -1, "input height must be a positive integer, got -1"),
        (True, 4, "input width must be a positive integer, got True"),
        (4.0, 4, "input width must be a positive integer, got 4.0"),
    ],
)
def test_with_input_size_validates_when_it_is_set(width: Any, height: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        Task().with_input_size(width, height)


def test_with_input_size_chains_and_two_nones_clear_it(
    expected: np.ndarray, tmp_path: Path
) -> None:
    task = Task()
    assert task.with_input_size(3, 3) is task
    assert task.with_input_size(None, None) is task
    path = _dump(tmp_path / "rows.pkl", _rows(expected))
    task.with_action("compress").with_input(str(path)).with_output(str(tmp_path / "o.cim")).run()


def test_a_declared_size_is_verified_for_a_format_that_has_its_own(
    gradient_ppm: Path, tmp_path: Path
) -> None:
    out = tmp_path / "o.cim"
    Task().with_input_size(16, 16).with_action("compress").with_input(
        str(gradient_ppm)
    ).with_output(str(out)).run()
    out.unlink()
    with pytest.raises(ValueError, match=r"gradient\.ppm is 16x16, not the 8x32 declared"):
        Task().with_input_size(8, 32).with_action("compress").with_input(
            str(gradient_ppm)
        ).with_output(str(out)).run()
    assert not out.exists()


def test_an_input_size_means_nothing_to_extract(gradient_ppm: Path, tmp_path: Path) -> None:
    Task().with_action("compress").with_input(str(gradient_ppm)).with_output(
        str(tmp_path / "g.cim")
    ).run()
    with pytest.raises(ValueError, match="applies to compress only"):
        Task().with_input_size(16, 16).with_action("extract").with_input(
            str(tmp_path / "g.cim")
        ).with_output(str(tmp_path / "b.ppm")).run()
    assert not (tmp_path / "b.ppm").exists()


@pytest.mark.parametrize("bad", [(0, 1), (1, -1), (True, 1), (1.5, 1)])
def test_declare_size_rejects_anything_but_positive_integers(bad: Any) -> None:
    with pytest.raises(ValueError, match="must be a positive integer"):
        PickleImage().declare_size(*bad)


def test_the_command_line_takes_the_size_of_a_flat_list(
    expected: np.ndarray, tmp_path: Path
) -> None:
    flat = _dump(tmp_path / "flat.pkl", _flat(expected))
    rows = _dump(tmp_path / "rows.pkl", _rows(expected))
    runner = CliRunner()

    missing = runner.invoke(main, ["compress", str(flat), str(tmp_path / "no.cim")])
    assert missing.exit_code == 1
    assert "Error:" in missing.output and "--width and --height" in missing.output
    assert not (tmp_path / "no.cim").exists()

    sized = runner.invoke(
        main, ["compress", str(flat), str(tmp_path / "flat.cim"), "--width", "5", "--height", "4"]
    )
    assert sized.exit_code == 0, sized.output
    assert runner.invoke(main, ["compress", str(rows), str(tmp_path / "rows.cim")]).exit_code == 0
    assert (tmp_path / "flat.cim").read_bytes() == (tmp_path / "rows.cim").read_bytes()

    wrong = runner.invoke(
        main, ["compress", str(rows), str(tmp_path / "w.cim"), "--width", "4", "--height", "5"]
    )
    assert wrong.exit_code == 1 and "Error:" in wrong.output and "not the 4x5" in wrong.output


@pytest.mark.parametrize(
    "arguments",
    [["--width", "5"], ["--height", "4"], ["--width", "0", "--height", "4"]],
)
def test_half_a_size_or_a_zero_is_a_usage_error(
    arguments: list[str], expected: np.ndarray, tmp_path: Path
) -> None:
    flat = _dump(tmp_path / "flat.pkl", _flat(expected))
    result = CliRunner().invoke(main, ["compress", str(flat), str(tmp_path / "o.cim"), *arguments])
    assert result.exit_code == 2
    assert not (tmp_path / "o.cim").exists()


def test_a_hostile_pickle_is_a_clean_error_on_the_command_line(tmp_path: Path) -> None:
    marker = tmp_path / "pwned"
    path = _dump(tmp_path / "evil.pkl", _RunsACommand(marker))
    result = CliRunner().invoke(main, ["compress", str(path), str(tmp_path / "o.cim")])
    assert result.exit_code == 1
    assert "Error:" in result.output and "ever executed" in result.output
    assert not marker.exists() and not (tmp_path / "o.cim").exists()


def test_the_help_names_the_new_input_and_its_safety() -> None:
    result = CliRunner().invoke(main, ["compress", "--help"])
    assert ".pkl" in result.output and "--width" in result.output and "--height" in result.output
    assert "nothing in it is ever executed" in " ".join(result.output.split())
