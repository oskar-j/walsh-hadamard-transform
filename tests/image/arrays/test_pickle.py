"""Pickled arrays and pixel lists as images, read without executing anything (0.4.15)."""

from __future__ import annotations

import os
import pickle
import re
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from conftest import gradient_pixels
from walsh import Codec, PickleImage
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
            r"invalid pickle width: expected a positive integer, found 0$",
        ),
        (
            lambda a: {"width": WIDTH, "height": -4, "pixels": _flat(a)},
            None,
            r"invalid pickle height: expected a positive integer, found -4$",
        ),
        (
            lambda a: {"width": True, "height": HEIGHT, "pixels": _flat(a)},
            None,
            r"invalid pickle width: expected a positive integer, found True$",
        ),
        (
            lambda a: {"width": 5.0, "height": HEIGHT, "pixels": _flat(a)},
            None,
            r"invalid pickle width: expected a positive integer, found 5\.0$",
        ),
        (
            lambda a: {"width": "5", "height": HEIGHT, "pixels": _flat(a)},
            None,
            r"invalid pickle width: expected a positive integer, found '5'$",
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
        (b"not a pickle at all", r"not a readable pickle: byte 0 is 0x6e, which is not a pickle"),
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


# --- the boundary, opcode by opcode (#54) -------------------------------------


def test_an_allowlisted_name_cannot_be_altered_by_a_build() -> None:
    """BUILD sets attributes on whatever is on the stack, and NumPy's own
    functions are on it. _Sealed has nothing a BUILD can set, so an 80-byte
    pickle can no longer rewrite _frombuffer.__defaults__ for the process."""
    sealed = pkl_module._Sealed(lambda value: value + 1)
    assert sealed(1) == 2
    with pytest.raises(AttributeError, match=r"has no attribute __defaults__ to set"):
        sealed.__defaults__ = (5,)  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "state",
    [
        (2, (), np.dtype("u1"), False, b""),  # a version that is not 0 or 1
        (1, (2,), np.dtype("u1"), False),  # one item short
        (1, [2], np.dtype("u1"), False, b"\x00\x00"),  # a shape that is not a tuple
        (1, (-1,), np.dtype("u1"), False, b""),  # a negative axis
        (1, (2,), "u1", False, b"\x00\x00"),  # a dtype that is not a dtype
        (1, (2,), np.dtype("u1"), False, [0, 0]),  # a list where bytes are due
        (1, (2,), np.dtype("u1"), False, b"\x00"),  # too few bytes for the shape
        (1, (2,), np.dtype("O"), False, [0]),  # too few objects for the shape
    ],
    ids=[
        "a strange version",
        "one item short",
        "a shape that is not a tuple",
        "a negative axis",
        "a dtype that is not one",
        "a list of bytes",
        "too few bytes",
        "too few objects",
    ],
)
def test_a_pickled_array_state_numpy_would_not_write_is_refused(state: tuple[Any, ...]) -> None:
    """NumPy's ndarray.__setstate__ crashed the interpreter on a short object
    array (#54); the array a pickle builds checks its own state first."""
    array = pkl_module._reconstruct(pkl_module._NDARRAY, (0,), b"b")
    with pytest.raises(pickle.UnpicklingError, match=r"an array (state|reconstruction)"):
        array.__setstate__(state)


def test_a_dtype_named_by_something_that_is_not_a_string_is_refused() -> None:
    with pytest.raises(pickle.UnpicklingError, match=r"a NumPy dtype named by a list"):
        pkl_module._dtype([1, 2])


def test_below_covers_the_ends_of_its_range() -> None:
    """A four-byte little-endian count below the bound, as a pattern."""
    assert pkl_module._below(2**32) == b"...."
    assert pkl_module._below(0) == b"(?!)"


@pytest.mark.parametrize(
    ("stop", "limit", "end", "outcome"),
    [(5, 10, 10, True), (11, 10, 10, False), (11, 10, 20, "raises")],
    ids=["fits", "the data ends first", "runs past a frame"],
)
def test_ends_within_tells_a_short_file_from_a_broken_frame(
    stop: int, limit: int, end: int, outcome: bool | str
) -> None:
    if outcome == "raises":
        with pytest.raises(UnsupportedFileFormatError, match=r"runs past the end of its frame"):
            pkl_module._ends_within(stop, limit, end, 0, "pickle")
    else:
        assert pkl_module._ends_within(stop, limit, end, 0, "pickle") is outcome


@pytest.mark.parametrize("index", [-1, 100], ids=["negative", "past the file"])
def test_a_memo_index_the_file_cannot_fill_is_refused(index: int) -> None:
    with pytest.raises(UnsupportedFileFormatError, match=r"names a memo slot outside"):
        pkl_module._check_memo_index(index, "PUT", 0, 10, "pickle")
    pkl_module._check_memo_index(5, "PUT", 0, 10, "pickle")  # one that fits does not


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (b"r" + (2**30).to_bytes(4, "little") + b".", r"the LONG_BINPUT at byte 0 names a memo"),
        (b"p" + b"9" * 40 + b"\n.", r"the PUT at byte 0 names a memo"),
        (b"p-3\n.", r"the PUT at byte 0 names a memo"),
        (b"pxx\n.", r"the PUT at byte 0 names a memo"),
        (b"\x8e" + (2**48).to_bytes(8, "little") + b".", r"BINBYTES8 at byte 0 declares"),
    ],
    ids=["a huge LONG_BINPUT", "a huge PUT", "a negative PUT", "a non-number PUT", "a length bomb"],
)
def test_check_opcodes_refuses_a_number_larger_than_the_file(data: bytes, match: str) -> None:
    """The unpickler reserves these before reading what they describe, so the
    walk refuses them first: 12 bytes asked for 256 TiB, 9 bytes for 256 MiB."""
    with pytest.raises(UnsupportedFileFormatError, match=match):
        pkl_module._check_opcodes(data, "pickle")


@pytest.mark.parametrize(
    "data", [b"K", b"\x8a", b"\x8e\x01\x00"], ids=["a plain opcode", "a one-byte field", "a length"]
)
def test_check_opcodes_leaves_a_truncated_tail_to_the_unpickler(data: bytes) -> None:
    """A stream that ends mid-opcode is not the walk's to report: the
    unpickler stops at the same byte and says the data was truncated."""
    pkl_module._check_opcodes(data, "pickle")  # returns, raising nothing
    with pytest.raises(UnsupportedFileFormatError, match=r"not a readable pickle"):
        safe_loads(data)


def test_pixels_from_object_is_usable_on_its_own(expected: np.ndarray) -> None:
    np.testing.assert_array_equal(pixels_from_object(_flat(expected), (WIDTH, HEIGHT)), expected)
    assert pixels_from_object(_rows(expected)).flags.c_contiguous


def test_a_pickled_array_is_handed_on_as_a_plain_ndarray(
    expected: np.ndarray, tmp_path: Path
) -> None:
    """Unpickling builds a private subclass that checks its state (#54), and
    that is not what a caller of get_array() should be holding."""
    loaded = safe_loads(pickle.dumps(expected))
    assert isinstance(loaded, np.ndarray) and type(loaded) is not np.ndarray
    assert type(pixels_from_object(loaded)) is np.ndarray
    assert type(_load(_dump(tmp_path / "a.pkl", expected))) is np.ndarray


@pytest.mark.parametrize(
    ("array", "name"),
    [
        (np.array([["ab", "cd"]]), r"<U2|>U2"),
        (np.array([[b"ab", b"cd"]]), r"\|S2"),
        (np.zeros((2, 2), dtype="M8[s]"), r"[<>]M8"),
    ],
    ids=["text", "bytes", "datetime"],
)
def test_only_numpys_built_in_number_types_are_read(array: np.ndarray, name: str) -> None:
    """A pickled dtype is resolved to the instance NumPy shares, which a
    text, datetime or byte-swapped type does not have (#54). None of them
    could be an image's samples."""
    with pytest.raises(
        UnsupportedFileFormatError,
        match=rf"unsupported pickle: it holds the NumPy dtype ({name}); only plain numbers",
    ):
        safe_loads(pickle.dumps(array))


def test_a_pickled_byte_order_is_not_applied(expected: np.ndarray) -> None:
    """It arrives as a state for the dtype, which the shared instance does
    not take. uint8 has no byte order to lose, and a wider type is refused
    as an image's samples either way."""
    swapped = np.dtype("u2").newbyteorder()
    loaded = safe_loads(pickle.dumps(expected.astype(swapped)))
    assert isinstance(loaded, np.ndarray) and loaded.dtype.isnative
    with pytest.raises(UnsupportedFileFormatError, match=r"unsupported pickle dtype uint16"):
        pixels_from_object(loaded)


# --- what a file may cost -----------------------------------------------------


@pytest.mark.parametrize(
    "sample",
    [10**30, 2**63, -(2**70), np.uint64(2**63)],
    ids=["a large int", "one past int64", "a large negative", "numpy uint64"],
)
def test_a_sample_beyond_64_bits_is_refused_by_name(sample: Any, tmp_path: Path) -> None:
    """The range check ran after a conversion to int64, which such a sample
    fails with OverflowError: a traceback on the command line (#58)."""
    path = _dump(tmp_path / "big.pkl", [[(sample, 0, 0)]])
    with pytest.raises(
        UnsupportedFileFormatError, match=r"sample beyond 64 bits: expected integers in 0-255"
    ):
        PickleImage().load(str(path))


def test_the_largest_int64_sample_is_still_named(tmp_path: Path) -> None:
    path = _dump(tmp_path / "big.pkl", [[(2**63 - 1, 0, 0)]])
    with pytest.raises(UnsupportedFileFormatError, match=r"sample 9223372036854775807: expected"):
        PickleImage().load(str(path))


def _nested(depth: int) -> tuple[Any, ...]:
    value: tuple[Any, ...] = (0,)
    for _ in range(depth):
        value = (value,)
    return value


@pytest.mark.parametrize(
    ("width", "shown"),
    [
        (_nested(50), "a tuple"),
        ([1, 2, 3], "a list"),
        (2**100, "a 101-bit int"),
        (-(2**100), "a 101-bit int"),
        ("w" * 41, "a str"),
        ("w" * 40, "'" + "w" * 40 + "'"),
        (None, "None"),
        (np.float64(2.5), r"(np\.float64\()?2\.5\)?"),
        (np.int64(-3), r"(np\.int64\()?-3\)?"),
    ],
    ids=[
        "nested tuples",
        "a list",
        "a huge int",
        "a huge negative int",
        "a long string",
        "a short string",
        "None",
        "a numpy float",
        "a numpy integer",
    ],
)
def test_a_message_shows_a_value_from_the_file_at_a_bounded_length(
    width: Any, shown: str, expected: np.ndarray
) -> None:
    """The message used repr() of the value, which the file chooses (#58).
    Nested tuples recurse in repr() and ended in RecursionError, and shared
    ones expand, so a few hundred bytes of pickle printed megabytes."""
    wrapper = {"width": width, "height": HEIGHT, "pixels": _flat(expected)}
    with pytest.raises(UnsupportedFileFormatError) as refused:
        pixels_from_object(wrapper)
    message = str(refused.value)
    assert re.fullmatch(
        rf"invalid pickle width: expected a positive integer, found {shown}", message
    ), message


def test_a_nesting_too_deep_for_repr_is_still_refused_by_name(expected: np.ndarray) -> None:
    """20,000 levels, which the unpickler builds without recursing and
    repr() cannot print: RecursionError, where a message was due (#58)."""
    wrapper = {"width": _nested(20_000), "height": HEIGHT, "pixels": _flat(expected)}
    with pytest.raises(
        UnsupportedFileFormatError,
        match=r"invalid pickle width: expected a positive integer, found a tuple$",
    ):
        pixels_from_object(wrapper)


def test_a_dict_with_many_keys_names_a_few_of_them() -> None:
    crowded = {f"key{index}": index for index in range(9)}
    with pytest.raises(
        UnsupportedFileFormatError,
        match=r"found 'key0', 'key1', 'key2', 'key3', 'key4' and 4 more$",
    ):
        pixels_from_object(crowded)
    with pytest.raises(UnsupportedFileFormatError, match=r"found 'width', a tuple$"):
        pixels_from_object({"width": 1, _nested(3): 2})


@pytest.mark.parametrize("edge", [100, 1000])
def test_a_small_canvas_of_one_repeated_row_still_loads(edge: int, tmp_path: Path) -> None:
    """A pickle stores the repeated row once, so the file is small for its
    picture. Up to 1032 samples a byte that is a picture like any other."""
    path = _dump(tmp_path / "canvas.pkl", [[(1, 2, 3)] * edge] * edge)
    assert path.stat().st_size < 5 * edge
    image = PickleImage()
    image.load(str(path))
    assert image.get_dimensions() == (edge, edge)
    assert (image.get_array() == (1, 2, 3)).all()


def test_a_picture_its_file_is_too_small_to_hold_is_refused_before_it_is_walked(
    tmp_path: Path,
) -> None:
    """16 KB that describe 4000x4000 pixels took 4 s and 570 MiB to walk,
    and the cost grows with the square of the file's size (#54)."""
    path = _dump(tmp_path / "canvas.pkl", [[(1, 2, 3)] * 4000] * 4000)
    size = path.stat().st_size
    assert size < 17_000

    tracemalloc.start()
    try:
        with pytest.raises(
            UnsupportedFileFormatError,
            match=rf"its {size} bytes describe 4000x4000 pixels of 3 samples.*"
            r"at most 1032 samples a byte are read\. Pickle the picture as a NumPy array",
        ):
            PickleImage().load(str(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2_000_000


def test_the_bound_is_on_samples_so_one_long_pixel_cannot_pass_it(tmp_path: Path) -> None:
    """Every pixel the same tuple of 2000 samples: 5 KB cost 10 s and
    1.6 GiB before the channel count was refused. It is refused first now."""
    path = _dump(tmp_path / "wide.pkl", [[tuple([7] * 2000)] * 300] * 300)
    with pytest.raises(UnsupportedFileFormatError, match=r"shape \(300, 300, 2000\): 2000 chan"):
        PickleImage().load(str(path))


def test_the_wrapper_and_an_object_array_are_held_to_the_same_bound(tmp_path: Path) -> None:
    canvas = [[(1, 2, 3)] * 2000] * 2000
    wrapped = _dump(tmp_path / "wrapped.pkl", {"width": 2000, "height": 2000, "pixels": canvas})
    with pytest.raises(UnsupportedFileFormatError, match=r"describe 2000x2000 pixels of 3"):
        PickleImage().load(str(wrapped))

    rows = np.empty(2000, dtype=object)
    rows[:] = canvas
    np.save(tmp_path / "rows.npy", rows, allow_pickle=True)
    with pytest.raises(
        UnsupportedFileFormatError,
        match=r"unsupported \.npy object array: its \d+ bytes describe 2000x2000 pixels of 3",
    ):
        reader_for("rows.npy").load(str(tmp_path / "rows.npy"))


def test_an_object_of_the_callers_own_has_no_bound() -> None:
    """The bound is a file's. pixels_from_object on a list made in memory
    takes no size, and reads whatever it is given."""
    canvas = [[(1, 2, 3)] * 300] * 300
    assert pixels_from_object(canvas).shape == (300, 300, 3)
    assert pixels_from_object(canvas, source_bytes=262).shape == (300, 300, 3)
    with pytest.raises(UnsupportedFileFormatError, match=r"its 261 bytes describe 300x300"):
        pixels_from_object(canvas, source_bytes=261)

    flat = [(1, 2, 3)] * 400
    assert pixels_from_object(flat, (20, 20), source_bytes=2).shape == (20, 20, 3)
    with pytest.raises(UnsupportedFileFormatError, match=r"its 1 bytes describe 20x20 pixels"):
        pixels_from_object(flat, (20, 20), source_bytes=1)


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


# --- the declared size, through Codec and the command line ---------------------


def test_every_form_compresses_to_the_same_cim_as_the_ppm(
    expected: np.ndarray, gradient_ppm: Path, tmp_path: Path
) -> None:
    from conftest import write_ppm

    source = write_ppm(tmp_path / "g.ppm", WIDTH, HEIGHT, gradient_pixels(WIDTH, HEIGHT))
    Codec().compress(input=str(source), output=str(tmp_path / "ref.cim")).run()
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
        Codec().with_input_size(*size).compress(input=str(path), output=str(out)).run()
        assert out.read_bytes() == reference


def test_extract_writes_a_pickle_that_round_trips(gradient_ppm: Path, tmp_path: Path) -> None:
    Codec().compress(input=str(gradient_ppm), output=str(tmp_path / "g.cim")).run()
    for target in ("back.pkl", "back.ppm"):
        Codec().extract(input=str(tmp_path / "g.cim"), output=str(tmp_path / target)).run()
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
        Codec().with_input_size(width, height)


def test_with_input_size_chains_and_two_nones_clear_it(
    expected: np.ndarray, tmp_path: Path
) -> None:
    codec = Codec()
    assert codec.with_input_size(3, 3) is codec
    assert codec.with_input_size(None, None) is codec
    path = _dump(tmp_path / "rows.pkl", _rows(expected))
    codec.compress(input=str(path), output=str(tmp_path / "o.cim")).run()


def test_a_declared_size_is_verified_for_a_format_that_has_its_own(
    gradient_ppm: Path, tmp_path: Path
) -> None:
    out = tmp_path / "o.cim"
    Codec().with_input_size(16, 16).compress(input=str(gradient_ppm), output=str(out)).run()
    out.unlink()
    with pytest.raises(ValueError, match=r"gradient\.ppm is 16x16, not the 8x32 declared"):
        Codec().with_input_size(8, 32).compress(input=str(gradient_ppm), output=str(out)).run()
    assert not out.exists()


def test_an_input_size_means_nothing_to_extract(gradient_ppm: Path, tmp_path: Path) -> None:
    Codec().compress(input=str(gradient_ppm), output=str(tmp_path / "g.cim")).run()
    with pytest.raises(ValueError, match="applies to compress only"):
        Codec().with_input_size(16, 16).extract(
            input=str(tmp_path / "g.cim"), output=str(tmp_path / "b.ppm")
        ).run()
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
