"""Pickled pixels as images, read without executing anything in the file.

A pickle is a program for a small stack machine, and ``pickle.load`` and
``numpy.load(allow_pickle=True)`` run it with the power to import any module
and call anything in it: opening an untrusted pickle is running untrusted
code. This reader never does that. It unpickles through an allowlist
(:func:`safe_loads`): the container opcodes need no lookups at all, so lists,
tuples, dicts, integers and bytes load as they are, and the only names a file
may refer to are the handful NumPy's own pickles use to rebuild an array. Any
other name is refused, by name, before anything is called. ``numpy.ndarray``
itself is resolved to an inert token, because a pickle only ever passes it as
an argument, and handing out the class would let a file ask for an array of
any size. So the same files ``numpy.load(allow_pickle=True)`` and
``pickle.load`` read are read here, and nothing in them is executed.

An allowlist is not the whole boundary, because the machine does more than
look names up (#54). Its BUILD opcode hands a state to whatever object is on
the stack, through no lookup at all, and NumPy's ``__setstate__`` methods
trust their input: a 164-byte file crashed the interpreter through a dtype,
and an object array with too short a list did the same. So everything the
allowlist hands out either ignores a state (NumPy's shared built-in dtypes),
checks one before NumPy sees it (the arrays), or has nothing a state could
set (the functions). And the machine allocates by the numbers in its
opcodes before reading what they describe: 12 bytes asked for 256 TiB, and 9
bytes made it write 256 MiB of memo. So the opcode stream is walked once
before it runs, and a length or an index larger than the file is refused.

What may be inside, since a pickle can hold anything:

* a NumPy array, under the rules every array reader shares (``uint8``;
  ``(h, w, 3)`` RGB, ``(h, w)`` or ``(h, w, 1)`` greyscale, ``(h, w, 4)``
  RGBA only when fully opaque); pickles written under NumPy 1 and NumPy 2
  both load, whichever is installed;
* **rows of pixels**, ``[[(r, g, b), ...], ...]``, which carry their own size;
* a **flat list of pixels**, ``[(r, g, b), ...]``, top row first. That does
  not say how wide the picture is, and nothing here guesses, so the size must
  be declared: by pickling ``{"width": w, "height": h, "pixels": [...]}``
  instead, or with ``--width`` / ``--height`` (``Codec.with_input_size``).

Lists and tuples are interchangeable at every level, samples must be integers
in 0-255 (Python's or NumPy's, never floats or booleans), and ragged rows are
refused. The writer produces rows of ``(r, g, b)`` tuples of plain ``int`` at
protocol 4: that needs no NumPy to load and its bytes do not depend on which
NumPy wrote it, unlike a pickled array.
"""

from __future__ import annotations

import io
import logging
import math
import pickle
import re
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from functools import partial
from itertools import chain
from typing import Any, cast

import numpy as np

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, open_binary_read, open_binary_write
from walsh.image.arrays._rules import IMAGE_DTYPE, to_rgb, validate_image_array
from walsh.image.base import RasterImage, SizeCheck

__all__ = ["PICKLE_PROTOCOL", "PickleImage", "pixels_from_object", "safe_loads"]

log = logging.getLogger(__name__)

#: The protocol written. 4 is the newest every supported Python reads and
#: writes identically; 5 adds only out-of-band buffers, which a file cannot use.
PICKLE_PROTOCOL = 4

#: The keys of the wrapper that gives a flat pixel list its size.
_WRAPPER_KEYS = frozenset({"width", "height", "pixels"})

_MAX_SAMPLE = 255

#: The longest string :func:`_brief` shows as it is, and the most dict keys a
#: message lists.
_BRIEF_TEXT = 40
_BRIEF_KEYS = 5

#: The most samples a pickle may describe for each of its bytes. It is
#: DEFLATE's largest expansion, so a pickle costs no more to read than a PNG
#: of the same size already may.
_MAX_SAMPLES_PER_BYTE = 1032


class _NdarrayToken:
    """Stands in for ``numpy.ndarray`` while unpickling.

    NumPy's pickles mention the class only to pass it to ``_reconstruct``.
    Resolving the name to this inert object instead means a hostile pickle
    cannot call ``ndarray(shape)`` and have it allocate whatever it likes.
    It has no attributes, so BUILD has nothing to set on the one instance
    every load shares.
    """

    __slots__ = ()


_NDARRAY = _NdarrayToken()


class _Sealed:
    """An allowlisted function as a pickle sees it: callable, and nothing more.

    BUILD applies a state to whatever is on the stack, and that includes the
    objects :meth:`_RestrictedUnpickler.find_class` returns. On a Python
    function it can set any attribute: an 80-byte pickle rewrote NumPy's own
    ``_frombuffer.__defaults__`` for the rest of the process. This wrapper
    has no ``__dict__`` and refuses ``setattr``, so BUILD fails on it.
    """

    __slots__ = ("_function",)
    _function: Callable[..., Any]

    def __init__(self, function: Callable[..., Any]) -> None:
        """Wrap ``function``.

        Args:
            function: What a call on the wrapper runs.
        """
        object.__setattr__(self, "_function", function)

    def __setattr__(self, name: str, value: object) -> None:
        """Refuse every attribute, which is what BUILD would set.

        Args:
            name: The attribute.
            value: Its value.

        Raises:
            AttributeError: Always.
        """
        raise AttributeError(f"an allowlisted name has no attribute {name} to set")

    def __call__(self, *args: object) -> Any:
        """Call the wrapped function, as REDUCE does, with positional arguments.

        Args:
            *args: The arguments the pickle gives.

        Returns:
            What the function returns.
        """
        return self._function(*args)


class _PickledArray(np.ndarray[Any, np.dtype[Any]]):
    """The array a pickle builds, whose state is checked before NumPy sees it.

    NumPy fills an unpickled array through ``ndarray.__setstate__``, which
    BUILD reaches without any lookup, and which does not check an object
    array's list of elements against its shape: ``(1, (), dtype('O'), False,
    [])`` crashed the interpreter. So the arrays this module builds are of
    this class, whose ``__setstate__`` accepts only what NumPy writes.
    :func:`pixels_from_object` hands on a plain ``ndarray`` view of one.
    """

    def __setstate__(self, state: object) -> None:
        """Check a pickled array state, then let NumPy apply it.

        Args:
            state: ``(version, shape, dtype, is_fortran, data)``, as NumPy
                writes it, or the same without the version, as it once did.
                ``data`` holds exactly the shape's worth of elements: bytes
                for a number type, a list for ``object``.

        Raises:
            pickle.UnpicklingError: If the state is anything else.
        """
        parts = state if isinstance(state, tuple) else ()
        if len(parts) == 5 and type(parts[0]) is int and parts[0] in (0, 1):
            parts = parts[1:]
        if len(parts) != 4:
            raise pickle.UnpicklingError("an array state NumPy would not have written")
        shape, dtype, fortran, data = parts
        if not (
            isinstance(shape, tuple)
            and all(type(n) is int and n >= 0 for n in shape)
            and isinstance(dtype, np.dtype)
            and isinstance(fortran, int)
        ):
            raise pickle.UnpicklingError("an array state NumPy would not have written")
        count = math.prod(shape)
        if dtype.hasobject:
            fits = type(data) is list and len(data) == count
        else:
            fits = type(data) is bytes and len(data) == count * dtype.itemsize
        if not fits:
            raise pickle.UnpicklingError(
                f"an array state whose data does not fill its shape {shape}"
            )
        super().__setstate__(cast(Any, state))


#: NumPy's own reconstruction functions, taken from what NumPy itself emits so
#: that no private module path is imported here.
_REDUCED: Any = np.zeros(1, dtype=np.uint8).__reduce__()
_REDUCED_5: Any = np.zeros(1, dtype=np.uint8).__reduce_ex__(5)
_REDUCED_SCALAR: Any = np.uint8(0).__reduce__()
_NUMPY_RECONSTRUCT: Callable[..., Any] = _REDUCED[0]
_NUMPY_FROMBUFFER: Callable[..., Any] | None = (
    _REDUCED_5[0] if getattr(_REDUCED_5[0], "__name__", "") == "_frombuffer" else None
)
_NUMPY_SCALAR: Callable[..., Any] = _REDUCED_SCALAR[0]


class _Unsupported(pickle.UnpicklingError):
    """A pickle holds something well formed that this reader does not read."""


class _RefusedGlobal(_Unsupported):
    """A pickle referred to a name outside the allowlist."""


def _reconstruct(subtype: object, shape: object, dtype: object) -> Any:
    """Build the empty array NumPy's pickles start from, and only that.

    Args:
        subtype: Must be the token ``numpy.ndarray`` resolves to here.
        shape: Must be ``(0,)``, which is what NumPy always writes: the real
            shape arrives with the data, where NumPy checks one against the
            other.
        dtype: Must be ``b"b"``, the placeholder NumPy always writes (``"b"``
            from a pickle of Python 2's day): the real dtype arrives with the
            data as well.

    Returns:
        An empty :class:`_PickledArray` for the unpickler to fill.

    Raises:
        pickle.UnpicklingError: If the call is not the one NumPy writes. A
            larger shape would be an allocation the file has not paid for.
    """
    if subtype is not _NDARRAY or shape != (0,) or dtype not in (b"b", "b"):
        raise pickle.UnpicklingError("an array reconstruction that NumPy would not have written")
    return _NUMPY_RECONSTRUCT(_PickledArray, (0,), dtype)


def _frombuffer(numpy_frombuffer: Callable[..., Any], *args: object) -> Any:
    """Stand in for NumPy's ``_frombuffer``, which protocol 5 builds arrays with.

    Args:
        numpy_frombuffer: NumPy's function, bound when it is allowlisted.
        *args: What NumPy writes: the buffer, the dtype, the shape and the
            memory order, and in NumPy 2 the axis order.

    Returns:
        The array NumPy's function builds, as a :class:`_PickledArray`, so a
        BUILD on it is checked like any other.
    """
    return numpy_frombuffer(*args).view(_PickledArray)


def _dtype(spec: object, align: object = False, copy: object = False) -> np.dtype[Any]:
    """Resolve a pickled dtype to the instance NumPy shares, and only to one.

    NumPy pickles a dtype as a call, ``dtype(spec, align, copy)``, then a
    BUILD that hands the result its byte order and flags. With ``copy`` true,
    as NumPy writes it, the call makes a new dtype, and NumPy's
    ``__setstate__`` on a new one is not memory-safe: a state two items short
    crashed the interpreter (#54). BUILD goes through no lookup, so the state
    cannot be checked; what can be chosen is the object it lands on. NumPy
    keeps one instance of each built-in type, whose ``__setstate__`` returns
    without reading its argument, and that is what this returns whatever
    ``copy`` says. ``isbuiltin == 1`` is NumPy's own test for such an
    instance; a text, structured, datetime or non-native type is not one and
    is refused.

    So a pickled byte order is ignored: every type is in this machine's.
    Nothing changes for ``uint8``, the one type an image may have, whose
    samples have no byte order. A multi-byte NumPy integer pickled on a
    machine of the other byte order reads as a large number, and is refused as a
    sample rather than taken for another value in 0-255.

    Args:
        spec: The type's name, such as ``"u1"``.
        align: Ignored; it only matters for structured types.
        copy: Ignored, for the reason above.

    Returns:
        NumPy's shared dtype for ``spec``.

    Raises:
        pickle.UnpicklingError: If ``spec`` is not a string.
        _Unsupported: If it names anything but a built-in number or
            ``object`` type.
    """
    if not isinstance(spec, str):
        raise pickle.UnpicklingError(f"a NumPy dtype named by {_brief(spec)}")
    dtype = np.dtype(spec)
    if dtype.isbuiltin != 1:
        raise _Unsupported(f"it holds the NumPy dtype {dtype.str}; only plain numbers are read")
    return dtype


def _encode(text: str, encoding: str = "latin1") -> bytes:
    """Stand in for ``_codecs.encode``, which protocols 0-2 use to carry bytes.

    Args:
        text: The bytes, as the latin-1 string those protocols store.
        encoding: Must be ``"latin1"``, the only one pickle writes.

    Returns:
        The bytes.

    Raises:
        pickle.UnpicklingError: If another encoding is asked for.
    """
    if encoding != "latin1":
        raise pickle.UnpicklingError(
            f"bytes encoded as {_brief(encoding)}, which pickle never writes"
        )
    return text.encode("latin1")


def _allowed_globals() -> dict[tuple[str, str], object]:
    """Build the table of every name a pickle may refer to.

    NumPy 1 pickles say ``numpy.core`` and NumPy 2 pickles ``numpy._core``.
    Both spellings map to the installed NumPy's functions, so a file loads
    whichever NumPy wrote it, which ``numpy.load`` itself does not manage.
    Every function is :class:`_Sealed`, so a BUILD cannot change it for the
    loads that follow.

    Returns:
        ``(module, name)`` to the object handed to the unpickler in its place.
    """
    allowed: dict[tuple[str, str], object] = {
        ("numpy", "ndarray"): _NDARRAY,
        ("numpy", "dtype"): _Sealed(_dtype),
        ("_codecs", "encode"): _Sealed(_encode),
    }
    for root in ("numpy.core", "numpy._core"):
        allowed[(f"{root}.multiarray", "_reconstruct")] = _Sealed(_reconstruct)
        allowed[(f"{root}.multiarray", "scalar")] = _Sealed(_NUMPY_SCALAR)
        if _NUMPY_FROMBUFFER is not None:
            allowed[(f"{root}.numeric", "_frombuffer")] = _Sealed(
                partial(_frombuffer, _NUMPY_FROMBUFFER)
            )
    return allowed


_ALLOWED = _allowed_globals()


class _RestrictedUnpickler(pickle.Unpickler):
    """An unpickler that resolves names from the allowlist and nowhere else."""

    def find_class(self, module_name: str, global_name: str) -> Any:
        """Resolve a name the pickle refers to, without importing anything.

        Args:
            module_name: The module the pickle names.
            global_name: The attribute it names in that module.

        Returns:
            The allowlisted stand-in.

        Raises:
            _RefusedGlobal: If the name is not allowlisted. The default
                implementation would import the module and return whatever it
                found, which is how a pickle runs code.
        """
        try:
            return _ALLOWED[(module_name, global_name)]
        except KeyError:
            raise _RefusedGlobal(f"{module_name}.{global_name}") from None


# --- the opcode stream, walked before the machine runs it ---------------------

#: Opcodes whose argument has a fixed width and sizes nothing, to that width.
#: A pickle of pixels is made of these almost entirely.
_PLAIN_OPCODES: dict[int, int] = {
    **dict.fromkeys(
        b"()012NQR]abdelostu}\x81\x85\x86\x87\x88\x89\x8f\x90\x91\x92\x93\x94\x97\x98", 0
    ),
    **dict.fromkeys(b"Khq\x80\x82", 1),
    **dict.fromkeys(b"M\x83", 2),
    **dict.fromkeys(b"Jj\x84", 4),
    **dict.fromkeys(b"G", 8),
}

_STOP = ord(".")
_FRAME = 0x95
_LONG_BINPUT = ord("r")
_PUT = ord("p")

#: Opcodes followed by a number, to their name, the number's width and
#: whether it is signed. The number is a length for all but two: FRAME's
#: length covers opcodes, walked like any others, and LONG_BINPUT's number
#: is a memo index.
_FIELD_OPCODES: dict[int, tuple[str, int, bool]] = {
    ord("U"): ("SHORT_BINSTRING", 1, False),
    ord("C"): ("SHORT_BINBYTES", 1, False),
    0x8C: ("SHORT_BINUNICODE", 1, False),
    0x8A: ("LONG1", 1, False),
    ord("T"): ("BINSTRING", 4, True),
    ord("B"): ("BINBYTES", 4, False),
    ord("X"): ("BINUNICODE", 4, False),
    0x8B: ("LONG4", 4, True),
    0x8E: ("BINBYTES8", 8, False),
    0x8D: ("BINUNICODE8", 8, False),
    0x96: ("BYTEARRAY8", 8, False),
    _FRAME: ("FRAME", 8, False),
    _LONG_BINPUT: ("LONG_BINPUT", 4, False),
}

#: Opcodes followed by lines of text, to how many. PUT's line is a memo index.
_LINE_OPCODES: dict[int, int] = {**dict.fromkeys(b"ILFSVgPp", 1), **dict.fromkeys(b"ci", 2)}

#: The plain opcodes as pattern branches, by the width of their argument.
#: BININT1, a sample, is most of a pickle of pixels, so one byte comes first.
_PLAIN_BRANCHES = [
    b"["
    + re.escape(bytes(sorted(code for code, n in _PLAIN_OPCODES.items() if n == width)))
    + b"]"
    + b"." * width
    for width in (1, 0, 2, 4, 8)
]

#: The one-line opcodes but PUT: a number, a string or a memo index to read,
#: none of which sizes anything but itself.
_TEXT_BRANCH = rb"[ILFSVgP][^\n]*\n"

#: The most opcodes one match takes. It bounds what Python 3.10's matcher,
#: which has no possessive repeat, keeps to backtrack into.
_RUN_LENGTH = 4096

#: From Python 3.11 the repeat is possessive. A first byte decides every
#: opcode's length, so there is never anything to backtrack into.
_POSSESSIVE = b"+" if sys.version_info >= (3, 11) else b""


def _below(bound: int) -> bytes:
    """Build a pattern for four bytes that, little-endian, count below ``bound``.

    Args:
        bound: The limit, which itself does not match.

    Returns:
        The pattern: any four bytes when ``bound`` is past 32 bits, and
        nothing at all when it is 0.
    """
    if bound >= 2**32:
        return b"...."
    digits = bound.to_bytes(4, "little")
    # Lower than the bound's byte at one place, equal to it above that, and
    # anything below.
    branches = [
        b"." * place
        + b"[\\x00-\\x%02x]" % (digits[place] - 1)
        + b"".join(b"\\x%02x" % digit for digit in digits[place + 1 :])
        for place in range(4)
        if digits[place]
    ]
    return b"(?:" + b"|".join(branches) + b")" if branches else b"(?!)"


def _quiet_run(size: int) -> re.Pattern[bytes]:
    """Build the pattern for a run of opcodes that need no check in ``size`` bytes.

    Those are matched in C: a pickle of pixels is millions of opcodes, and a
    loop over them in Python would double its load time. Besides the plain
    opcodes and the text lines, that includes a memo index already below
    ``size``: protocols 1 to 3 give every tuple past the 256th a LONG_BINPUT,
    and protocol 0 a PUT, and walking those one at a time took four seconds
    for 2000x2000 pixels. A PUT with fewer digits than ``size`` has is below
    it. :func:`_check_opcodes` handles the rest one opcode at a time.

    The first branch is a whole pixel as protocols 2 to 5 write one, three
    BININT1, a TUPLE3 and the tuple memoized, because each repetition costs
    the matcher more than the bytes it reads: it walks the pickle this
    package writes in 0.16 s for 2000x2000 pixels rather than 0.5 s.

    Args:
        size: The length of the pickle.

    Returns:
        The compiled pattern, which ``re`` caches by its text.
    """
    memo = b"r" + _below(size)
    branches = [
        rb"K.K.K.\x87(?:\x94|q.|" + memo + b")",
        rb"K.K.K.",
        _TEXT_BRANCH,
        *_PLAIN_BRANCHES,
        memo,
    ]
    digits = len(str(size)) - 1
    if digits:
        branches.append(rb"p\d{1,%d}\n" % digits)
    return re.compile(
        b"(?:" + b"|".join(branches) + b"){0,%d}" % _RUN_LENGTH + _POSSESSIVE, re.DOTALL
    )


def _ends_within(stop: int, limit: int, end: int, pos: int, label: str) -> bool:
    """Tell whether the opcode at ``pos``, ending at ``stop``, fits where it is.

    Args:
        stop: Where the opcode, or the part of it read so far, ends.
        limit: Where its frame ends, or the data when it is in none.
        end: Where the data ends.
        pos: Where the opcode starts, for the message.
        label: What to call the source in a message.

    Returns:
        ``True`` if it fits, ``False`` if the data ends first: the unpickler
        stops at the same place and reports the truncation itself.

    Raises:
        UnsupportedFileFormatError: If it runs past the end of a frame.
    """
    if stop <= limit:
        return True
    if limit == end:
        return False
    raise UnsupportedFileFormatError(
        f"not a readable {label}: the opcode at byte {pos} runs past the end of its frame"
    )


def _check_memo_index(index: int, name: str, pos: int, end: int, label: str) -> None:
    """Refuse a memo index the pickle is too short to have objects for.

    Args:
        index: The index, or ``-1`` for a PUT whose line is not a number.
        name: The opcode, for the message.
        pos: Where it starts, for the message.
        end: The length of the pickle.
        label: What to call the source in a message.

    Raises:
        UnsupportedFileFormatError: If the index is negative or not below
            ``end``.
    """
    if not 0 <= index < end:
        raise UnsupportedFileFormatError(
            f"not a readable {label}: the {name} at byte {pos} names a memo slot outside "
            f"what a pickle of {end} bytes can fill"
        )


def _check_opcodes(data: bytes, label: str) -> None:
    """Walk the opcodes the unpickler will run, before it runs any.

    CPython's unpickler allocates by the numbers in its opcodes before it
    reads what they describe (#54). BINBYTES, BINBYTES8 and BYTEARRAY8
    reserve their declared length first, so 12 bytes asked for 256 TiB and
    ended in a bare ``MemoryError``. LONG_BINPUT and PUT grow the memo, an
    array it writes in full, to twice the index they name, so 9 bytes wrote
    256 MiB. Neither passes through anything a subclass can override, so the
    stream is checked first: every length must fit in the bytes that follow
    it, and every memo index must be below the length of the pickle. Each
    memo slot costs an opcode byte to fill, so the memo stays within what
    MEMOIZE, one byte a slot, could make of the same file anyway.

    The walk must see the opcodes the unpickler sees, and the one place the
    two could part is a frame. The unpickler reads a frame whole and parses
    that buffer; an opcode running past the buffer's end has the rest of it
    dropped and is read on from the file, at an offset this walk would not
    share. No pickler writes one, and the pure-Python unpickler refuses one
    too, so it is refused.

    A stream that ends early, mid-opcode or without a STOP, is left to the
    unpickler, which stops at the same place and says so. Data after the
    STOP is refused here. So is a byte that is no opcode this walk knows,
    since passing it on would let a future opcode through unchecked.

    Args:
        data: The pickle.
        label: What to call the source in a message.

    Raises:
        UnsupportedFileFormatError: If an opcode declares more bytes than
            follow it, names a memo slot the pickle is too short to fill,
            runs past the end of its frame or is not an opcode at all, or if
            data follows the STOP.
    """
    end = len(data)
    limit = end
    pos = 0
    run = _quiet_run(end)
    while True:
        # A repeat from zero matches anywhere, so there always is a match.
        pos = cast("re.Match[bytes]", run.match(data, pos, limit)).end()
        if pos == limit:
            if limit == end:
                return
            limit = end  # the frame is used up
            continue
        code, start = data[pos], pos + 1
        if code == _STOP:
            if start != end:
                raise UnsupportedFileFormatError(
                    f"unsupported {label}: data follows the first pickle"
                )
            return
        # Past this point the run stopped at its length, at an opcode that
        # runs past the limit, or at one it leaves to be checked here.
        if code in _PLAIN_OPCODES:
            stop = start + _PLAIN_OPCODES[code]
        elif code in _LINE_OPCODES:
            stop = start
            for _ in range(_LINE_OPCODES[code]):
                newline = data.find(b"\n", stop, limit)
                stop = newline + 1 if newline >= 0 else limit + 1
            if _ends_within(stop, limit, end, pos, label) and code == _PUT:
                try:
                    index = int(data[start : stop - 1])
                except ValueError:
                    index = -1
                _check_memo_index(index, "PUT", pos, end, label)
        elif code in _FIELD_OPCODES:
            name, width, signed = _FIELD_OPCODES[code]
            stop = start + width
            if not _ends_within(stop, limit, end, pos, label):
                return
            number = int.from_bytes(data[start:stop], "little", signed=signed)
            if code == _LONG_BINPUT:
                _check_memo_index(number, name, pos, end, label)
            elif not 0 <= number <= limit - stop:
                raise UnsupportedFileFormatError(
                    f"not a readable {label}: the {name} at byte {pos} declares {number} "
                    f"bytes, and only {limit - stop} follow"
                )
            elif code == _FRAME:
                # Nested in another frame, it is read from that one's buffer.
                limit = stop + number if limit == end else limit
            else:
                stop += number
        else:
            raise UnsupportedFileFormatError(
                f"not a readable {label}: byte {pos} is 0x{code:02x}, which is not a pickle opcode"
            )
        if not _ends_within(stop, limit, end, pos, label):
            return
        pos = stop


def safe_loads(data: bytes, label: str = "pickle") -> object:
    """Unpickle ``data`` without executing anything in it.

    The opcodes are walked first (:func:`_check_opcodes`), so nothing the
    file declares is allocated before it is known to fit in the file.

    Args:
        data: Exactly one pickle.
        label: What to call the source in a message.

    Returns:
        The object: built from lists, tuples, dicts, numbers, strings, bytes
        and NumPy arrays and scalars, because nothing else can be. An array
        is an ``ndarray`` of a private subclass that checks its pickled
        state; ``.view(numpy.ndarray)`` makes a plain one.

    Raises:
        UnsupportedFileFormatError: If the data refers to any name outside
            the allowlist, declares more than it holds, is not a well-formed
            pickle, or is followed by further data. ``MemoryError`` is
            deliberately not converted: it means the machine ran out of
            memory, not that the file is bad.
    """
    _check_opcodes(data, label)
    try:
        loaded: object = _RestrictedUnpickler(io.BytesIO(data)).load()
    except MemoryError:
        raise
    except _RefusedGlobal as refused:
        raise UnsupportedFileFormatError(
            f"unsupported {label}: it refers to {refused}; only lists, tuples, integers "
            f"and NumPy arrays are read, and nothing in a pickle is ever executed"
        ) from None
    except _Unsupported as refused:
        raise UnsupportedFileFormatError(f"unsupported {label}: {refused}") from None
    except Exception as error:
        # The pickle documentation lists AttributeError, EOFError, ImportError
        # and IndexError and says the list is not exhaustive: a malformed
        # pickle can raise nearly anything from inside the machine.
        raise UnsupportedFileFormatError(
            f"not a readable {label}: {type(error).__name__}: {error}"
        ) from None
    return loaded


def _is_sequence(value: object) -> bool:
    """Tell whether ``value`` is a list or a tuple, the two pixel containers.

    Args:
        value: Anything.

    Returns:
        ``True`` for a list or tuple. Strings and bytes are sized and
        iterable too, which is exactly why this is not a duck-typed check.
    """
    return isinstance(value, (list, tuple))


def _brief(value: object) -> str:
    """Show a value from a pickle in a message, at a length the file cannot choose.

    ``repr`` will not do (#58). A pickle may hold a tuple nested 20,000 deep,
    which the unpickler builds without recursing and ``repr`` cannot print,
    and ``repr`` expands shared substructure, so a 226-byte file asked for
    335 MB of message.

    Args:
        value: Anything a pickle can hold.

    Returns:
        The value as written when it is a number, ``None`` or a short string,
        and otherwise its type.
    """
    if value is None or isinstance(value, (bool, float, np.bool_, np.number)):
        return repr(value)
    if isinstance(value, int):
        bits = value.bit_length()
        return str(value) if bits <= 64 else f"a {bits}-bit int"
    if isinstance(value, str) and len(value) <= _BRIEF_TEXT:
        return repr(value)
    return f"a {type(value).__name__}"


def _check_size(width: object, height: object, label: str) -> tuple[int, int]:
    """Validate a declared size.

    Args:
        width: The declared width.
        height: The declared height.
        label: What to call the source in a message.

    Returns:
        The ``(width, height)`` pair.

    Raises:
        UnsupportedFileFormatError: If either is not a positive integer, or
            is one past 64 bits, which no list of pixels could match and
            which a message could not even print.
    """
    sizes: list[int] = []
    for name, value in (("width", width), ("height", height)):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or not 1 <= value < 2**64
        ):
            raise UnsupportedFileFormatError(
                f"invalid {label} {name}: expected a positive integer, found {_brief(value)}"
            )
        sizes.append(int(value))
    return sizes[0], sizes[1]


def _samples(flat: Iterable[Any], count: int, label: str) -> np.ndarray:
    """Turn ``count`` samples into a ``uint8`` vector, refusing to coerce.

    Args:
        flat: The samples, in order. Walked twice, so it must be
            re-iterable: a list, or a :class:`_Reiterable`, never a bare
            ``chain``.
        count: How many there are.
        label: What to call the source in a message.

    Returns:
        The samples as ``uint8``.

    Raises:
        UnsupportedFileFormatError: If any sample is not an integer (a float
            could be scaled 0-1 or 0-255, and a bool is a logic error), or
            falls outside 0-255, where a cast would silently wrap.
    """
    kinds = set(map(type, flat))
    wrong = sorted(
        kind.__name__
        for kind in kinds
        if kind is bool or not (kind is int or issubclass(kind, np.integer))
    )
    if wrong:
        raise UnsupportedFileFormatError(
            f"unsupported {label} samples of type {', '.join(wrong)}: expected integers in 0-255"
        )
    try:
        values = np.fromiter(flat, dtype=np.int64, count=count)
    except OverflowError:
        # A sample past 64 bits, a Python int or a numpy.uint64 from 2**63,
        # fails the conversion before the range check below can name it (#58).
        raise UnsupportedFileFormatError(
            f"unsupported {label} sample beyond 64 bits: expected integers in 0-255"
        ) from None
    low, high = int(values.min()), int(values.max())
    if low < 0 or high > _MAX_SAMPLE:
        raise UnsupportedFileFormatError(
            f"unsupported {label} sample {low if low < 0 else high}: expected integers in 0-255"
        )
    return values.astype(IMAGE_DTYPE)


def _uniform_length(items: Iterable[Any], what: str, label: str) -> int:
    """Return the one length every item has.

    Args:
        items: Lists or tuples. Walked twice, so it must be re-iterable.
        what: What the items are, for the message: ``"row"`` or ``"pixel"``.
        label: What to call the source in a message.

    Returns:
        The shared length.

    Raises:
        UnsupportedFileFormatError: If an item is not a list or tuple, or the
            lengths differ, or the length is zero.
    """
    kinds = set(map(type, items))
    if not all(issubclass(kind, (list, tuple)) for kind in kinds):
        names = ", ".join(sorted(kind.__name__ for kind in kinds))
        raise UnsupportedFileFormatError(
            f"unsupported {label}: every {what} must be a list or tuple, found {names}"
        )
    lengths = set(map(len, items))
    if len(lengths) != 1 or 0 in lengths:
        raise UnsupportedFileFormatError(
            f"unsupported {label}: every {what} must have the same, non-zero length, "
            f"found {sorted(lengths)}"
        )
    return lengths.pop()


def _check_shape(
    shape: tuple[int, int, int],
    source_bytes: int | None,
    label: str,
    size_check: SizeCheck | None,
) -> None:
    """Refuse a picture no image has, or its file cannot hold, before it is walked.

    A pickle stores a list as references, and the same row may be referenced
    by every row of the picture, the same pixel by every pixel of that row:
    ``[[(1, 2, 3)] * k] * k`` is ``4 * k`` bytes and ``k * k`` pixels (#54).
    16 KB described 4000x4000 pixels, which took 4 s and 570 MiB to walk,
    and the growth is quadratic, so a few hundred kilobytes never finished.
    One tuple as every pixel makes it cubic: 5 KB cost 10 s and 1.6 GiB
    before its 2000 channels were refused. The bound is
    :data:`_MAX_SAMPLES_PER_BYTE`, and it is applied to the size the lists
    declare, which costs nothing to learn.

    A small flat-colour canvas built that way still loads: up to about 1370
    pixels square. A larger picture is pickled as a NumPy array, whose bytes
    are in the file, or as rows that are lists of their own.

    Last, the caller's own check, from :meth:`RasterImage.set_size_check`:
    this is the pickle's equivalent of a header, the first point where its
    size is known and nothing has been walked (0.5.8, #81).

    Args:
        shape: ``(height, width, channels)``, as the lists' lengths give it.
        source_bytes: The length of the pickle, or ``None`` for an object
            that did not come from one, which has no bound.
        label: What to call the source in a message.
        size_check: Called with ``(width, height)`` once the picture is
            known to be one this reader takes; ``None`` for none.

    Raises:
        UnsupportedFileFormatError: If the channel count is not one the
            array rules accept, or the shape holds more samples than
            ``source_bytes`` allows.
        Exception: Whatever ``size_check`` raises to refuse the picture.
    """
    validate_image_array(shape, IMAGE_DTYPE, label)
    height, width, channels = shape
    if source_bytes is not None and height * width * channels > (
        source_bytes * _MAX_SAMPLES_PER_BYTE
    ):
        raise UnsupportedFileFormatError(
            f"unsupported {label}: its {source_bytes} bytes describe {width}x{height} pixels "
            f"of {channels} samples, which only lists that repeat one row or pixel can; at "
            f"most {_MAX_SAMPLES_PER_BYTE} samples a byte are read. Pickle the picture as "
            f"a NumPy array, or as rows that are separate lists"
        )
    if size_check is not None:
        size_check(width, height)


def _array_from_sequence(
    pixels: Sequence[Any],
    declared: tuple[int, int] | None,
    label: str,
    source_bytes: int | None,
    size_check: SizeCheck | None,
) -> np.ndarray:
    """Build the ``(height, width, channels)`` array a list of pixels describes.

    A list of tuples must never go through ``np.asarray``, which is slower
    than per-pixel Python; it is flattened by ``itertools.chain`` into
    ``np.fromiter`` with an explicit count, as ``RasterImage.set_raw_data``
    does.

    The lists' own lengths give the shape, and it is checked, against the
    array rules and against the size of the pickle, before anything walks
    the pixels: every walk below is then over a picture that may be read.

    Args:
        pixels: Rows of pixels, or a flat list of pixels.
        declared: The ``(width, height)`` declared for it, if any.
        label: What to call the source in a message.
        source_bytes: The length of the pickle the lists came from, which
            bounds the picture they may describe; ``None`` for no bound.
        size_check: The caller's check of ``(width, height)``, put before
            the walk; ``None`` for none.

    Returns:
        The array, ``uint8``, of a shape the array rules accept.

    Raises:
        UnsupportedFileFormatError: If the structure is ragged or not pixels
            at all, a flat list has no declared size or one that does not
            match its length, rows disagree with a declared size, the
            channel count is not one an image has, or the picture is larger
            than its pickle can hold.
        Exception: Whatever ``size_check`` raises to refuse the picture.
    """
    if len(pixels) == 0:
        raise UnsupportedFileFormatError(f"unsupported {label}: the pixel list is empty")
    first = pixels[0]
    if not _is_sequence(first) or len(first) == 0:
        raise UnsupportedFileFormatError(
            f"unsupported {label}: expected a list of (r, g, b) pixels or a list of rows "
            f"of them, found a list of {type(first).__name__}"
        )

    if _is_sequence(first[0]):
        height = len(pixels)
        width = _uniform_length(pixels, "row", label)
        # By the first pixel, which the walk below then holds every other to.
        _check_shape((height, width, len(first[0])), source_bytes, label, size_check)
        channels = _uniform_length(_Reiterable(lambda: chain.from_iterable(pixels)), "pixel", label)
        samples = _samples(
            _Reiterable(lambda: chain.from_iterable(chain.from_iterable(pixels))),
            height * width * channels,
            label,
        )
        return samples.reshape(height, width, channels)

    count = len(pixels)
    channels = _uniform_length(pixels, "pixel", label)
    if declared is None:
        raise UnsupportedFileFormatError(
            f"unsupported {label}: a flat list of {count} pixels does not say how wide the "
            f"picture is. Nest the pixels in rows, pickle {{'width': w, 'height': h, "
            f"'pixels': [...]}} instead, or declare the size with --width and --height "
            f"(Codec.with_input_size)"
        )
    width, height = declared
    if width * height != count:
        raise UnsupportedFileFormatError(
            f"unsupported {label}: {count} pixels cannot fill the declared {width}x{height} "
            f"({width * height} pixels)"
        )
    _check_shape((height, width, channels), source_bytes, label, size_check)
    samples = _samples(_Reiterable(lambda: chain.from_iterable(pixels)), count * channels, label)
    return samples.reshape(height, width, channels)


class _Reiterable:
    """An iterable that can be walked more than once, from a factory.

    ``itertools.chain`` is a one-shot iterator, and the samples are walked
    twice: once to check their types, once to build the array.
    """

    def __init__(self, factory: Callable[[], Iterator[Any]]) -> None:
        """Remember how to start a fresh walk.

        Args:
            factory: Returns a new iterator over the same items each call.
        """
        self._factory = factory

    def __iter__(self) -> Iterator[Any]:
        """Start a fresh walk.

        Returns:
            A new iterator over the items.
        """
        return self._factory()


def pixels_from_object(
    loaded: object,
    declared: tuple[int, int] | None = None,
    label: str = "pickle",
    source_bytes: int | None = None,
    size_check: SizeCheck | None = None,
) -> np.ndarray:
    """Turn an unpickled object into the ``(height, width, 3)`` RGB array.

    Args:
        loaded: What the pickle held: an array, rows of pixels, a flat list
            of pixels, or the ``{"width", "height", "pixels"}`` wrapper
            around any of them.
        declared: The ``(width, height)`` the caller declared, if any. It
            gives a flat list its size, and anything that carries its own
            size must agree with it.
        label: What to call the source in a message.
        source_bytes: The length of the pickle ``loaded`` came from. Lists
            may then describe at most 1032 samples for each of its bytes,
            since lists in a pickle can repeat one row without storing it
            again; see :func:`_check_shape`. ``None``, for an object of the
            caller's own making, sets no bound.
        size_check: Called with the picture's ``(width, height)`` once it is
            known to be one this reader takes and before a list of pixels is
            walked, as :meth:`RasterImage.set_size_check` describes; ``None``
            for none.

    Returns:
        The RGB array, ``uint8`` and C-contiguous.

    Raises:
        UnsupportedFileFormatError: If the object is none of those, breaks
            the rules of the one it is, contradicts a declared size, or
            describes more than ``source_bytes`` can hold.
        Exception: Whatever ``size_check`` raises to refuse the picture.
    """
    if isinstance(loaded, dict):
        if set(loaded) != _WRAPPER_KEYS:
            shown = sorted(map(_brief, loaded))
            more = len(shown) - _BRIEF_KEYS
            keys = ", ".join(shown[:_BRIEF_KEYS]) + (f" and {more} more" if more > 0 else "")
            raise UnsupportedFileFormatError(
                f"unsupported {label}: a dict must have exactly the keys 'width', 'height' "
                f"and 'pixels', found {keys or 'none'}"
            )
        wrapped = _check_size(loaded["width"], loaded["height"], label)
        if declared is not None and declared != wrapped:
            raise UnsupportedFileFormatError(
                f"the {label} declares {wrapped[0]}x{wrapped[1]}, not the "
                f"{declared[0]}x{declared[1]} asked for"
            )
        if isinstance(loaded["pixels"], dict):
            raise UnsupportedFileFormatError(f"unsupported {label}: 'pixels' is another dict")
        return pixels_from_object(loaded["pixels"], wrapped, label, source_bytes, size_check)

    if isinstance(loaded, np.ndarray) and loaded.dtype.hasobject:
        loaded = loaded.tolist()

    if isinstance(loaded, np.ndarray):
        validate_image_array(loaded.shape, loaded.dtype, label)
        # Its bytes were in the pickle, so it is no larger than the file; the
        # check still comes before to_rgb, which may triple a greyscale one.
        if size_check is not None:
            size_check(loaded.shape[1], loaded.shape[0])
        # A plain ndarray, not the _PickledArray unpickling made.
        array = loaded.view(np.ndarray)
    elif isinstance(loaded, (list, tuple)):
        array = _array_from_sequence(loaded, declared, label, source_bytes, size_check)
    else:
        raise UnsupportedFileFormatError(
            f"unsupported {label}: it holds a {type(loaded).__name__}, not an image; expected "
            f"a NumPy array, rows of (r, g, b) pixels, or a flat list of them with its size"
        )

    height, width = array.shape[0], array.shape[1]
    if declared is not None and declared != (width, height):
        raise UnsupportedFileFormatError(
            f"the {label} holds a {width}x{height} picture, not the "
            f"{declared[0]}x{declared[1]} declared"
        )
    return to_rgb(array, label)


class PickleImage(RasterImage):
    """An image stored as a pickle of an array or of lists of pixels.

    See the module documentation for what a file may hold and for why loading
    one cannot execute anything.
    """

    def load(self, filename: FileSource) -> None:
        """Read a pickle from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from ``sys.stdin``.
                Reads forward only, so a pipe works.

        Raises:
            UnsupportedFileFormatError: If the file is not a pickle, refers
                to anything outside the allowlist, or does not hold an image
                under the rules in the module documentation. A flat list of
                pixels needs a size from :meth:`declare_size` or from the
                wrapper dict.
            OSError: If the file cannot be read.
        """
        with open_binary_read(filename) as file:
            # The whole file, whose size is its own and not a field's.
            data = file.read()
        loaded = safe_loads(data)
        self.set_array(
            pixels_from_object(
                loaded,
                self._declared_size,
                source_bytes=len(data),
                size_check=self._check_dimensions,
            )
        )
        log.debug("loaded pickle %dx%d from %s", self._width, self._height, filename)

    def save(self, filename: FileSource) -> None:
        """Write this image to ``filename`` as rows of ``(r, g, b)`` tuples.

        Plain ``int`` samples at protocol 4: any Python loads it without
        NumPy, and the bytes do not depend on which NumPy is installed, which
        a pickled array's do.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            ValueError: If the pixel count does not match the dimensions.
            OSError: If the file cannot be written.
        """
        rows = [list(map(tuple, row)) for row in self.get_array().tolist()]
        with open_binary_write(filename) as file:
            pickle.dump(rows, file, protocol=PICKLE_PROTOCOL)
