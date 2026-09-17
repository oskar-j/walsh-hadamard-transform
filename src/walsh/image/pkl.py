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

What may be inside, since a pickle can hold anything:

* a NumPy array, under the rules every array reader shares (``uint8``;
  ``(h, w, 3)`` RGB, ``(h, w)`` or ``(h, w, 1)`` greyscale, ``(h, w, 4)``
  RGBA only when fully opaque); pickles written under NumPy 1 and NumPy 2
  both load, whichever is installed;
* **rows of pixels**, ``[[(r, g, b), ...], ...]``, which carry their own size;
* a **flat list of pixels**, ``[(r, g, b), ...]``, top row first. That does
  not say how wide the picture is, and nothing here guesses, so the size must
  be declared: by pickling ``{"width": w, "height": h, "pixels": [...]}``
  instead, or with ``--width`` / ``--height`` (``Task.with_input_size``).

Lists and tuples are interchangeable at every level, samples must be integers
in 0-255 (Python's or NumPy's, never floats or booleans), and ragged rows are
refused. The writer produces rows of ``(r, g, b)`` tuples of plain ``int`` at
protocol 4: that needs no NumPy to load and its bytes do not depend on which
NumPy wrote it, unlike a pickled array.
"""

from __future__ import annotations

import io
import logging
import pickle
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import chain
from typing import Any

import numpy as np

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._arrays import IMAGE_DTYPE, to_rgb, validate_image_array
from walsh.image._io import FileSource, open_binary_read, open_binary_write
from walsh.image.base import RasterImage

__all__ = ["PICKLE_PROTOCOL", "PickleImage", "pixels_from_object", "safe_loads"]

log = logging.getLogger(__name__)

#: The protocol written. 4 is the newest every supported Python reads and
#: writes identically; 5 adds only out-of-band buffers, which a file cannot use.
PICKLE_PROTOCOL = 4

#: The keys of the wrapper that gives a flat pixel list its size.
_WRAPPER_KEYS = frozenset({"width", "height", "pixels"})

_MAX_SAMPLE = 255


class _NdarrayToken:
    """Stands in for ``numpy.ndarray`` while unpickling.

    NumPy's pickles mention the class only to pass it to ``_reconstruct``.
    Resolving the name to this inert object instead means a hostile pickle
    cannot call ``ndarray(shape)`` and have it allocate whatever it likes.
    """


_NDARRAY = _NdarrayToken()

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


def _reconstruct(subtype: object, shape: object, dtype: object) -> Any:
    """Build the empty array NumPy's pickles start from, and only that.

    Args:
        subtype: Must be the token ``numpy.ndarray`` resolves to here.
        shape: Must be ``(0,)``, which is what NumPy always writes: the real
            shape arrives with the data, where NumPy checks one against the
            other.
        dtype: The dtype, passed through.

    Returns:
        An empty array for the unpickler to fill.

    Raises:
        pickle.UnpicklingError: If the call is not the one NumPy writes. A
            larger shape would be an allocation the file has not paid for.
    """
    if subtype is not _NDARRAY or shape != (0,):
        raise pickle.UnpicklingError("an array reconstruction that NumPy would not have written")
    return _NUMPY_RECONSTRUCT(np.ndarray, (0,), dtype)


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
        raise pickle.UnpicklingError(f"bytes encoded as {encoding!r}, which pickle never writes")
    return text.encode("latin1")


def _allowed_globals() -> dict[tuple[str, str], object]:
    """Build the table of every name a pickle may refer to.

    NumPy 1 pickles say ``numpy.core`` and NumPy 2 pickles ``numpy._core``.
    Both spellings map to the installed NumPy's functions, so a file loads
    whichever NumPy wrote it, which ``numpy.load`` itself does not manage.

    Returns:
        ``(module, name)`` to the object handed to the unpickler in its place.
    """
    allowed: dict[tuple[str, str], object] = {
        ("numpy", "ndarray"): _NDARRAY,
        ("numpy", "dtype"): np.dtype,
        ("_codecs", "encode"): _encode,
    }
    for root in ("numpy.core", "numpy._core"):
        allowed[(f"{root}.multiarray", "_reconstruct")] = _reconstruct
        allowed[(f"{root}.multiarray", "scalar")] = _NUMPY_SCALAR
        if _NUMPY_FROMBUFFER is not None:
            allowed[(f"{root}.numeric", "_frombuffer")] = _NUMPY_FROMBUFFER
    return allowed


_ALLOWED = _allowed_globals()


class _RefusedGlobal(pickle.UnpicklingError):
    """A pickle referred to a name outside the allowlist."""


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


def safe_loads(data: bytes, label: str = "pickle") -> object:
    """Unpickle ``data`` without executing anything in it.

    Args:
        data: Exactly one pickle.
        label: What to call the source in a message.

    Returns:
        The object: built from lists, tuples, dicts, numbers, strings, bytes
        and NumPy arrays and scalars, because nothing else can be.

    Raises:
        UnsupportedFileFormatError: If the data refers to any name outside
            the allowlist, is not a well-formed pickle, or is followed by
            further data. ``MemoryError`` is deliberately not converted: it
            means the machine ran out of memory, not that the file is bad.
    """
    stream = io.BytesIO(data)
    try:
        loaded: object = _RestrictedUnpickler(stream).load()
    except MemoryError:
        raise
    except _RefusedGlobal as refused:
        raise UnsupportedFileFormatError(
            f"unsupported {label}: it refers to {refused}; only lists, tuples, integers "
            f"and NumPy arrays are read, and nothing in a pickle is ever executed"
        ) from None
    except Exception as error:
        # The pickle documentation lists AttributeError, EOFError, ImportError
        # and IndexError and says the list is not exhaustive: a malformed
        # pickle can raise nearly anything from inside the machine.
        raise UnsupportedFileFormatError(
            f"not a readable {label}: {type(error).__name__}: {error}"
        ) from None
    if stream.read(1):
        raise UnsupportedFileFormatError(f"unsupported {label}: data follows the first pickle")
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


def _check_size(width: object, height: object, label: str) -> tuple[int, int]:
    """Validate a declared size.

    Args:
        width: The declared width.
        height: The declared height.
        label: What to call the source in a message.

    Returns:
        The ``(width, height)`` pair.

    Raises:
        UnsupportedFileFormatError: If either is not a positive integer.
    """
    sizes: list[int] = []
    for name, value in (("width", width), ("height", height)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise UnsupportedFileFormatError(
                f"invalid {label} {name} {value!r}: expected a positive integer"
            )
        if value < 1:
            raise UnsupportedFileFormatError(
                f"invalid {label} {name} {value!r}: expected a positive integer"
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
    values = np.fromiter(flat, dtype=np.int64, count=count)
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


def _array_from_sequence(
    pixels: Sequence[Any], declared: tuple[int, int] | None, label: str
) -> np.ndarray:
    """Build the ``(height, width, channels)`` array a list of pixels describes.

    A list of tuples must never go through ``np.asarray``, which is slower
    than per-pixel Python; it is flattened by ``itertools.chain`` into
    ``np.fromiter`` with an explicit count, as ``RasterImage.set_raw_data``
    does.

    Args:
        pixels: Rows of pixels, or a flat list of pixels.
        declared: The ``(width, height)`` declared for it, if any.
        label: What to call the source in a message.

    Returns:
        The array, ``uint8``.

    Raises:
        UnsupportedFileFormatError: If the structure is ragged or not pixels
            at all, a flat list has no declared size or one that does not
            match its length, or rows disagree with a declared size.
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
            f"(Task.with_input_size)"
        )
    width, height = declared
    if width * height != count:
        raise UnsupportedFileFormatError(
            f"unsupported {label}: {count} pixels cannot fill the declared {width}x{height} "
            f"({width * height} pixels)"
        )
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
    loaded: object, declared: tuple[int, int] | None = None, label: str = "pickle"
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

    Returns:
        The RGB array, ``uint8`` and C-contiguous.

    Raises:
        UnsupportedFileFormatError: If the object is none of those, breaks
            the rules of the one it is, or contradicts a declared size.
    """
    if isinstance(loaded, dict):
        if set(loaded) != _WRAPPER_KEYS:
            keys = ", ".join(sorted(map(repr, loaded)))
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
        return pixels_from_object(loaded["pixels"], wrapped, label)

    if isinstance(loaded, np.ndarray) and loaded.dtype.hasobject:
        loaded = loaded.tolist()

    if isinstance(loaded, np.ndarray):
        validate_image_array(loaded.shape, loaded.dtype, label)
        array = loaded
    elif isinstance(loaded, (list, tuple)):
        array = _array_from_sequence(loaded, declared, label)
        validate_image_array(array.shape, array.dtype, label)
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
        self.set_array(pixels_from_object(safe_loads(data), self._declared_size))
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
