"""Shared byte-level helpers for the raster format readers and writers."""

from __future__ import annotations

import contextlib
import os
import stat
import sys
import tempfile
from collections.abc import Generator
from os import PathLike
from typing import BinaryIO

__all__ = ["FileSource", "align", "open_binary", "open_binary_read", "open_binary_write"]

#: Where an image is read from or written to. ``None`` means the standard
#: streams, which lets the codec be used in a shell pipeline.
FileSource = str | PathLike[str] | None

#: Prefix for the staging file :func:`open_binary_write` writes through. The
#: leading dot keeps it out of ``ls`` and glob expansion for the moment it
#: exists; the rest names the owner, so a leaked one is traceable.
_STAGING_PREFIX = ".walsh-"
_STAGING_SUFFIX = ".tmp"

#: Permissions for a destination that does not exist yet: what ``open()`` would
#: have produced, i.e. 0666 less the process umask.
_DEFAULT_FILE_MODE = 0o666


def align(x: int, a: int) -> int:
    """Round ``x`` up to the next multiple of ``a``.

    Args:
        x: The value to round. Assumed non-negative.
        a: The alignment. Must be positive.

    Returns:
        The smallest multiple of ``a`` that is greater than or equal to ``x``.
    """
    return (((x - 1) // a) + 1) * a


@contextlib.contextmanager
def open_binary_read(source: FileSource) -> Generator[BinaryIO, None, None]:
    """Open ``source`` for binary reading, or fall back to stdin.

    Args:
        source: Path to open, or ``None`` to read from stdin.

    Yields:
        The open binary stream. Standard input is deliberately not closed.

    Raises:
        OSError: If ``source`` names a file that cannot be opened.
    """
    if source is None:
        yield sys.stdin.buffer
        return
    with open(source, "rb") as handle:
        yield handle


def _target_mode(target: str) -> int:
    """Work out what permissions the finished file should carry.

    Args:
        target: The destination path, which need not exist.

    Returns:
        The existing file's permission bits, so replacing it does not change
        them; or what ``open(target, "wb")`` would have created, which is
        ``0o666`` less the process umask.
    """
    try:
        return stat.S_IMODE(os.stat(target).st_mode)
    except OSError:
        # There is no way to read the umask without setting it, so set it to a
        # known value and immediately put it back. The window is two calls
        # wide and this is the only place in the package that touches it.
        umask = os.umask(0)
        os.umask(umask)
        return _DEFAULT_FILE_MODE & ~umask


@contextlib.contextmanager
def open_binary_write(source: FileSource) -> Generator[BinaryIO, None, None]:
    """Open ``source`` for binary writing, or fall back to stdout.

    The write is **staged**: bytes go to a temporary file beside the
    destination and are moved onto it with :func:`os.replace` only once the
    caller has finished without raising. A write that fails part way -- an
    image too large for the container's fields, a full disk, an I/O error --
    therefore leaves whatever was already at ``source`` untouched, rather than
    replacing it with a truncated stub. ``os.replace`` is atomic within a
    filesystem, which is why the staging file is created in the destination's
    own directory rather than in the system temporary directory.

    Two consequences are deliberate. The destination gets a new inode, so an
    existing hard link to it keeps the old contents. A symlinked destination is
    resolved first, so the link itself survives and its target is replaced,
    which is what writing through it did before. Destinations that are not
    regular files -- ``/dev/null``, a FIFO -- cannot be replaced at all and so
    are written directly, as they always were.

    Args:
        source: Path to write, or ``None`` to write to stdout.

    Yields:
        The open binary stream. Standard output is deliberately not closed.

    Raises:
        OSError: If ``source`` names a file that cannot be opened, or the
            staging file cannot be created in its directory.
    """
    if source is None:
        yield sys.stdout.buffer
        return

    # Resolving first means a symlinked destination keeps its link: the file it
    # points at is what gets replaced, exactly as writing through it did.
    target = os.path.realpath(source)
    if os.path.exists(target) and not os.path.isfile(target):
        with open(target, "wb") as handle:
            yield handle
        return

    mode = _target_mode(target)
    descriptor, staged = tempfile.mkstemp(
        dir=os.path.dirname(target), prefix=_STAGING_PREFIX, suffix=_STAGING_SUFFIX
    )
    try:
        with open(descriptor, "wb") as handle:
            yield handle
        os.chmod(staged, mode)
        os.replace(staged, target)
    except BaseException:
        # Including KeyboardInterrupt: an interrupted write must not leave the
        # staging file behind any more than it may damage the destination.
        with contextlib.suppress(OSError):
            os.unlink(staged)
        raise


@contextlib.contextmanager
def open_binary(source: FileSource, mode: str) -> Generator[BinaryIO, None, None]:
    """Open ``source`` in binary mode, falling back to the standard streams.

    Kept for callers that predate the split into :func:`open_binary_read` and
    :func:`open_binary_write`. Prefer those: passing the mode as a string means
    neither the type checker nor the reader can tell which stream comes back,
    which is what forced a suppression here before.

    Args:
        source: Path to open, or ``None`` to use stdin/stdout.
        mode: Either a read mode or a write mode; the presence of ``"r"``
            selects stdin over stdout when ``source`` is ``None``.

    Yields:
        The open binary stream. A write goes through the staging described on
        :func:`open_binary_write`.

    Raises:
        OSError: If ``source`` names a file that cannot be opened.
    """
    opener = open_binary_read if "r" in mode else open_binary_write
    with opener(source) as handle:
        yield handle
