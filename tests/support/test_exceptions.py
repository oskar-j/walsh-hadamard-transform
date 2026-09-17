from __future__ import annotations

import struct

import pytest

import walsh
from walsh.exceptions import (
    EXPECTED_ERRORS,
    UnsupportedFileFormatError,
    WalshError,
)


def test_package_errors_share_a_base() -> None:
    assert issubclass(UnsupportedFileFormatError, WalshError)
    assert issubclass(WalshError, Exception)


def test_base_class_catches_package_errors() -> None:
    with pytest.raises(WalshError):
        raise UnsupportedFileFormatError("not a BMP")


def test_exceptions_are_exported_from_the_package_root() -> None:
    assert walsh.UnsupportedFileFormatError is UnsupportedFileFormatError
    assert walsh.WalshError is WalshError


def test_image_still_re_exports_the_format_error() -> None:
    """`from walsh.image import UnsupportedFileFormatError` predates the move."""
    from walsh.image import UnsupportedFileFormatError as from_image

    assert from_image is UnsupportedFileFormatError


@pytest.mark.parametrize(
    "error",
    [
        UnsupportedFileFormatError("bad header"),
        OSError("no such file"),
        ValueError("bad value"),
        struct.error("unpack requires a buffer of 54 bytes"),
    ],
)
def test_expected_errors_covers_the_recoverable_failures(error: Exception) -> None:
    assert isinstance(error, EXPECTED_ERRORS)


def test_struct_error_is_not_reachable_via_value_error() -> None:
    """The reason struct.error has to be listed separately. Guards a real bug."""
    assert not issubclass(struct.error, ValueError)
    assert not issubclass(struct.error, OSError)


def test_expected_errors_does_not_swallow_programming_mistakes() -> None:
    for bug in (TypeError("nope"), AttributeError("nope"), KeyError("nope")):
        assert not isinstance(bug, EXPECTED_ERRORS)
