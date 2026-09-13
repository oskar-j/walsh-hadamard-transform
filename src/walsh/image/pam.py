"""Portable Arbitrary Map (PAM) images, Netpbm's ``P7`` container.

PAM generalises the older Netpbm formats. The raster is the same run of raw
samples that follows a PPM header, but the header is a block of ``KEY value``
lines closed by ``ENDHDR``, and it states how many samples make a pixel
(``DEPTH``) and what they mean (``TUPLTYPE``). One container therefore serves
bitmaps, greyscale, RGB and anything with alpha.

Exactly one profile is supported: ``DEPTH 3`` with ``TUPLTYPE RGB``, or with no
``TUPLTYPE`` at all, which by convention means the same, and ``MAXVAL`` at
most 255. Greyscale, alpha and 16-bit files are rejected by name rather than
guessed at. A ``maxval`` below 255 is rescaled on load, as for PPM. Header
keys may come in any order; comment lines and blank lines are skipped.
"""

from __future__ import annotations

import logging
from typing import BinaryIO

from walsh.exceptions import UnsupportedFileFormatError
from walsh.image._io import FileSource, open_binary_read, open_binary_write
from walsh.image._netpbm import NETPBM_MAX_SAMPLE, encode_samples, read_samples
from walsh.image.base import RasterImage

__all__ = ["PAM_DEPTH", "PAM_MAGIC", "PAM_TUPLTYPE", "PAMImage"]

log = logging.getLogger(__name__)

PAM_MAGIC = b"P7"

#: Samples per pixel in the one supported profile.
PAM_DEPTH = 3

#: The tuple type of the one supported profile.
PAM_TUPLTYPE = b"RGB"

_ENDHDR = b"ENDHDR"
_TUPLTYPE = b"TUPLTYPE"
_COMMENT = b"#"

#: The integer header fields. Each must appear exactly once.
_REQUIRED = (b"WIDTH", b"HEIGHT", b"DEPTH", b"MAXVAL")


class PAMImage(RasterImage):
    """A Portable Arbitrary Map holding RGB pixels.

    As with PPM, the samples are stored red-green-blue and top-down on disk,
    which is already this package's in-memory contract, so no conversion is
    needed either way.
    """

    @staticmethod
    def _read_line(file: BinaryIO) -> bytes | None:
        """Read one header line, without its line ending.

        Args:
            file: Stream positioned at the start of a line.

        Returns:
            The line's bytes, or ``None`` if the stream ends before a newline.
            A trailing carriage return is dropped so CRLF files parse the same
            as LF ones.
        """
        line = bytearray()
        while True:
            char = file.read(1)
            if not char:
                return None
            if char == b"\n":
                return bytes(line.rstrip(b"\r"))
            line += char

    @staticmethod
    def _parse_int(keyword: bytes, value: bytes) -> int:
        """Interpret a header value as a positive integer.

        Args:
            keyword: The field name, used only in the error message.
            value: The bytes following the keyword on its line.

        Returns:
            The parsed value.

        Raises:
            UnsupportedFileFormatError: If the value is not a positive integer.
        """
        name = keyword.decode()
        try:
            parsed = int(value)
        except ValueError:
            raise UnsupportedFileFormatError(
                f"invalid PAM {name}: {value!r} is not a number"
            ) from None
        if parsed <= 0:
            raise UnsupportedFileFormatError(f"invalid PAM {name}: {parsed}")
        return parsed

    @staticmethod
    def _check_profile(depth: int, maxval: int, tupltype: bytes) -> None:
        """Reject, by name, anything but 8-bit RGB.

        Args:
            depth: The declared samples per pixel.
            maxval: The declared maximum sample value.
            tupltype: The declared tuple type, empty if the header had none.

        Raises:
            UnsupportedFileFormatError: If the file is greyscale, carries
                alpha, has any other tuple type or depth, or uses 16-bit
                samples.
        """
        if tupltype and tupltype != PAM_TUPLTYPE:
            shown = tupltype.decode(errors="replace")
            raise UnsupportedFileFormatError(
                f"PAM tuple type {shown!r} is not supported; expected {PAM_TUPLTYPE.decode()}"
            )
        if depth != PAM_DEPTH:
            raise UnsupportedFileFormatError(
                f"PAM depth {depth} is not supported; expected {PAM_DEPTH} (RGB)"
            )
        if maxval > NETPBM_MAX_SAMPLE:
            raise UnsupportedFileFormatError(
                f"16-bit PAM samples are not supported: maxval is {maxval}, "
                f"expected at most {NETPBM_MAX_SAMPLE}"
            )

    def _read_header(self, file: BinaryIO) -> int:
        """Parse the header through ``ENDHDR`` and check it is the RGB profile.

        Args:
            file: Stream positioned at the start of the file.

        Returns:
            The declared ``maxval``.

        Raises:
            UnsupportedFileFormatError: If the magic is not ``P7``, a field is
                missing, repeated or malformed, a keyword is unknown, the
                header ends without ``ENDHDR``, or the profile is not 8-bit RGB.
        """
        magic = self._read_line(file)
        if magic is None or magic.strip() != PAM_MAGIC:
            raise UnsupportedFileFormatError(f"expected a PAM starting P7, got {magic!r}")

        fields: dict[bytes, int] = {}
        tupltype: list[bytes] = []
        while True:
            line = self._read_line(file)
            if line is None:
                raise UnsupportedFileFormatError("truncated PAM header: no ENDHDR")
            parts = line.split(None, 1)
            if not parts or parts[0].startswith(_COMMENT):
                continue
            keyword = parts[0]
            value = parts[1].strip() if len(parts) > 1 else b""

            if keyword == _ENDHDR:
                break
            if keyword == _TUPLTYPE:
                # Several TUPLTYPE lines are one value, joined by spaces.
                tupltype.append(value)
            elif keyword in _REQUIRED:
                if keyword in fields:
                    raise UnsupportedFileFormatError(
                        f"invalid PAM header: {keyword.decode()} given twice"
                    )
                fields[keyword] = self._parse_int(keyword, value)
            else:
                raise UnsupportedFileFormatError(f"unknown PAM header line {keyword!r}")

        missing = [keyword.decode() for keyword in _REQUIRED if keyword not in fields]
        if missing:
            raise UnsupportedFileFormatError(f"invalid PAM header: missing {', '.join(missing)}")

        self._check_profile(fields[b"DEPTH"], fields[b"MAXVAL"], b" ".join(tupltype).strip())
        self._width, self._height = fields[b"WIDTH"], fields[b"HEIGHT"]
        return fields[b"MAXVAL"]

    def load(self, filename: FileSource) -> None:
        """Read a PAM from ``filename``, replacing any current contents.

        Args:
            filename: Path to read, or ``None`` to read from stdin.

        Raises:
            UnsupportedFileFormatError: If the data is not a supported PAM.
            OSError: If the file cannot be read.
        """
        with open_binary_read(filename) as file:
            maxval = self._read_header(file)
            self._raw_data = read_samples(file, self._width * self._height, maxval, "PAM")
        log.debug("loaded PAM %dx%d from %s", self._width, self._height, filename)

    def save(self, filename: FileSource) -> None:
        """Write this image to ``filename`` as an 8-bit RGB PAM.

        The header is written in the order Netpbm's own tools use, so a file
        this package writes is byte-identical to one ``pamtopam`` would.

        Args:
            filename: Path to write, or ``None`` to write to stdout.

        Raises:
            OSError: If the file cannot be written.
        """
        header = b"%s\nWIDTH %d\nHEIGHT %d\nDEPTH %d\nMAXVAL %d\nTUPLTYPE %s\nENDHDR\n" % (
            PAM_MAGIC,
            self._width,
            self._height,
            PAM_DEPTH,
            NETPBM_MAX_SAMPLE,
            PAM_TUPLTYPE,
        )
        with open_binary_write(filename) as file:
            file.write(header)
            file.write(encode_samples(self._raw_data))
