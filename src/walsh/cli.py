"""Command line interface: ``walsh compress`` / ``walsh extract``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from walsh import __version__
from walsh.image import UnsupportedFileFormatError
from walsh.task import (
    DEFAULT_CHROMA_BLOCK_SIZE,
    DEFAULT_PACKED_BLOCK_SIZE,
    DEFAULT_Y_BLOCK_SIZE,
    Action,
    Task,
)

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="walsh",
        description="Compress and restore images with the Walsh-Hadamard transform.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="increase log verbosity (repeat for debug output)",
    )

    subparsers = parser.add_subparsers(dest="action", required=True)

    compress = subparsers.add_parser(
        Action.COMPRESS.value, help="transform a 24-bit BMP into a .cim file"
    )
    compress.add_argument("input", help="path to a 24-bit uncompressed BMP")
    compress.add_argument("output", help="path of the .cim file to write")
    compress.add_argument(
        "--y-block-size",
        type=int,
        default=DEFAULT_Y_BLOCK_SIZE,
        help="luma block edge (default: %(default)s)",
    )
    compress.add_argument(
        "--chroma-block-size",
        type=int,
        default=DEFAULT_CHROMA_BLOCK_SIZE,
        help="Cb/Cr block edge (default: %(default)s)",
    )
    compress.add_argument(
        "--packed-block-size",
        type=int,
        default=DEFAULT_PACKED_BLOCK_SIZE,
        help="coefficients kept per axis; lower means smaller and lossier (default: %(default)s)",
    )
    compress.add_argument(
        "--coeff-removal",
        type=float,
        default=None,
        help="zero Hadamard matrix entries at or below this threshold",
    )

    extract = subparsers.add_parser(Action.EXTRACT.value, help="restore a BMP from a .cim file")
    extract.add_argument("input", help="path to a .cim file")
    extract.add_argument("output", help="path of the BMP to write")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    level = logging.WARNING - min(args.verbose, 2) * 10
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    action = Action(args.action)
    if action is Action.COMPRESS:
        task = Task(
            y_block_size=args.y_block_size,
            cb_block_size=args.chroma_block_size,
            cr_block_size=args.chroma_block_size,
            packed_block_size=args.packed_block_size,
        ).with_coeff_removal(args.coeff_removal)
    else:
        task = Task()

    task.with_action(action).with_input(args.input).with_output(args.output)

    try:
        task.run()
    except (OSError, UnsupportedFileFormatError, ValueError) as error:
        print(f"walsh: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
