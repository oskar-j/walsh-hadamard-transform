"""The ``walsh`` command-line interface: ``walsh compress`` / ``walsh extract``."""

from __future__ import annotations

import logging

import click

from walsh.exceptions import EXPECTED_ERRORS
from walsh.task import (
    DEFAULT_CHROMA_BLOCK_SIZE,
    DEFAULT_PACKED_BLOCK_SIZE,
    DEFAULT_Y_BLOCK_SIZE,
    Action,
    Task,
)

__all__ = ["main"]

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

_INPUT_FILE = click.Path(exists=True, dir_okay=False, readable=True)
_OUTPUT_FILE = click.Path(dir_okay=False, writable=True)


def _run(task: Task) -> None:
    """Run ``task``, turning expected failures into a clean CLI error.

    Raises:
        click.ClickException: if the task fails for a reason the user can act
            on, such as a missing or malformed input file.
    """
    try:
        task.run()
    except EXPECTED_ERRORS as error:
        raise click.ClickException(str(error)) from error


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase log verbosity. Repeat for debug output.",
)
@click.version_option(package_name="walsh", prog_name="walsh")
def main(verbose: int) -> None:
    """Compress and restore images with the Walsh-Hadamard transform."""
    level = logging.WARNING - min(verbose, 2) * 10
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("input_path", metavar="INPUT", type=_INPUT_FILE)
@click.argument("output_path", metavar="OUTPUT", type=_OUTPUT_FILE)
@click.option(
    "--y-block-size",
    type=int,
    default=DEFAULT_Y_BLOCK_SIZE,
    show_default=True,
    help="Luma block edge.",
)
@click.option(
    "--chroma-block-size",
    type=int,
    default=DEFAULT_CHROMA_BLOCK_SIZE,
    show_default=True,
    help="Cb/Cr block edge.",
)
@click.option(
    "--packed-block-size",
    type=int,
    default=DEFAULT_PACKED_BLOCK_SIZE,
    show_default=True,
    help="Coefficients kept per axis. Lower means smaller and lossier.",
)
@click.option(
    "--coeff-removal",
    type=float,
    default=None,
    help="Zero Hadamard matrix entries at or below this threshold.",
)
def compress(
    input_path: str,
    output_path: str,
    y_block_size: int,
    chroma_block_size: int,
    packed_block_size: int,
    coeff_removal: float | None,
) -> None:
    """Transform a 24-bit BMP into a .cim file."""
    task = Task(
        y_block_size=y_block_size,
        cb_block_size=chroma_block_size,
        cr_block_size=chroma_block_size,
        packed_block_size=packed_block_size,
    )
    _run(
        task.with_coeff_removal(coeff_removal)
        .with_action(Action.COMPRESS)
        .with_input(input_path)
        .with_output(output_path)
    )


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("input_path", metavar="INPUT", type=_INPUT_FILE)
@click.argument("output_path", metavar="OUTPUT", type=_OUTPUT_FILE)
def extract(input_path: str, output_path: str) -> None:
    """Restore a BMP from a .cim file."""
    _run(Task().with_action(Action.EXTRACT).with_input(input_path).with_output(output_path))


if __name__ == "__main__":  # pragma: no cover
    main()
