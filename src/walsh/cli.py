"""The ``walsh`` command-line interface: ``walsh compress`` / ``walsh extract``."""

from __future__ import annotations

import logging
import os
from pathlib import Path

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


def _reject_writing_over_the_input(input_path: str, output_path: str) -> None:
    """Refuse to let OUTPUT name the same file as INPUT.

    Both pipelines read the whole image before writing anything, so naming one
    file twice does not fail -- it silently replaces the original with its own
    lossy reconstruction. Nothing recovers the discarded coefficients
    afterwards, so this is checked before a single byte is read.

    Args:
        input_path: The image or ``.cim`` file to read.
        output_path: Where the result would be written.

    Raises:
        click.UsageError: If the two paths name one file. Hard links and
            symlinks count, which is why an existing destination is compared
            by identity rather than by name.
    """
    try:
        same = os.path.samefile(input_path, output_path)
    except OSError:
        # OUTPUT does not exist yet, so there is no inode to compare against.
        same = Path(output_path).resolve() == Path(input_path).resolve()
    if same:
        raise click.UsageError(
            f"OUTPUT must not be the same file as INPUT ({input_path!r}); "
            f"the original would be replaced by a lossy reconstruction of itself"
        )


def _run(task: Task) -> None:
    """Run ``task``, turning expected failures into a clean CLI error.

    Args:
        task: A fully configured task, ready to run.

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
    """Compress and restore images with the Walsh-Hadamard transform.
    \f
    The form feed above is where click stops printing this docstring in
    ``--help``. The Google sections are for readers of the source; reflowed
    into a help screen they came out as one unreadable run, backticks and all.
    The same marker sits in ``compress`` and ``extract``. These docstrings must
    not become raw strings: click splits on the form-feed *character*.

    Args:
        verbose: How many times ``-v`` was given. One enables info logging,
            two or more enables debug logging.
    """
    level = logging.WARNING - min(verbose, 2) * 10
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@main.command(context_settings=CONTEXT_SETTINGS)
@click.argument("input_path", metavar="INPUT", type=_INPUT_FILE)
@click.argument("output_path", metavar="OUTPUT", type=_OUTPUT_FILE)
@click.option(
    "--y-block-size",
    type=click.IntRange(min=1),
    default=DEFAULT_Y_BLOCK_SIZE,
    show_default=True,
    help="Luma block edge.",
)
@click.option(
    "--chroma-block-size",
    type=click.IntRange(min=1),
    default=DEFAULT_CHROMA_BLOCK_SIZE,
    show_default=True,
    help="Cb/Cr block edge.",
)
@click.option(
    "--packed-block-size",
    type=click.IntRange(min=1),
    default=DEFAULT_PACKED_BLOCK_SIZE,
    show_default=True,
    help="Coefficients kept per axis. Lower means smaller and lossier.",
)
@click.option(
    "--coeff-removal",
    type=float,
    default=None,
    help=(
        "Zero spectral coefficients whose magnitude is strictly below this "
        "threshold. Absolute, so scale it with the block size: a value tuned at "
        "the default 8-pixel luma block prunes far less at a larger one."
    ),
)
def compress(
    input_path: str,
    output_path: str,
    y_block_size: int,
    chroma_block_size: int,
    packed_block_size: int,
    coeff_removal: float | None,
) -> None:
    """Transform a raster image into a .cim file.

    The input format is taken from the filename suffix: .bmp, .ppm, .pnm,
    .pam, .tif, .tiff, or .npy for a bare NumPy array.
    \f
    Args:
        input_path: Image to read.
        output_path: Path of the .cim file to write.
        y_block_size: Block edge for the luma channel.
        chroma_block_size: Block edge for both chroma channels.
        packed_block_size: Coefficients kept per axis. This is the lossy knob.
        coeff_removal: Optional second lossy threshold, or ``None``.

    Raises:
        click.ClickException: If the image cannot be read or is malformed.
        click.UsageError: If OUTPUT names the same file as INPUT.
    """
    _reject_writing_over_the_input(input_path, output_path)
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
    """Restore an image from a .cim file.

    The output format is taken from the filename suffix, so a picture
    compressed from a BMP can be written back out as a PPM.
    \f
    Args:
        input_path: The .cim file to read.
        output_path: Image to write, ending .bmp, .ppm, .pnm, .pam, .tif,
            .tiff or .npy.

    Raises:
        click.ClickException: If the .cim file is malformed, or the output
            suffix names a format that is not supported.
        click.UsageError: If OUTPUT names the same file as INPUT.
    """
    _reject_writing_over_the_input(input_path, output_path)
    _run(Task().with_action(Action.EXTRACT).with_input(input_path).with_output(output_path))


if __name__ == "__main__":  # pragma: no cover
    main()
