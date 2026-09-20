from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from walsh.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_compress_and_extract_via_cli(
    runner: CliRunner, gradient_bmp: Path, tmp_path: Path
) -> None:
    compressed = tmp_path / "cli.cim"
    restored = tmp_path / "cli.bmp"

    result = runner.invoke(main, ["compress", str(gradient_bmp), str(compressed)])
    assert result.exit_code == 0, result.output
    assert compressed.exists()

    result = runner.invoke(main, ["extract", str(compressed), str(restored)])
    assert result.exit_code == 0, result.output
    assert restored.stat().st_size == gradient_bmp.stat().st_size


def test_missing_input_is_a_usage_error(runner: CliRunner, tmp_path: Path) -> None:
    """click.Path(exists=True) rejects it before the task ever runs."""
    result = runner.invoke(
        main, ["compress", str(tmp_path / "nope.bmp"), str(tmp_path / "out.cim")]
    )
    assert result.exit_code == 2
    assert "does not exist" in result.output


def test_malformed_input_reports_a_clean_error(runner: CliRunner, tmp_path: Path) -> None:
    """A file that exists but is not a supported BMP fails with a message, not a traceback."""
    bogus = tmp_path / "bogus.bmp"
    bogus.write_bytes(b"not a bitmap" * 20)

    result = runner.invoke(main, ["compress", str(bogus), str(tmp_path / "out.cim")])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("truncated.bmp", b"BM\x00\x00"),  # shorter than a 54-byte header
        ("empty.bmp", b""),
    ],
)
def test_truncated_input_reports_a_clean_error(
    runner: CliRunner, tmp_path: Path, name: str, content: bytes
) -> None:
    """Too short to unpack a header. struct.error must not reach the user."""
    path = tmp_path / name
    path.write_bytes(content)

    result = runner.invoke(main, ["compress", str(path), str(tmp_path / "out.cim")])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_corrupt_cim_reports_a_clean_error(runner: CliRunner, tmp_path: Path) -> None:
    path = tmp_path / "corrupt.cim"
    path.write_bytes(b"\x01\x02\x03")

    result = runner.invoke(main, ["extract", str(path), str(tmp_path / "out.bmp")])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "Traceback" not in result.output


def test_packed_block_size_flag_is_honoured(
    runner: CliRunner, gradient_bmp: Path, tmp_path: Path
) -> None:
    small = tmp_path / "small.cim"
    large = tmp_path / "large.cim"
    assert (
        runner.invoke(
            main, ["compress", "--packed-block-size", "2", str(gradient_bmp), str(small)]
        ).exit_code
        == 0
    )
    assert (
        runner.invoke(
            main, ["compress", "--packed-block-size", "4", str(gradient_bmp), str(large)]
        ).exit_code
        == 0
    )
    assert small.stat().st_size < large.stat().st_size


def test_block_size_options_change_the_layout(
    runner: CliRunner, gradient_bmp: Path, tmp_path: Path
) -> None:
    default = tmp_path / "default.cim"
    custom = tmp_path / "custom.cim"
    runner.invoke(main, ["compress", str(gradient_bmp), str(default)])
    result = runner.invoke(
        main,
        [
            "compress",
            "--y-block-size",
            "4",
            "--chroma-block-size",
            "8",
            str(gradient_bmp),
            str(custom),
        ],
    )
    assert result.exit_code == 0, result.output
    assert custom.stat().st_size != default.stat().st_size


def test_coeff_removal_changes_the_output_only_when_it_bites(
    runner: CliRunner, gradient_bmp: Path, tmp_path: Path
) -> None:
    """A byte comparison at 0.5, the value this test once used, would assert
    nothing: no coefficient of the gradient is that small. 500 zeroes most of
    them, so the bytes must differ from the default run's, in the same file
    size. Exit code and existence alone left the option free to be ignored."""
    default = tmp_path / "default.cim"
    assert runner.invoke(main, ["compress", str(gradient_bmp), str(default)]).exit_code == 0

    outputs = {}
    for coeff in ("0.5", "500"):
        output = tmp_path / f"{coeff}.cim"
        result = runner.invoke(
            main, ["compress", "--coeff-removal", coeff, str(gradient_bmp), str(output)]
        )
        assert result.exit_code == 0, result.output
        outputs[coeff] = output.read_bytes()

    assert outputs["0.5"] == default.read_bytes()
    assert outputs["500"] != default.read_bytes()
    assert len(outputs["500"]) == len(default.read_bytes())


@pytest.mark.parametrize("flag", ["-v", "-vv"])
def test_verbose_flags_are_accepted(
    runner: CliRunner, flag: str, gradient_bmp: Path, tmp_path: Path
) -> None:
    result = runner.invoke(main, [flag, "compress", str(gradient_bmp), str(tmp_path / "v.cim")])
    assert result.exit_code == 0, result.output


def test_no_arguments_shows_help(runner: CliRunner) -> None:
    result = runner.invoke(main, [])
    assert "compress" in result.output
    assert "extract" in result.output


def test_unknown_command_is_rejected(runner: CliRunner) -> None:
    result = runner.invoke(main, ["sharpen"])
    assert result.exit_code == 2


def test_version_flag(runner: CliRunner) -> None:
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "walsh" in result.output


@pytest.mark.parametrize("help_flag", ["-h", "--help"])
def test_help_is_available_under_both_flags(runner: CliRunner, help_flag: str) -> None:
    result = runner.invoke(main, [help_flag])
    assert result.exit_code == 0
    assert "Walsh-Hadamard" in result.output


def test_subcommand_help_lists_the_options(runner: CliRunner) -> None:
    result = runner.invoke(main, ["compress", "--help"])
    assert result.exit_code == 0
    for option in (
        "--y-block-size",
        "--chroma-block-size",
        "--packed-block-size",
        "--coeff-removal",
    ):
        assert option in result.output


# -- writing must not destroy what is already there ------------------------


@pytest.mark.parametrize("command", ["compress", "extract"])
def test_output_may_not_be_the_input_file(
    runner: CliRunner, gradient_bmp: Path, command: str
) -> None:
    """Both pipelines read everything before writing, so this used to succeed."""
    before = gradient_bmp.read_bytes()

    result = runner.invoke(main, [command, str(gradient_bmp), str(gradient_bmp)])

    assert result.exit_code == 2, result.output
    assert "must not be the same file" in result.output
    assert gradient_bmp.read_bytes() == before


def test_output_may_not_reach_the_input_through_a_symlink(
    runner: CliRunner, gradient_bmp: Path, tmp_path: Path
) -> None:
    """Comparison is by identity, not by name, so an alias is caught too."""
    alias = tmp_path / "alias.bmp"
    alias.symlink_to(gradient_bmp)
    before = gradient_bmp.read_bytes()

    result = runner.invoke(main, ["compress", str(gradient_bmp), str(alias)])

    assert result.exit_code == 2, result.output
    assert gradient_bmp.read_bytes() == before


def test_a_different_output_file_is_still_allowed(
    runner: CliRunner, gradient_bmp: Path, tmp_path: Path
) -> None:
    """The guard must not reject the ordinary case, including a re-run."""
    output = tmp_path / "out.cim"
    output.write_bytes(b"a previous run")

    result = runner.invoke(main, ["compress", str(gradient_bmp), str(output)])

    assert result.exit_code == 0, result.output
    assert output.read_bytes() != b"a previous run"


def test_a_failed_compress_leaves_the_existing_output_intact(
    runner: CliRunner, gradient_bmp: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure after `save()` has opened the destination must not replace it.

    That is what the staged write in `_io` exists to survive: ENOSPC, EIO, a
    crash mid-write. Every header-field overflow that used to serve as the
    example here is now caught before the file is opened (#19 in 0.4.6, #22 in
    0.4.7, #21 in 0.4.11), so the failure is injected where such a fault would
    land, in the block writer, after the header has gone out.
    """
    from walsh.image import CustomizableImage

    def fail_mid_write(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(CustomizableImage, "_write_blocks", fail_mid_write)
    output = tmp_path / "archive.cim"
    output.write_bytes(b"PREVIOUS ENCODE")

    result = runner.invoke(main, ["compress", str(gradient_bmp), str(output)])

    assert result.exit_code == 1, result.output
    assert "No space left on device" in result.output
    assert output.read_bytes() == b"PREVIOUS ENCODE"
    assert not [p for p in tmp_path.iterdir() if p.name.startswith(".walsh-")]


def test_a_failed_extract_leaves_the_existing_output_intact(
    runner: CliRunner, tmp_path: Path
) -> None:
    """Dimensions too large for the BMP header raise after the file is opened."""
    import struct

    source = tmp_path / "huge.cim"
    source.write_bytes(struct.pack("<II", 2**31, 0) + struct.pack("<HHH", 8, 4, 0) * 3)
    output = tmp_path / "existing.bmp"
    output.write_bytes(b"BM" + b"\x00" * 100)
    before = output.read_bytes()

    result = runner.invoke(main, ["extract", str(source), str(output)])

    assert result.exit_code == 1, result.output
    assert output.read_bytes() == before


# -- a .cim that is not one, or lies about itself, must fail cleanly ------


@pytest.mark.parametrize(
    ("name", "data"),
    [
        (
            "allocation bomb",
            b"\x08\x00\x00\x00\x08\x00\x00\x00\xff\xff\x00\x00\xff\xff"
            + b"\x10\x00\x04\x00\x00\x00" * 2,
        ),
        ("zero width", b"\x00\x00\x00\x00\x10\x00\x00\x00" + b"\x08\x00\x04\x00\x00\x00" * 3),
        ("zero height", b"\x10\x00\x00\x00\x00\x00\x00\x00" + b"\x08\x00\x04\x00\x00\x00" * 3),
        ("packed size 0", b"\x10\x00\x00\x00\x10\x00\x00\x00" + b"\x08\x00\x00\x00\x04\x00" * 3),
    ],
)
def test_malformed_cim_geometry_is_a_clean_error(
    runner: CliRunner, tmp_path: Path, name: str, data: bytes
) -> None:
    """Each of these used to be a bare traceback (MemoryError, ZeroDivisionError)
    or, worse, exit 0 with a corrupt image. Under CliRunner an uncaught
    exception lands in result.exception and is not printed, so assert on the
    exit code and the Error: line rather than on the absence of a traceback.
    """
    path = tmp_path / "bad.cim"
    path.write_bytes(data)

    result = runner.invoke(main, ["extract", str(path), str(tmp_path / "out.bmp")])

    assert result.exit_code == 1, (name, result.output, result.exception)
    assert "Error: invalid .cim" in result.output, (name, result.output)
    assert not (tmp_path / "out.bmp").exists()


def test_a_raster_image_given_to_extract_fails_fast(
    runner: CliRunner, sample_bmp: Path, tmp_path: Path
) -> None:
    """The ordinary argument mix-up. It used to run for a minute and try to
    allocate 78 GB before blaming a block size."""
    result = runner.invoke(main, ["extract", str(sample_bmp), str(tmp_path / "out.bmp")])
    assert result.exit_code == 1, result.output
    assert "Error: invalid .cim" in result.output


@pytest.mark.parametrize("option", ["--packed-block-size", "--y-block-size", "--chroma-block-size"])
@pytest.mark.parametrize("value", ["0", "-4"])
def test_block_sizes_below_one_are_refused_before_any_work(
    runner: CliRunner, gradient_bmp: Path, tmp_path: Path, option: str, value: str
) -> None:
    """--packed-block-size 0 used to encode a header-only file that decoded to
    solid green; a negative value failed by accident inside struct.pack."""
    result = runner.invoke(
        main, ["compress", option, value, str(gradient_bmp), str(tmp_path / "o.cim")]
    )
    assert result.exit_code == 2, result.output
    assert "Invalid value" in result.output
    assert not (tmp_path / "o.cim").exists()


# -- the help screens are for users, not for readers of the source --------


@pytest.mark.parametrize("command", [[], ["compress"], ["extract"]])
def test_help_stops_before_the_docstring_sections(runner: CliRunner, command: list[str]) -> None:
    """Click reflowed the Google Args:/Raises: sections into one run, complete
    with RST backticks and an internal exception class name. A form feed in
    each docstring now ends what --help prints. The docstrings must stay plain
    strings: a raw string would print a literal backslash-f instead."""
    result = runner.invoke(main, [*command, "--help"])
    assert result.exit_code == 0, result.output
    for leak in ("Args:", "Raises:", "``", "ClickException", "form feed"):
        assert leak not in result.output, (command, leak, result.output)


def test_help_still_says_what_each_command_does(runner: CliRunner) -> None:
    """The form feed must cut after the summary, not before it."""
    assert "Walsh-Hadamard" in runner.invoke(main, ["--help"]).output
    assert "Transform a raster image" in runner.invoke(main, ["compress", "--help"]).output
    assert "Restore an image" in runner.invoke(main, ["extract", "--help"]).output


def test_coeff_removal_help_describes_what_the_code_does(runner: CliRunner) -> None:
    """It thresholds spectral coefficients, strictly below, never the matrix.

    The old text, "Zero Hadamard matrix entries at or below this threshold",
    was the 0.2.1 bug restated as documentation, and it sent users to a scale
    (every matrix entry is 0.35 at block 8) where the option is a no-op.
    """
    output = runner.invoke(main, ["compress", "--help"]).output
    assert "spectral coefficients" in output
    assert "strictly below" in output
    assert "block size" in output
    assert "matrix" not in output


@pytest.mark.parametrize(
    ("options", "match"),
    [
        (["--packed-block-size", "16"], "smallest block edge"),
        (["--y-block-size", "6"], "positive power of two"),
        (["--y-block-size", "256", "--chroma-block-size", "256"], "at most 128"),
    ],
)
def test_bad_block_geometry_is_a_clean_error_that_writes_nothing(
    runner: CliRunner, gradient_bmp: Path, tmp_path: Path, options: list[str], match: str
) -> None:
    """--packed-block-size 16 used to exit 0 and write a file extract refuses forever."""
    output = tmp_path / "o.cim"
    result = runner.invoke(main, ["compress", *options, str(gradient_bmp), str(output)])
    assert result.exit_code == 1, result.output
    assert "Error:" in result.output and match in result.output
    assert not output.exists()


def test_a_picture_as_output_skips_the_cim_file(
    runner: CliRunner, gradient_ppm: Path, tmp_path: Path
) -> None:
    """`walsh compress photo.ppm photo_lossy.ppm` writes the reconstruction,
    exactly what the two commands would have written between them."""
    direct = tmp_path / "direct.ppm"
    result = runner.invoke(main, ["compress", str(gradient_ppm), str(direct)])
    assert result.exit_code == 0, result.output

    container, expected = tmp_path / "two.cim", tmp_path / "two.ppm"
    assert runner.invoke(main, ["compress", str(gradient_ppm), str(container)]).exit_code == 0
    assert runner.invoke(main, ["extract", str(container), str(expected)]).exit_code == 0
    assert direct.read_bytes() == expected.read_bytes()
    assert direct.read_bytes() != gradient_ppm.read_bytes(), "the codec is lossy"


def test_the_options_apply_to_a_picture_output_too(
    runner: CliRunner, gradient_ppm: Path, tmp_path: Path
) -> None:
    plain, coarse = tmp_path / "plain.ppm", tmp_path / "coarse.ppm"
    assert runner.invoke(main, ["compress", str(gradient_ppm), str(plain)]).exit_code == 0
    result = runner.invoke(
        main, ["compress", "--packed-block-size", "1", str(gradient_ppm), str(coarse)]
    )
    assert result.exit_code == 0, result.output
    assert coarse.read_bytes() != plain.read_bytes()


def test_a_picture_of_another_type_is_a_clean_error_that_names_the_release(
    runner: CliRunner, gradient_ppm: Path, tmp_path: Path
) -> None:
    """NotImplementedError is not one of EXPECTED_ERRORS, so without its own
    handler this would be a traceback."""
    output = tmp_path / "crossed.png"
    result = runner.invoke(main, ["compress", str(gradient_ppm), str(output)])
    assert result.exit_code == 1
    assert "Error:" in result.output
    assert "planned for 0.6.0" in result.output and "issues/51" in result.output
    assert "extract that to .png" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert not output.exists()


def test_the_help_says_a_picture_can_be_the_output(runner: CliRunner) -> None:
    result = runner.invoke(main, ["compress", "--help"])
    text = " ".join(result.output.split())
    assert "the .cim file is skipped" in text
    assert "A different picture type is not supported yet" in text
