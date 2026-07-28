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


def test_coeff_removal_is_accepted(runner: CliRunner, gradient_bmp: Path, tmp_path: Path) -> None:
    output = tmp_path / "coeff.cim"
    result = runner.invoke(
        main, ["compress", "--coeff-removal", "0.5", str(gradient_bmp), str(output)]
    )
    assert result.exit_code == 0, result.output
    assert output.exists()


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
