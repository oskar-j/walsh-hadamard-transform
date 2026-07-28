from __future__ import annotations

from pathlib import Path

import pytest

from walsh.cli import main


def test_compress_and_extract_via_cli(gradient_bmp: Path, tmp_path: Path) -> None:
    compressed = tmp_path / "cli.cim"
    restored = tmp_path / "cli.bmp"

    assert main(["compress", str(gradient_bmp), str(compressed)]) == 0
    assert compressed.exists()

    assert main(["extract", str(compressed), str(restored)]) == 0
    assert restored.stat().st_size == gradient_bmp.stat().st_size


def test_missing_input_reports_an_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["compress", str(tmp_path / "nope.bmp"), str(tmp_path / "out.cim")])
    assert code == 1
    assert "walsh:" in capsys.readouterr().err


def test_packed_block_size_flag_is_honoured(gradient_bmp: Path, tmp_path: Path) -> None:
    small = tmp_path / "small.cim"
    large = tmp_path / "large.cim"
    main(["compress", "--packed-block-size", "2", str(gradient_bmp), str(small)])
    main(["compress", "--packed-block-size", "4", str(gradient_bmp), str(large)])
    assert small.stat().st_size < large.stat().st_size


def test_action_is_required(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main([])


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert "walsh" in capsys.readouterr().out
