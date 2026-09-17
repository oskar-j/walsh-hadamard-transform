"""`Task(transform=...)`: another block transform through the same pipeline (#41)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from conftest import ROOT, write_ppm
from walsh import (
    DiscreteCosineTransform,
    HaarTransform,
    Task,
    Transform,
    WalshHadamardTransform,
    reader_for,
)
from walsh.cli import main
from walsh.transforms import Block, remove_small_coefficients

HEADER = 26
EXAMPLE = ROOT / "examples" / "compare_transforms.py"


def _dct_matrix(edge: int) -> Block:
    k = np.arange(edge)[:, None]
    i = np.arange(edge)[None, :]
    matrix = np.cos(np.pi * (2 * i + 1) * k / (2 * edge)) * np.sqrt(2 / edge)
    matrix[0] /= np.sqrt(2)
    return matrix


class Dct(Transform):
    """An orthonormal DCT-II that knows nothing of stacks: the minimum a
    subclass has to write, so Task reaches it through the per-block defaults."""

    def transform(self, src: Block) -> Block:
        m = _dct_matrix(src.shape[-1])
        return m @ src @ m.T

    def inverse_transform(self, src: Block) -> Block:
        m = _dct_matrix(src.shape[-1])
        return m.T @ src @ m


class StackedDct(Dct):
    """The same transform with the stack methods overridden for speed."""

    def transform_stack(self, stack: Block) -> Block:
        return self.transform(stack)

    def inverse_transform_stack(self, stack: Block) -> Block:
        return self.inverse_transform(stack)


@pytest.fixture
def picture(tmp_path: Path) -> Path:
    """64x48, smooth in one channel and ramped in the others: enough blocks
    in every channel for the transforms to differ, small enough to be quick."""
    width, height = 64, 48
    y, x = np.mgrid[0:height, 0:width]
    planes = np.stack(
        [128 + 100 * np.sin(x / 9.0) * np.cos(y / 7.0), 40 + 3 * x + 0 * y, 255 - 4 * y + 0 * x],
        axis=-1,
    )
    pixels = [tuple(int(v) for v in p) for p in np.clip(planes, 0, 255).reshape(-1, 3)]
    return write_ppm(tmp_path / "picture.ppm", width, height, pixels)


def _pixels(path: Path) -> np.ndarray:
    image = reader_for(str(path))
    image.load(str(path))
    return image.get_array().astype(np.float64)


def _psnr(original: Path, restored: Path) -> float:
    mse = float(np.mean((_pixels(original) - _pixels(restored)) ** 2))
    return float(10 * np.log10(255.0**2 / mse))


def _compress(task: Task, source: Path, output: Path) -> bytes:
    task.with_action("compress").with_input(str(source)).with_output(str(output)).run()
    return output.read_bytes()


def _extract(task: Task, source: Path, output: Path) -> Path:
    task.with_action("extract").with_input(str(source)).with_output(str(output)).run()
    return output


@pytest.mark.parametrize("transform", [Dct(), StackedDct()], ids=["per-block", "stacked"])
def test_a_custom_transform_round_trips_through_the_whole_pipeline(
    transform: Transform, picture: Path, tmp_path: Path
) -> None:
    _compress(Task(transform=transform), picture, tmp_path / "dct.cim")
    restored = _extract(Task(transform=transform), tmp_path / "dct.cim", tmp_path / "back.ppm")
    assert _psnr(picture, restored) > 35


def test_the_per_block_default_and_a_stack_override_agree(picture: Path, tmp_path: Path) -> None:
    rng = np.random.default_rng(seed=41)
    stack = rng.uniform(0, 255, size=(9, 8, 8))
    np.testing.assert_allclose(
        Dct().transform_stack(stack), StackedDct().transform_stack(stack), atol=1e-9
    )
    np.testing.assert_allclose(
        Dct().inverse_transform_stack(stack), StackedDct().inverse_transform_stack(stack), atol=1e-9
    )


def test_the_transform_changes_the_coefficients_and_nothing_else(
    picture: Path, tmp_path: Path
) -> None:
    """Same header, same size (it depends on the geometry alone), other payload."""
    walsh_bytes = _compress(Task(), picture, tmp_path / "walsh.cim")
    dct_bytes = _compress(Task(transform=Dct()), picture, tmp_path / "dct.cim")
    assert dct_bytes[:HEADER] == walsh_bytes[:HEADER]
    assert len(dct_bytes) == len(walsh_bytes)
    assert dct_bytes[HEADER:] != walsh_bytes[HEADER:]


def test_a_cim_does_not_record_its_transform(picture: Path, tmp_path: Path) -> None:
    """The documented caveat, pinned: the default decodes a DCT file without
    complaint, into a visibly worse picture than the right transform gives."""
    _compress(Task(transform=Dct()), picture, tmp_path / "dct.cim")
    right = _extract(Task(transform=Dct()), tmp_path / "dct.cim", tmp_path / "right.ppm")
    wrong = _extract(Task(), tmp_path / "dct.cim", tmp_path / "wrong.ppm")
    assert _psnr(picture, right) - _psnr(picture, wrong) > 6


def test_passing_the_default_explicitly_changes_nothing(picture: Path, tmp_path: Path) -> None:
    implicit = _compress(Task(), picture, tmp_path / "implicit.cim")
    explicit = _compress(
        Task(transform=WalshHadamardTransform()), picture, tmp_path / "explicit.cim"
    )
    assert implicit == explicit
    back_implicit = _extract(Task(), tmp_path / "implicit.cim", tmp_path / "a.ppm")
    back_explicit = _extract(
        Task(transform=WalshHadamardTransform()), tmp_path / "implicit.cim", tmp_path / "b.ppm"
    )
    assert back_implicit.read_bytes() == back_explicit.read_bytes()


def test_coeff_removal_by_the_task_equals_coeff_removal_by_the_transform(
    picture: Path, tmp_path: Path
) -> None:
    """Task applies the threshold itself now; the built-in class still can.
    The two routes must write the same bytes."""
    by_task = _compress(Task().with_coeff_removal(20.0), picture, tmp_path / "task.cim")
    by_transform = _compress(
        Task(transform=WalshHadamardTransform(coeff=20.0)), picture, tmp_path / "transform.cim"
    )
    assert by_task == by_transform
    assert by_task != _compress(Task(), picture, tmp_path / "plain.cim")


def test_coeff_removal_works_for_a_custom_transform(picture: Path, tmp_path: Path) -> None:
    def zeros(payload: bytes) -> int:
        return int((np.frombuffer(payload[HEADER:], dtype="<i2") == 0).sum())

    plain = _compress(Task(transform=Dct()), picture, tmp_path / "plain.cim")
    thinned = _compress(
        Task(transform=Dct()).with_coeff_removal(50.0), picture, tmp_path / "thinned.cim"
    )
    assert zeros(thinned) > zeros(plain)
    coefficients = np.frombuffer(thinned[HEADER:], dtype="<i2")
    assert not np.any((coefficients != 0) & (np.abs(coefficients) < 49))


@pytest.mark.parametrize(
    "not_a_transform",
    [WalshHadamardTransform, b"dct", 8, lambda block: block, object()],
    ids=["the class", "bytes", "a number", "a function", "an object"],
)
def test_anything_but_a_transform_instance_is_rejected_at_construction(
    not_a_transform: Any,
) -> None:
    with pytest.raises(TypeError, match="must be a Transform instance"):
        Task(transform=not_a_transform)


def test_negative_coeff_removal_is_rejected_when_it_is_set() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        Task().with_coeff_removal(-1.0)
    Task().with_coeff_removal(0.0).with_coeff_removal(None)


class _Shrinks(Dct):
    def transform(self, src: Block) -> Block:
        return super().transform(src)[:-1]


class _Scalar(Dct):
    def transform(self, src: Block) -> Block:
        scalar: Any = np.float64(1.0)
        return scalar


@pytest.mark.parametrize("transform", [_Shrinks(), _Scalar()], ids=["smaller", "scalar"])
def test_a_block_that_comes_back_another_shape_is_rejected_by_name(
    transform: Transform, picture: Path, tmp_path: Path
) -> None:
    """A scalar would broadcast over the block without the explicit check."""
    output = tmp_path / "out.cim"
    with pytest.raises(ValueError, match=rf"{type(transform).__name__}\.transform returned shape"):
        _compress(Task(transform=transform), picture, output)
    assert not output.exists()


def test_a_stack_override_that_changes_the_shape_is_rejected_by_name(
    picture: Path, tmp_path: Path
) -> None:
    class BadForward(StackedDct):
        def transform_stack(self, stack: Block) -> Block:
            return super().transform_stack(stack)[1:]

    class BadInverse(StackedDct):
        def inverse_transform_stack(self, stack: Block) -> Block:
            return super().inverse_transform_stack(stack)[:, :4, :4]

    with pytest.raises(ValueError, match=r"BadForward\.transform_stack returned shape"):
        _compress(Task(transform=BadForward()), picture, tmp_path / "bad.cim")

    _compress(Task(transform=BadInverse()), picture, tmp_path / "good.cim")
    with pytest.raises(ValueError, match=r"BadInverse\.inverse_transform_stack returned shape"):
        _extract(Task(transform=BadInverse()), tmp_path / "good.cim", tmp_path / "back.ppm")
    assert not (tmp_path / "back.ppm").exists()


def test_the_base_class_stack_methods_loop_over_blocks() -> None:
    calls: list[tuple[int, ...]] = []

    class Negate(Transform):
        def transform(self, src: Block) -> Block:
            calls.append(src.shape)
            return -src

        def inverse_transform(self, src: Block) -> Block:
            return -src

    stack = np.arange(3 * 4 * 4, dtype=np.float64).reshape(3, 4, 4)
    np.testing.assert_array_equal(Negate().transform_stack(stack), -stack)
    np.testing.assert_array_equal(Negate().inverse_transform_stack(-stack), stack)
    assert calls == [(4, 4)] * 3

    empty = Negate().transform_stack(np.empty((0, 4, 4)))
    assert empty.shape == (0, 4, 4)
    with pytest.raises(ValueError, match=r"\(count, edge, edge\) stack"):
        Negate().transform_stack(np.zeros((4, 4)))


def test_the_built_in_transform_takes_a_stack_in_one_call() -> None:
    """The per-block default would undo 0.4.0's vectorisation."""
    seen: list[int] = []

    class Counting(WalshHadamardTransform):
        def transform(self, src: Block) -> Block:
            seen.append(np.asarray(src).ndim)
            return super().transform(src)

        def inverse_transform(self, src: Block) -> Block:
            seen.append(np.asarray(src).ndim)
            return super().inverse_transform(src)

    rng = np.random.default_rng(seed=43)
    stack = rng.uniform(0, 255, size=(50, 8, 8))
    transform = Counting()
    spectra = transform.transform_stack(stack)
    transform.inverse_transform_stack(spectra)
    assert seen == [3, 3]
    np.testing.assert_array_equal(spectra, WalshHadamardTransform().transform(stack))


def test_remove_small_coefficients_is_strict_and_does_not_alias() -> None:
    spectrum = np.array([[-5.0, 4.999], [5.0, 0.0]])
    thinned = remove_small_coefficients(spectrum, 5.0)
    assert thinned.tolist() == [[-5.0, 0.0], [5.0, 0.0]]
    assert thinned is not spectrum
    assert spectrum[0, 1] == 4.999


def test_the_command_line_does_not_offer_a_transform() -> None:
    """Library only, on purpose: the container cannot say which one wrote it."""
    for command in ("compress", "extract"):
        result = CliRunner().invoke(main, [command, "--help"])
        assert result.exit_code == 0
        assert "--transform" not in result.output


def _load_example() -> ModuleType:
    if not EXAMPLE.exists():  # pragma: no cover - a distribution without examples
        pytest.skip(f"example missing: {EXAMPLE}")
    spec = importlib.util.spec_from_file_location("compare_transforms", EXAMPLE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_the_comparison_example_runs_and_its_transforms_are_orthonormal(
    picture: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    example = _load_example()
    for edge in (8, 16):
        m = example.HartleyTransform().matrix(edge)
        np.testing.assert_allclose(m @ m.T, np.eye(edge), atol=1e-12)

    monkeypatch.setattr(sys, "argv", ["compare_transforms.py", str(picture)])
    example.main()
    table = capsys.readouterr().out
    for heading in example.TRANSFORMS:
        assert heading in table
    assert table.count(" dB") == len(example.TRANSFORMS) * len(example.KEPT_PER_AXIS)


# --- transforms by name (0.4.14) ---------------------------------------------


def _round_trip_psnr(transform: Transform | str, packed: int, picture: Path, tmp: Path) -> float:
    cim, back = tmp / f"{packed}.cim", tmp / f"{packed}.ppm"
    _compress(Task(transform=transform, packed_block_size=packed), picture, cim)
    _extract(Task(transform=transform, packed_block_size=packed), cim, back)
    return _psnr(picture, back)


@pytest.mark.parametrize(
    ("name", "transform_class"),
    [
        ("walsh", WalshHadamardTransform),
        ("dct", DiscreteCosineTransform),
        ("haar", HaarTransform),
        ("DCT", DiscreteCosineTransform),
        ("Haar", HaarTransform),
    ],
)
def test_a_name_means_an_instance_of_that_transform(
    name: str, transform_class: type[Transform], picture: Path, tmp_path: Path
) -> None:
    by_name = _compress(Task(transform=name), picture, tmp_path / "name.cim")
    by_instance = _compress(Task(transform=transform_class()), picture, tmp_path / "instance.cim")
    assert by_name == by_instance

    back_by_name = _extract(Task(transform=name), tmp_path / "name.cim", tmp_path / "a.ppm")
    back_by_instance = _extract(
        Task(transform=transform_class()), tmp_path / "name.cim", tmp_path / "b.ppm"
    )
    assert back_by_name.read_bytes() == back_by_instance.read_bytes()


def test_the_name_walsh_is_the_default(picture: Path, tmp_path: Path) -> None:
    assert _compress(Task(transform="walsh"), picture, tmp_path / "named.cim") == _compress(
        Task(), picture, tmp_path / "default.cim"
    )


def test_an_unknown_name_is_rejected_at_construction_with_the_known_ones() -> None:
    with pytest.raises(ValueError, match=r"unknown transform 'fourier'.*dct, haar, walsh"):
        Task(transform="fourier")


def test_every_named_transform_round_trips_and_the_dct_wins_on_a_smooth_picture(
    picture: Path, tmp_path: Path
) -> None:
    quality = {
        name: _round_trip_psnr(name, 4, picture, tmp_path) for name in ("walsh", "dct", "haar")
    }
    assert min(quality.values()) > 25
    assert quality["dct"] > quality["walsh"] + 1


def test_haar_ties_walsh_at_a_power_of_two_and_only_there(picture: Path, tmp_path: Path) -> None:
    """The first 2**k Walsh functions and the first 2**k Haar functions span
    the same piecewise-constant subspace, so keeping that many per axis gives
    the same picture up to coefficient rounding. At 3 and 6 they part ways."""
    for packed in (2, 4, 8):
        walsh = _round_trip_psnr("walsh", packed, picture, tmp_path)
        haar = _round_trip_psnr("haar", packed, picture, tmp_path)
        assert haar == pytest.approx(walsh, abs=0.15), packed
    for packed in (3, 6):
        walsh = _round_trip_psnr("walsh", packed, picture, tmp_path)
        haar = _round_trip_psnr("haar", packed, picture, tmp_path)
        assert abs(haar - walsh) > 0.3, packed


def test_coeff_removal_works_for_a_named_transform(picture: Path, tmp_path: Path) -> None:
    plain = _compress(Task(transform="haar"), picture, tmp_path / "plain.cim")
    thinned = _compress(
        Task(transform="haar").with_coeff_removal(50.0), picture, tmp_path / "thinned.cim"
    )
    assert (np.frombuffer(thinned[HEADER:], dtype="<i2") == 0).sum() > (
        np.frombuffer(plain[HEADER:], dtype="<i2") == 0
    ).sum()
