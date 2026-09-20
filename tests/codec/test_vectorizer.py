"""`Vectorizer`: a picture as the coefficient vectors the codec keeps of it.

    Vectorizer(transform=...).parse(file_name=...).compute()

The claims held here are that the vectors are exactly what the `.cim` file
holds, that everything the object reports follows the vectors when they are
changed, and that its figures agree with ones worked out independently.
"""

from __future__ import annotations

import io
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from conftest import Sample, gradient_pixels, write_bmp, write_png, write_ppm
from walsh import (
    Codec,
    CompressionStats,
    UnsupportedFileFormatError,
    Vectorizer,
    reader_for,
)
from walsh.image import BlockDescription, CustomizableImage

WIDTH, HEIGHT = 40, 24
HEADER = 26


def _picture() -> list[tuple[int, int, int]]:
    rng = np.random.default_rng(11)
    noise = rng.integers(0, 24, size=(WIDTH * HEIGHT, 3))
    return [
        (int(min(255, r + n[0])), int(min(255, g + n[1])), int(min(255, b + n[2])))
        for (r, g, b), n in zip(gradient_pixels(WIDTH, HEIGHT), noise, strict=True)
    ]


@pytest.fixture
def picture(tmp_path: Path) -> Path:
    return write_ppm(tmp_path / "picture.ppm", WIDTH, HEIGHT, _picture())


def _pixels(path: Path) -> np.ndarray:
    image = reader_for(str(path))
    image.load(str(path))
    return image.get_array()


def _cim(codec: Codec, source: Path, output: Path) -> bytes:
    codec.compress(input=str(source), output=str(output)).run()
    return output.read_bytes()


# --- the vectors are the file -------------------------------------------------


def test_the_vectors_are_the_cim_after_its_header(picture: Path, tmp_path: Path) -> None:
    vectorizer = Vectorizer().parse(file_name=str(picture)).compute()
    vectors = vectorizer.vectors

    # 40x24 in 8-pixel luma blocks is 5x3, and in 16-pixel chroma blocks 3x2.
    assert vectors.dtype == np.int16
    assert vectors.shape == (15 + 6 + 6, 16)
    assert vectorizer._vectors is vectors, "`_vectors` is the array itself, not a copy"

    expected = _cim(Codec(), picture, tmp_path / "expected.cim")
    assert len(expected) == HEADER + vectors.nbytes
    assert vectors.astype("<i2").tobytes() == expected[HEADER:]


def test_the_rows_run_luma_then_cb_then_cr_each_row_major(picture: Path, tmp_path: Path) -> None:
    vectors = Vectorizer().parse(file_name=str(picture)).compute().vectors
    _cim(Codec(), picture, tmp_path / "c.cim")
    container = CustomizableImage.load(str(tmp_path / "c.cim"))

    start = 0
    for channel in ("y", "cb", "cr"):
        kept = container.get_stack(channel)[:, :4, :4]
        rows = vectors[start : start + len(kept)]
        assert np.array_equal(rows, kept.reshape(len(kept), 16)), channel
        start += len(kept)
    assert start == len(vectors)


@pytest.mark.parametrize(
    "settings",
    [
        {},
        {"packed_block_size": 2},
        {"y_block_size": 16, "cb_block_size": 32, "cr_block_size": 32, "packed_block_size": 6},
        {"transform": "dct"},
        {"transform": "haar", "packed_block_size": 3},
    ],
    ids=["defaults", "packed-2", "larger-blocks", "dct", "haar-packed-3"],
)
def test_saving_writes_what_the_codec_writes(
    picture: Path, tmp_path: Path, settings: dict[str, Any]
) -> None:
    saved = tmp_path / "saved.cim"
    Vectorizer(**settings).parse(file_name=str(picture)).compute().save(output_file_name=str(saved))
    assert saved.read_bytes() == _cim(Codec(**settings), picture, tmp_path / "expected.cim")


def test_cb_and_cr_keep_their_places_when_their_blocks_differ(
    picture: Path, tmp_path: Path
) -> None:
    """With equal chroma blocks the two channels are interchangeable in every
    count, so only unequal ones show whether the rows and the descriptions
    are paired up in the file's order: luma, then Cb, then Cr."""
    settings = {"cb_block_size": 8, "cr_block_size": 16}
    vectorizer = Vectorizer(**settings).parse(file_name=str(picture)).compute()
    stats = vectorizer.describe()
    assert (stats.y_vectors, stats.cb_vectors, stats.cr_vectors) == (15, 15, 6)
    assert (stats.cb_block_size, stats.cr_block_size) == (8, 16)

    vectorizer.save(output_file_name=str(tmp_path / "saved.cim"))
    expected = _cim(Codec(**settings), picture, tmp_path / "expected.cim")
    assert (tmp_path / "saved.cim").read_bytes() == expected

    Codec(**settings).extract(
        input=str(tmp_path / "expected.cim"), output=str(tmp_path / "back.ppm")
    ).run()
    assert np.array_equal(vectorizer.reconstruct(), _pixels(tmp_path / "back.ppm"))

    loaded = Vectorizer().load(file_name=str(tmp_path / "expected.cim"))
    assert np.array_equal(loaded.vectors, vectorizer.vectors)


def test_coefficient_removal_is_applied_by_compute(picture: Path, tmp_path: Path) -> None:
    plain = Vectorizer().parse(file_name=str(picture)).compute()
    thinned = Vectorizer(coeff_removal=40.0).parse(file_name=str(picture)).compute()
    assert np.count_nonzero(thinned.vectors) < np.count_nonzero(plain.vectors)

    saved = tmp_path / "thinned.cim"
    thinned.save(output_file_name=str(saved))
    expected = _cim(Codec().with_coeff_removal(40.0), picture, tmp_path / "expected.cim")
    assert saved.read_bytes() == expected


def test_loading_a_cim_gives_the_vectors_that_wrote_it(picture: Path, tmp_path: Path) -> None:
    computed = Vectorizer().parse(file_name=str(picture)).compute()
    computed.save(output_file_name=str(tmp_path / "a.cim"))

    loaded = Vectorizer().load(file_name=str(tmp_path / "a.cim"))
    assert np.array_equal(loaded.vectors, computed.vectors)
    loaded.save(output_file_name=str(tmp_path / "b.cim"))
    assert (tmp_path / "b.cim").read_bytes() == (tmp_path / "a.cim").read_bytes()


def test_the_checked_in_cim_loads_and_saves_unchanged(sample: Sample, tmp_path: Path) -> None:
    reference = sample("transformed_earth.cim")
    vectorizer = Vectorizer().load(file_name=str(reference))
    assert vectorizer.vectors.shape == (2500 + 625 + 625, 16)
    vectorizer.save(output_file_name=str(tmp_path / "again.cim"))
    assert (tmp_path / "again.cim").read_bytes() == reference.read_bytes()


def test_reconstruct_is_what_extracting_the_saved_cim_gives(picture: Path, tmp_path: Path) -> None:
    vectorizer = Vectorizer(packed_block_size=3).parse(file_name=str(picture)).compute()
    vectorizer.save(output_file_name=str(tmp_path / "v.cim"))
    Codec().extract(input=str(tmp_path / "v.cim"), output=str(tmp_path / "v.ppm")).run()

    reconstructed = vectorizer.reconstruct()
    assert reconstructed.dtype == np.uint8 and reconstructed.shape == (HEIGHT, WIDTH, 3)
    assert np.array_equal(reconstructed, _pixels(tmp_path / "v.ppm"))


# --- describe -----------------------------------------------------------------


def test_describe_agrees_with_figures_worked_out_separately(picture: Path, tmp_path: Path) -> None:
    stats = Vectorizer().parse(file_name=str(picture)).compute().describe()
    assert isinstance(stats, CompressionStats)

    container = _cim(Codec(), picture, tmp_path / "c.cim")
    Codec().extract(input=str(tmp_path / "c.cim"), output=str(tmp_path / "back.ppm")).run()
    error = _pixels(picture).astype(float) - _pixels(tmp_path / "back.ppm").astype(float)
    mse = float((error**2).mean())

    assert (stats.width, stats.height) == (WIDTH, HEIGHT)
    assert stats.transform == "WalshHadamardTransform"
    assert (stats.y_block_size, stats.cb_block_size, stats.cr_block_size) == (8, 16, 16)
    assert (stats.y_vectors, stats.cb_vectors, stats.cr_vectors) == (15, 6, 6)
    assert (stats.packed_block_size, stats.vector_length) == (4, 16)
    assert stats.coefficients == 27 * 16
    assert stats.kept_share == pytest.approx(100 * 27 * 16 / (WIDTH * HEIGHT * 3))
    assert stats.raw_bytes == WIDTH * HEIGHT * 3
    assert stats.source_bytes == picture.stat().st_size
    assert stats.compressed_bytes == len(container) == HEADER + 27 * 16 * 2
    assert stats.reduction_percent == pytest.approx(100 * (1 - len(container) / stats.raw_bytes))
    assert stats.source_reduction_percent == pytest.approx(
        100 * (1 - len(container) / picture.stat().st_size)
    )
    assert stats.bits_per_pixel == pytest.approx(8 * len(container) / (WIDTH * HEIGHT))
    assert stats.mean_squared_error == pytest.approx(mse)
    assert stats.psnr_db == pytest.approx(10 * math.log10(255**2 / mse))
    assert stats.max_error == int(np.abs(error).max())


def test_the_sample_describes_as_the_readme_says(sample: Sample) -> None:
    """25.07 dB is the figure data/README.md gives for the Blue Marble."""
    stats = Vectorizer().parse(file_name=str(sample("earth.ppm"))).compute().describe()
    assert stats.psnr_db == pytest.approx(25.07, abs=0.005)
    assert stats.compressed_bytes == 120_026
    assert stats.reduction_percent == pytest.approx(75.0, abs=0.01)
    assert stats.bits_per_pixel == pytest.approx(6.0, abs=0.01)


def test_the_table_reads_as_a_table(picture: Path) -> None:
    text = str(Vectorizer().parse(file_name=str(picture)).compute().describe())
    lines = {line[:13].strip(): line[14:] for line in text.splitlines()}
    assert lines["picture"] == "40 x 24"
    assert lines["blocks"] == "Y 8, Cb 16, Cr 16; 4 x 4 kept of each"
    assert lines["vectors"] == "27 of 16 (Y 15, Cb 6, Cr 6)"
    assert lines["compressed"].startswith("890 B, 7.42 bits per pixel")
    assert lines["reduction"].startswith("69.1% smaller than the raw pixels, 69.")
    assert lines["reduction"].endswith("smaller than the source file")
    assert lines["PSNR"].endswith("of 255)") and " dB (mean squared error " in lines["PSNR"]
    assert list(lines) == [
        "picture",
        "transform",
        "blocks",
        "vectors",
        "coefficients",
        "raw pixels",
        "source file",
        "compressed",
        "reduction",
        "PSNR",
    ]
    assert all(line[13] == " " for line in text.splitlines()), "values start in one column"


def test_a_loaded_cim_has_no_original_so_no_psnr(picture: Path, tmp_path: Path) -> None:
    _cim(Codec(), picture, tmp_path / "c.cim")
    stats = Vectorizer().load(file_name=str(tmp_path / "c.cim")).describe()
    assert (stats.psnr_db, stats.mean_squared_error, stats.max_error) == (None, None, None)
    assert (stats.source_bytes, stats.source_reduction_percent) == (None, None)
    assert stats.compressed_bytes == 890

    text = str(stats)
    assert "PSNR          unknown: loaded from a .cim" in text
    assert "source file   none" in text
    assert "source file" not in text.split("reduction")[1]


def test_a_flat_picture_comes_back_exactly(tmp_path: Path) -> None:
    flat = write_ppm(tmp_path / "flat.ppm", 16, 16, [(128, 128, 128)] * 256)
    stats = Vectorizer().parse(file_name=str(flat)).compute().describe()
    assert stats.psnr_db == math.inf and stats.max_error == 0
    assert "infinite: the reconstruction equals the original" in str(stats)


def test_a_compressed_source_can_be_smaller_than_the_cim(tmp_path: Path) -> None:
    """A flat PNG is a few dozen bytes, which the fixed-size .cim cannot beat,
    and the table says "larger" rather than printing a negative reduction."""
    flat = write_png(tmp_path / "flat.png", 32, 32, [(10, 200, 90)] * 1024)
    stats = Vectorizer().parse(file_name=str(flat)).compute().describe()
    assert stats.source_reduction_percent is not None and stats.source_reduction_percent < 0
    assert "% larger than the source file" in str(stats)


def test_the_geometry_described_after_load_is_the_files(picture: Path, tmp_path: Path) -> None:
    _cim(Codec(y_block_size=4, packed_block_size=2), picture, tmp_path / "c.cim")
    stats = Vectorizer(y_block_size=8, packed_block_size=4).load(str(tmp_path / "c.cim")).describe()
    assert (stats.y_block_size, stats.packed_block_size, stats.vector_length) == (4, 2, 4)


def test_a_cim_with_no_blocks_still_describes(tmp_path: Path) -> None:
    empty = CustomizableImage()
    empty.set_dimensions(8, 8)
    empty.set_descriptions(*[BlockDescription(8, 4, 0)] * 3)
    empty.save(str(tmp_path / "empty.cim"))

    vectorizer = Vectorizer().load(file_name=str(tmp_path / "empty.cim"))
    assert vectorizer.vectors.shape == (0, 16)
    stats = vectorizer.describe()
    assert (stats.coefficients, stats.compressed_bytes) == (0, HEADER)
    assert "0, 0.00% of the picture's samples; 0 non-zero (0.0%)" in str(stats)
    assert vectorizer.reconstruct().shape == (8, 8, 3)


# --- the vectors are the state ------------------------------------------------


def test_changing_the_vectors_changes_everything_downstream(picture: Path, tmp_path: Path) -> None:
    vectorizer = Vectorizer().parse(file_name=str(picture)).compute()
    before = vectorizer.describe()
    picture_before = vectorizer.reconstruct()

    vectorizer.vectors[:, 1:] = 0  # keep each block's mean and nothing else
    after = vectorizer.describe()

    assert after.nonzero_coefficients <= 27 < before.nonzero_coefficients
    assert after.psnr_db is not None and before.psnr_db is not None
    assert after.psnr_db < before.psnr_db
    assert after.compressed_bytes == before.compressed_bytes, "zeros take as much room"
    assert not np.array_equal(vectorizer.reconstruct(), picture_before)

    vectorizer.save(output_file_name=str(tmp_path / "means.cim"))
    saved = np.frombuffer((tmp_path / "means.cim").read_bytes()[HEADER:], dtype="<i2")
    assert not saved.reshape(-1, 16)[:, 1:].any()
    Codec().extract(input=str(tmp_path / "means.cim"), output=str(tmp_path / "means.ppm")).run()
    assert np.array_equal(_pixels(tmp_path / "means.ppm"), vectorizer.reconstruct())


def test_computing_again_starts_over_and_after_a_load_does_nothing(
    picture: Path, tmp_path: Path
) -> None:
    vectorizer = Vectorizer().parse(file_name=str(picture)).compute()
    fresh = vectorizer.vectors.copy()
    vectorizer.vectors[:] = 0
    assert np.array_equal(vectorizer.compute().vectors, fresh)

    vectorizer.save(output_file_name=str(tmp_path / "c.cim"))
    loaded = Vectorizer().load(file_name=str(tmp_path / "c.cim"))
    loaded.vectors[:] = 7
    assert loaded.compute() is loaded
    assert (loaded.vectors == 7).all(), "there is no picture to compute from, so nothing moved"


@pytest.mark.parametrize(
    ("replacement", "found"),
    [
        (np.zeros((27, 16), dtype=np.float64), r"float64 \(27, 16\)"),
        (np.zeros((27, 4), dtype=np.int16), r"int16 \(27, 4\)"),
        (np.zeros((26, 16), dtype=np.int16), r"int16 \(26, 16\)"),
        ([[0] * 16] * 27, "list"),
    ],
    ids=["floats", "short-vectors", "a-row-missing", "a-list"],
)
def test_vectors_replaced_by_something_that_does_not_fit_are_refused(
    picture: Path, tmp_path: Path, replacement: Any, found: str
) -> None:
    vectorizer = Vectorizer().parse(file_name=str(picture)).compute()
    vectorizer._vectors = replacement
    message = rf"must be int16 of shape \(27, 16\), and are {found}"
    for use in (
        lambda: vectorizer.vectors,
        vectorizer.describe,
        vectorizer.reconstruct,
        lambda: vectorizer.save(output_file_name=str(tmp_path / "bad.cim")),
    ):
        with pytest.raises(ValueError, match=message):
            use()
    assert not (tmp_path / "bad.cim").exists()


def test_vectors_replaced_by_an_array_that_fits_are_used(picture: Path) -> None:
    vectorizer = Vectorizer().parse(file_name=str(picture)).compute()
    vectorizer._vectors = np.zeros((27, 16), dtype=np.int16)
    assert vectorizer.describe().nonzero_coefficients == 0


# --- order of calls -----------------------------------------------------------


def test_nothing_works_before_there_are_vectors(picture: Path, tmp_path: Path) -> None:
    empty = Vectorizer()
    assert empty._vectors is None
    with pytest.raises(ValueError, match=r"nothing to compute; call parse\(\) or load\(\)"):
        empty.compute()

    parsed = Vectorizer().parse(file_name=str(picture))
    assert parsed._vectors is None, "parse() reads; compute() transforms"
    for vectorizer in (empty, parsed):
        for use in (
            lambda v=vectorizer: v.vectors,
            vectorizer.describe,
            vectorizer.reconstruct,
            lambda v=vectorizer: v.save(output_file_name=str(tmp_path / "none.cim")),
        ):
            with pytest.raises(ValueError, match="there are no vectors yet"):
                use()
    assert not (tmp_path / "none.cim").exists()


def test_every_step_returns_the_vectorizer(picture: Path, tmp_path: Path) -> None:
    vectorizer = Vectorizer()
    assert vectorizer.parse(file_name=str(picture)) is vectorizer
    assert vectorizer.compute() is vectorizer
    assert vectorizer.save(output_file_name=str(tmp_path / "c.cim")) is vectorizer
    assert vectorizer.load(file_name=str(tmp_path / "c.cim")) is vectorizer


def test_a_new_parse_or_load_drops_what_was_held(picture: Path, tmp_path: Path) -> None:
    vectorizer = Vectorizer().parse(file_name=str(picture)).compute()
    vectorizer.save(output_file_name=str(tmp_path / "c.cim"))

    vectorizer.parse(file_name=str(picture))
    assert vectorizer._vectors is None

    vectorizer.compute().load(file_name=str(tmp_path / "c.cim"))
    assert vectorizer.describe().psnr_db is None, "the parsed picture went with the load"


# --- the wrong file in the wrong place ----------------------------------------


def test_parse_sends_a_cim_to_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"is a \.cim, which already holds vectors: use load\(\)"):
        Vectorizer().parse(file_name=str(tmp_path / "anything.CIM"))


def test_load_refuses_a_picture(picture: Path) -> None:
    with pytest.raises(UnsupportedFileFormatError):
        Vectorizer().load(file_name=str(picture))


@pytest.mark.parametrize("name", ["out.png", "out.PPM", "out.tiff"])
def test_save_refuses_a_picture_name(picture: Path, tmp_path: Path, name: str) -> None:
    vectorizer = Vectorizer().parse(file_name=str(picture)).compute()
    with pytest.raises(ValueError, match=r"save\(\) writes the \.cim container"):
        vectorizer.save(output_file_name=str(tmp_path / name))
    assert not (tmp_path / name).exists()


def test_a_failed_parse_leaves_what_was_held(picture: Path, tmp_path: Path) -> None:
    vectorizer = Vectorizer().parse(file_name=str(picture)).compute()
    bogus = tmp_path / "bogus.ppm"
    bogus.write_bytes(b"not a pixmap")
    with pytest.raises(UnsupportedFileFormatError):
        vectorizer.parse(file_name=str(bogus))
    assert vectorizer.vectors.shape == (27, 16)


def test_a_cim_whose_channels_keep_different_amounts_is_refused(tmp_path: Path) -> None:
    """The format allows it and nothing here writes it; its vectors would be
    of two lengths, which is not one array."""
    uneven = CustomizableImage()
    uneven.set_dimensions(16, 16)
    uneven.set_descriptions(
        BlockDescription(8, 4, 4), BlockDescription(16, 2, 1), BlockDescription(16, 2, 1)
    )
    uneven.set_data(np.ones((4, 8, 8)), np.ones((1, 16, 16)), np.ones((1, 16, 16)))
    uneven.save(str(tmp_path / "uneven.cim"))

    with pytest.raises(UnsupportedFileFormatError, match=r"keeps \[2, 4\] coefficients per axis"):
        Vectorizer().load(file_name=str(tmp_path / "uneven.cim"))


# --- a declared size, and the standard streams --------------------------------


def test_a_flat_pickled_list_needs_its_size_and_gets_it_checked(
    picture: Path, tmp_path: Path
) -> None:
    flat = tmp_path / "flat.pkl"
    flat.write_bytes(pickle.dumps(_picture(), protocol=4))

    declared = Vectorizer().parse(file_name=str(flat), width=WIDTH, height=HEIGHT).compute()
    from_ppm = Vectorizer().parse(file_name=str(picture)).compute()
    assert np.array_equal(declared.vectors, from_ppm.vectors)

    with pytest.raises(ValueError, match="width and height must be declared together"):
        Vectorizer().parse(file_name=str(flat), width=WIDTH)
    with pytest.raises(ValueError, match=f"is {WIDTH}x{HEIGHT}, not the 24x40 declared"):
        Vectorizer().parse(file_name=str(picture), width=HEIGHT, height=WIDTH)
    with pytest.raises(ValueError, match="declared width must be a positive integer"):
        Vectorizer().parse(file_name=str(picture), width=0, height=HEIGHT)


def test_the_standard_streams_work_and_have_no_size_to_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfdbinary: pytest.CaptureFixture[bytes]
) -> None:
    bmp = write_bmp(tmp_path / "p.bmp", WIDTH, HEIGHT, _picture())
    stream = io.BytesIO(bmp.read_bytes())
    monkeypatch.setattr(sys, "stdin", type("Stdin", (), {"buffer": stream})())

    vectorizer = Vectorizer().parse(file_name=None).compute()
    assert vectorizer.describe().source_bytes is None

    vectorizer.save(output_file_name=None)
    assert capfdbinary.readouterr().out == _cim(Codec(), bmp, tmp_path / "expected.cim")


# --- the constructor ----------------------------------------------------------


def test_the_settings_are_the_codecs_and_are_checked_as_it_checks_them() -> None:
    with pytest.raises(ValueError, match="y_block_size must be a positive power of two"):
        Vectorizer(y_block_size=12)
    with pytest.raises(ValueError, match="coeff must be non-negative"):
        Vectorizer(coeff_removal=-1.0)
    with pytest.raises(ValueError, match="unknown transform"):
        Vectorizer(transform="fourier")
    not_a_transform: Any = object()
    with pytest.raises(TypeError, match="transform must be a Transform instance"):
        Vectorizer(transform=not_a_transform)


def test_the_transform_is_named_in_the_description(picture: Path) -> None:
    stats = Vectorizer(transform="DCT").parse(file_name=str(picture)).compute().describe()
    assert stats.transform == "DiscreteCosineTransform"
