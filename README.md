# Walsh-hadamard transform

[![PyPI](https://img.shields.io/pypi/v/walsh.svg)](https://pypi.org/project/walsh/)
[![Python versions](https://img.shields.io/pypi/pyversions/walsh.svg)](https://pypi.org/project/walsh/)
[![Typed](https://img.shields.io/pypi/types/walsh)](https://github.com/oskar-j/walsh-hadamard-transform/blob/master/src/walsh/py.typed)
[![CI](https://github.com/oskar-j/walsh-hadamard-transform/actions/workflows/ci.yml/badge.svg)](https://github.com/oskar-j/walsh-hadamard-transform/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A5%2090%25-brightgreen)](https://github.com/oskar-j/walsh-hadamard-transform/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pepy/dt/walsh)](https://pepy.tech/project/walsh)
[![Stars](https://img.shields.io/github/stars/oskar-j/walsh-hadamard-transform)](https://github.com/oskar-j/walsh-hadamard-transform/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Compressing images with a Hadamard transform.

A transform from the 1920s that needs nothing but additions and subtractions,
turned into a complete image codec you can read in an afternoon. Every step of
it is exact, so the same picture compresses to the same bytes on every machine,
and seven file formats go in and come out. A DCT and a Haar transform are on board
to race it against, and more than seven hundred tests keep all of it honest.

![The sample image, before and after a compress and extract round trip](https://raw.githubusercontent.com/oskar-j/walsh-hadamard-transform/master/doc/sample_usage.jpg)

## Contents

<!-- toc -->
- [Description](#description)
  - [How it works, in one picture](#how-it-works-in-one-picture)
- [Contributing](#contributing)
  - [Contributors](#contributors)
- [Acknowledgement](#acknowledgement)
- [Installation](#installation)
  - [Development](#development)
- [How to run](#how-to-run)
  - [Command line](#command-line)
  - [Reading the PSNR figures](#reading-the-psnr-figures)
  - [As a library](#as-a-library)
    - [Skipping the `.cim` file](#skipping-the-cim-file)
    - [Looking at the vectors](#looking-at-the-vectors)
    - [Other transforms](#other-transforms)
    - [Writing your own](#writing-your-own)
  - [Examples](#examples)
- [Requirements](#requirements)
- [Development commands](#development-commands)
  - [Coverage](#coverage)
- [Releasing](#releasing)
- [File formats](#file-formats)
  - [Input and output](#input-and-output)
  - [PNG](#png)
  - [Pickled pixels](#pickled-pixels)
  - [The `.cim` container](#the-cim-container)
<!-- /toc -->

## Description

**From Wikipedia:** The Hadamard transform (also known as the *Walsh–Hadamard transform*, 
*Hadamard–Rademacher–Walsh transform*, *Walsh transform*, or *Walsh–Fourier transform*) is an example 
of a generalized class of Fourier transforms. It performs an orthogonal, symmetric, 
involutive, linear operation on 2m real numbers (or complex numbers, although the 
Hadamard matrices themselves are purely real).

The Hadamard transform can be regarded as being built out of *size-2 
discrete Fourier transforms* (DFTs), and is in fact equivalent to a 
multidimensional DFT of size `2 × 2 × ⋯ × 2 × 2`. It decomposes an 
arbitrary input vector into a superposition of *Walsh functions*.

The transform is named for the French mathematician Jacques Hadamard, 
the German-American mathematician Hans Rademacher, and the 
American mathematician Joseph L. Walsh. 

The Hadamard transform is also used in data encryption, as well as many signal processing 
and data compression algorithms, such as `JPEG XR` and `MPEG-4 AVC`. In video compression 
applications, it is usually used in the form of the sum of absolute transformed differences. 
It is also a crucial part of *Grover's algorithm* and *Shor's algorithm* in quantum computing. 

### How it works, in one picture

![How the Walsh-Hadamard transform sees a block of pixels: the eight Walsh functions, the 64 basis images with the 16 the codec keeps outlined, and one real block going through the codec](https://raw.githubusercontent.com/oskar-j/walsh-hadamard-transform/master/doc/how_it_works.png)

Top: the eight Walsh functions of an 8-sample block are square waves whose only
values are `+1/√8` and `-1/√8`, ordered by how often they change sign (their
*sequency*). Middle: two of them at a time, one across and one down, give the
64 patterns that any 8×8 block of pixels is a weighted mix of; the weights are
the block's coefficients, and the codec keeps only the 16 in the outlined
corner, the slow-changing patterns that carry most of a picture. Bottom: a real
block from the Blue Marble sample goes through the codec. Its 64 coefficients
are shown, the 16 that survive are rounded to integers, and the last panel is
what those 16 numbers reconstruct: the shape is there and the fine detail is
gone, which is the whole trade. Since the transform is exact, the coefficients
in that figure are the ones in the `.cim` file, on any machine.

The figure is generated by `examples/plot_transform.py` from the sample image
(needs the `demo` extra).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the checks CI
runs, and the conventions this codebase follows. Participation is covered by
the [Code of Conduct](CODE_OF_CONDUCT.md).

### Contributors

Everyone whose work is in the package, with thanks.

<table>
  <tr>
    <td align="center" width="180">
      <a href="https://github.com/oskar-j"><img src="https://avatars.githubusercontent.com/u/4602830?v=4&s=100" width="100" height="100" alt="Oskar Jarczyk"><br><b>Oskar Jarczyk</b></a><br>
      <sub><a href="https://github.com/oskar-j">@oskar-j</a></sub>
    </td>
    <td align="center" width="180">
      <a href="https://github.com/DYNOSuprovo"><img src="https://avatars.githubusercontent.com/u/143953131?v=4&s=100" width="100" height="100" alt="Suprovo Mallick"><br><b>Suprovo Mallick</b></a><br>
      <sub><a href="https://github.com/DYNOSuprovo">@DYNOSuprovo</a></sub>
    </td>
  </tr>
  <tr>
    <td align="center">🛠️ author and maintainer</td>
    <td align="center">🐛 three fixes in <a href="https://github.com/oskar-j/walsh-hadamard-transform/blob/master/CHANGELOG.md">0.5.3</a> (<a href="https://github.com/oskar-j/walsh-hadamard-transform/pull/72">#72</a>)</td>
  </tr>
</table>

A merged pull request earns a cell here: add yourself in the same PR, or the
maintainer will at the next release.

## Acknowledgement

This code is partially based on the solution from [ktisha/python2012](https://github.com/ktisha/python2012/tree/dee4beda8e22f3a66a3e31384d4b72ab66102e88/avereshchagin)

## Installation

Requires Python 3.10 or newer.

```
pip install walsh
```

or, with [uv](https://docs.astral.sh/uv/):

```
uv add walsh          # into a project
uv tool install walsh # just the command line tool
```

The example script additionally needs matplotlib and Pillow, which are the
`demo` extra: `pip install "walsh[demo]"`.

### Development

`uv.lock` is committed and CI installs from it with `--locked`, so a checkout
reproduces exactly the environment CI uses:

```
uv sync --group dev --all-extras
```

`--group dev` brings in pytest, ruff and mypy; `--all-extras` adds the `demo`
extra so `examples/roundtrip.py` runs too. Without uv:

```
pip install -e ".[demo]" -r requirements-dev.txt
```

## How to run

### Command line

```
walsh compress data/bmp/image.bmp data/cim/transformed.cim
walsh extract  data/cim/transformed.cim data/bmp/recreated.bmp
```

The format is taken from the filename suffix, so PPM works the same way, and a
picture can be compressed from one format and restored as another:

```
walsh compress photo.ppm out.cim
walsh extract  out.cim restored.bmp     # PPM in, BMP out
walsh extract  out.cim restored.png     # or PNG out
walsh extract  out.cim restored.tif     # or TIFF out
walsh extract  out.cim restored.pam     # or PAM out
walsh extract  out.cim restored.npy     # or a bare NumPy array
walsh extract  out.cim restored.pkl     # or a pickle of rows of (r, g, b) tuples
```

To see what the codec does to a picture without keeping the `.cim`, name a
picture as the output of `compress`. The picture is compressed and restored in
memory, and what is written is its lossy reconstruction, byte for byte what the
two commands above would have produced between them:

```
walsh compress photo.ppm photo_compressed.ppm
walsh compress photo.ppm photo_compressed.png    # or in any other format
```

The result is a picture like any other, as large as the original: it shows the
compression, it is not the compressed data. See
[Skipping the `.cim` file](#skipping-the-cim-file).

Pickled pixels go in the same way, and a flat list of them, which does not
carry its size, takes it from the command line:

```
walsh compress array.pkl  out.cim                            # a pickled NumPy array
walsh compress rows.pkl   out.cim                            # [[(r, g, b), ...], ...]
walsh compress pixels.pkl out.cim --width 400 --height 300   # [(r, g, b), ...]
```

A pickle is read through an allowlist and nothing in it is ever executed; see
[Pickled pixels](#pickled-pixels).

`compress` accepts `--packed-block-size` (how many low-frequency coefficients
per axis to keep -- lower is smaller and lossier), `--y-block-size`,
`--chroma-block-size`, `--coeff-removal`, and `--width` with `--height` for
input that cannot say how large it is. Add `-v`/`-vv` for progress
logging, and see `walsh compress -h` for the full list.

Writes are atomic: output goes to a temporary file beside the destination and
replaces it only on success, so a failed run leaves an existing file untouched.
`walsh` also refuses to write over its own input, since both pipelines read the
whole image before writing and would otherwise replace the original with a
lossy reconstruction of itself.

The `.cim` container counts each channel's blocks in a 16-bit field, so at the
default 8-pixel luma block an image must be under about 4.2 megapixels. Larger
images are refused with a message naming the block size that would fit them:
`--y-block-size 16` roughly quadruples the ceiling, at some cost in detail.

`--coeff-removal` is the second, independent lossy knob: spectral coefficients
smaller than the given magnitude are zeroed. It does not change the `.cim`
file's size, because the format stores a fixed count of `int16` values whether
or not they are zero, but it makes the result far more compressible. On
`data/ppm/earth.ppm`:

| `--coeff-removal` | non-zero coefficients | gzipped `.cim` | PSNR |
| --- | --- | --- | --- |
| *(unset)* | 48,121 / 60,000 | 59,404 B | 25.07 dB |
| 5 | 29,049 | 45,080 B | 25.06 dB |
| 25 | 14,769 | 27,924 B | 24.71 dB |
| 50 | 8,917 | 19,285 B | 23.85 dB |

That table was measured at the default block sizes, 8 for luma and 16 for
chroma, and the threshold is an **absolute** magnitude, so its effect depends
on them. The surviving low-frequency coefficients grow with the block edge,
and the same number prunes less at a larger one. The same `--coeff-removal
25` on `data/ppm/earth.ppm`, with luma and chroma blocks set equal:

| block edge | non-zero coefficients | zeroed | gzipped `.cim` |
| --- | --- | --- | --- |
| 8 | 88,277 → 18,292 | 79.3% | 94,658 → 32,235 B |
| 16 | 24,060 → 7,041 | 70.7% | 29,190 → 13,105 B |
| 32 | 6,921 → 2,617 | 62.2% | 9,435 → 5,192 B |
| 64 | 2,134 → 1,045 | 51.0% | 3,048 → 2,147 B |

A threshold tuned against the first table and then combined with a larger
`--y-block-size` has quietly stopped doing most of its work; retune it. Note
also that it thresholds *spectral coefficients*, never the Hadamard matrix,
whose entries all share one magnitude (0.35 at edge 8): a value at that scale
is a no-op.

### Reading the PSNR figures

**PSNR** is *peak signal-to-noise ratio*, the standard way to put a number on
how much a lossy codec changed an image. It compares the reconstruction against
the original pixel by pixel:

```
PSNR = 10 * log10(255**2 / MSE)
```

where `MSE` is the mean squared difference across every channel of every pixel,
and 255 is the largest value an 8-bit channel can hold. It is measured in
decibels, and **higher is better**: a perfect reconstruction has infinite PSNR,
and every 3 dB gained means the mean squared error was halved.

Because the scale is logarithmic, small-looking differences matter. Going from
23 dB to 25 dB is not an 8% improvement, it is roughly a 37% reduction in error
power. Equally, the near-identical 25.07 and 25.06 in the table above mean the
first step of coefficient removal cost essentially nothing.

Rough expectations for 8-bit images, though they vary by content:

| PSNR | Typically means |
| --- | --- |
| above 40 dB | differences invisible without pixel-peeping |
| 30-40 dB | good lossy compression, artefacts hard to spot |
| 25-30 dB | visible softening and blocking |
| below 25 dB | obvious degradation |

The figures here sit around 25 dB because the defaults are aggressive: each
8x8 luma block keeps 16 of its 64 coefficients and each 16x16 chroma block
keeps 16 of 256. Raise `--packed-block-size` for a gentler setting.

One caveat worth knowing: PSNR measures arithmetic difference, not perceived
quality. It is reproducible and easy to compare, which is why it is quoted here,
but two images with the same PSNR can look noticeably different — it under-
weights structured artefacts like block edges, which the eye picks out readily.
Treat it as a consistent yardstick for comparing settings of *this* codec rather
than an absolute measure of how good an image looks.

Exit codes follow the usual convention: `0` on success, `1` when the input
cannot be processed (not a 24-bit BMP, truncated, unreadable), and `2` for a
usage error such as a missing file or an unknown option.

### As a library

```python
from walsh import Codec

Codec().compress(input="data/bmp/image.bmp", output="out.cim").run()
Codec().extract(input="out.cim", output="back.bmp").run()
```

`compress` and `extract` say what to do and name the input and the output;
nothing is read or written until `run()`. The settings go on the `Codec`
itself, in the constructor or chained before `run()`:

```python
Codec(packed_block_size=2).with_coeff_removal(40).compress(
    input="photo.png", output="photo.cim"
).run()
```

Code written for 0.5.0 or earlier needs two small changes, listed under 0.5.1
in the [changelog](CHANGELOG.md).

#### Skipping the `.cim` file

Give `compress` a picture as its output and the `.cim` never reaches the disk:

```python
Codec().compress(input="data/ppm/earth.ppm", output="earth_compressed.ppm").run()
```

The picture goes through the whole codec in memory (colour conversion,
transform, the crop to the kept coefficients, the rounding to the container's
16-bit integers, and back), and its lossy reconstruction is written. It is the
same file, byte for byte, as compressing to a `.cim` and extracting that: the
decoder is handed the very bytes the `.cim` would have held. Every setting
applies, so it is also the short way to compare settings or transforms:

```python
for name in ("walsh", "dct", "haar"):
    Codec(transform=name).compress(input="earth.ppm", output=f"earth_{name}.ppm").run()
```

What decides is the output's suffix. A picture suffix (`.bmp`, `.png`, `.ppm`
and the rest of the table under [File formats](#file-formats)) writes the
reconstruction; anything else, `.cim` by convention, writes the container.

The picture written need not be the format of the picture read. The codec
works on pixels, which no format owns, so the output's suffix alone chooses the
writer, as it does for `extract`:

```python
Codec().compress(input="earth.ppm", output="earth_compressed.png").run()
Codec().compress(input="photo.png", output="photo_compressed.bmp").run()
```

All 36 pairs of the sample's formats are in the golden tests, each held to the
checked-in reconstruction for its target, so where a picture came from leaves
no trace in what is written.

#### Looking at the vectors

`Codec` goes from one file to another. `Vectorizer` stops in the middle, where
the picture is a table of numbers, and hands you the table:

```python
from walsh import Vectorizer

vectorizer = Vectorizer(transform="walsh").parse(file_name="data/png/earth.png").compute()

vectorizer.vectors  # int16, shape (3750, 16): one row per block
print(vectorizer.describe())
vectorizer.save(output_file_name="earth.cim")
```

```
picture       400 x 400
transform     WalshHadamardTransform
blocks        Y 8, Cb 16, Cr 16; 4 x 4 kept of each
vectors       3,750 of 16 (Y 2,500, Cb 625, Cr 625)
coefficients  60,000, 12.50% of the picture's samples; 48,122 non-zero (80.2%)
raw pixels    480,000 B
source file   301,514 B
compressed    120,026 B, 6.00 bits per pixel
reduction     75.0% smaller than the raw pixels, 60.2% smaller than the source file
PSNR          25.07 dB (mean squared error 202.13, largest error 143 of 255)
```

Each block of the picture becomes one vector: the coefficients the codec keeps
of it, low frequencies first. Every channel keeps the same number per block, so
they all fit one array, the luma blocks first, then Cb, then Cr. That is the
order of the `.cim` file, and the array is `int16` because the file is, so
`vectors.tobytes()` is exactly the file after its 26-byte header, and `save()`
writes the very `.cim` that `Codec().compress()` would.

| Call | What it does |
| --- | --- |
| `parse(file_name=...)` | Reads a picture in any supported format. `width=` and `height=` declare the size of a flat pickled list. |
| `load(file_name=...)` | Reads a `.cim`, which already is vectors, so nothing is left to compute. |
| `compute()` | Transforms the parsed picture into vectors. |
| `vectors` | The array itself, also reachable as `_vectors`. It is the object's state, not a copy. |
| `describe()` | Sizes, reduction, bits per pixel and PSNR, as a `CompressionStats`; `print()` it for the table above, or read its fields. |
| `reconstruct()` | The picture the vectors decode to, as a `(height, width, 3)` array. |
| `save(output_file_name=...)` | Writes the `.cim`. |

Because the vectors are the state, changing them changes everything after
them, which makes this a bench for experiments. Keep only each block's mean and
see what that costs:

```python
vectorizer.vectors[:, 1:] = 0
print(vectorizer.describe().psnr_db)  # 25.07 before, 19.37 now
vectorizer.save(output_file_name="earth_means.cim")
```

After `load()` there is no original picture to compare with, so `describe()`
reports no PSNR; and since a `.cim` does not record its transform,
`reconstruct()` inverts with whichever transform the `Vectorizer` was given.
The constructor takes what `Codec` takes, plus `coeff_removal=`.

#### Other transforms

`Codec` takes the block transform as a keyword, so another transform can reuse
the whole pipeline — the colour conversion, the padding, the crop to the
low-frequency corner, the container — with only the transform swapped. Three
ship with the package and are selected by name, in any case:

| Name | Transform | Notes |
| --- | --- | --- |
| `"walsh"` | Walsh-Hadamard, sequency ordered | The default, and what the `.cim` format and the `walsh` command mean. Exact arithmetic: byte-identical output on every platform. |
| `"dct"` | DCT-II, the transform inside JPEG | Best quality per byte on natural pictures. |
| `"haar"` | Haar wavelet | Block edges must be powers of two, which `Codec` requires anyway. |

```python
from walsh import Codec

Codec(transform="dct").compress(input="data/ppm/earth.ppm", output="dct.cim").run()
Codec(transform="dct").extract(input="dct.cim", output="back.ppm").run()
```

An unknown name is a `ValueError` that lists the known ones. `"dct"` and
`"haar"` are ordinary floating-point matrix products, accurate to rounding;
only `"walsh"` carries the bit-exactness guarantee.

> **The `.cim` does not record which transform wrote it.** A file written with
> anything but the default must be extracted by a `Codec` given the same
> transform. The `walsh` command never takes one, and will decode such a file
> without complaint into a degraded picture. This keyword is for experiments,
> not for files you hand to someone else.

`examples/compare_transforms.py` runs them over the Blue Marble sample. The
byte count depends on the geometry alone, so each row is a like-for-like
comparison of how much picture a transform packs into its first few
coefficients:

| Kept per axis | Bytes | Walsh-Hadamard | DCT-II | Haar | Hartley (custom) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 30,026 | 21.59 dB | 22.29 dB | 21.59 dB | 21.17 dB |
| 3 | 67,526 | 23.09 dB | 24.44 dB | 22.85 dB | 22.18 dB |
| 4 | 120,026 | 25.07 dB | 26.45 dB | 25.07 dB | 22.79 dB |
| 6 | 270,026 | 28.88 dB | 31.70 dB | 27.72 dB | 23.51 dB |
| 8 | 480,026 | 42.70 dB | 43.65 dB | 42.70 dB | 39.01 dB |

The DCT wins throughout, which is why JPEG uses it; Walsh-Hadamard needs no
multiplications and is exact. Haar ties Walsh-Hadamard wherever the kept size
is a power of two, and that is mathematics rather than coincidence: the first
2, 4 or 8 Walsh functions and the first 2, 4 or 8 Haar functions span the same
piecewise-constant subspace, so the two projections are the same picture.

#### Writing your own

The last column of that table is not in the package. Anything that subclasses
`Transform` can be passed as an instance, and for a separable orthonormal
transform `MatrixTransform` needs only the matrix:

```python
import numpy as np
from walsh import MatrixTransform, Codec


class Hartley(MatrixTransform):
    """cas(2*pi*i*k/n) / sqrt(n), with cas = cos + sin."""

    def matrix(self, size):
        angle = 2 * np.pi * np.outer(np.arange(size), np.arange(size)) / size
        return (np.cos(angle) + np.sin(angle)) / np.sqrt(size)


Codec(transform=Hartley()).compress(input="data/ppm/earth.ppm", output="hartley.cim").run()
```

It trails the others for an instructive reason: the codec keeps the top-left
corner of each spectrum, which assumes rows rise in frequency, and a Hartley
matrix puts half of its low frequencies in its *last* rows. A transform that
is not a matrix product subclasses `Transform` directly and implements
`transform` and `inverse_transform` for one square block; `Codec` calls
`transform_stack` and `inverse_transform_stack`, whose defaults loop over the
blocks, so override those when a whole `(count, edge, edge)` stack can go
through in one operation. `with_coeff_removal` is applied by the codec, so it
works for any transform.

### Examples

`examples/roundtrip.py` compresses the sample image, restores it, and plots
both images with their histograms side by side (needs the `demo` extra):

```
python examples/roundtrip.py
```

`examples/compare_transforms.py` prints the table above for any image, and
needs numpy only:

```
python examples/compare_transforms.py [image]
```

## Requirements

The package needs `numpy` and `click` -- every format is parsed by hand with
`struct`, PNG included, whose compression is the standard library's `zlib`, and
`click` powers the command line interface. `matplotlib` and
`Pillow` are needed only by the example script, and are declared as the `demo`
extra. Versions are pinned in `pyproject.toml`;
`requirements.txt`, `requirements-demo.txt` and `requirements-dev.txt` mirror
them for plain `pip install -r` workflows, and `tests/project/test_requirements_mirror.py`
fails if the two ever disagree. On pip 25.1 or newer,
`pip install -e ".[demo]" --group dev` reads the same groups straight from
`pyproject.toml` and needs no mirror at all.

## Development commands

```
uv run pytest                          # test suite
uv run pytest --cov --cov-report=term-missing   # with coverage
uv run ruff check .                    # lint
uv run ruff format .                   # format
uv run mypy                            # strict type check
uv build                               # sdist + wheel into dist/
```

CI runs exactly these on every pull request, plus the test suite against
Python 3.10 through 3.14.

### Coverage

Coverage is measured with branch coverage on, and **CI enforces a floor of 90%
on every supported Python version**. A pull request that drops below it fails
the `test` jobs, which are required checks on `master` — so the badge above
states what is actually guaranteed rather than a number that could drift.

Coverage is opt-in locally (`--cov`) so a plain `pytest` stays fast; CI always
passes it.

## Releasing

The version in `pyproject.toml` is the single source of truth. To cut a
release, bump it, add the matching `## [x.y.z]` section to `CHANGELOG.md`, and
merge to `master`. The release workflow then tags `v<version>`, creates a
GitHub Release with those notes, and publishes the sdist and wheel to
[PyPI](https://pypi.org/project/walsh/) using Trusted Publishing — no API token
is stored in this repository.

Merges that do not change the version are a no-op, since PyPI permanently
refuses to accept the same version twice.

## File formats

### Input and output

| Suffix | Format | Notes |
| --- | --- | --- |
| `.bmp` | Windows bitmap | 24-bit, single plane, uncompressed. Top-down (negative height) files are understood. |
| `.png` | Portable Network Graphics | 8-bit RGB is read and written, and 8-bit RGBA is read **when it is opaque throughout**, which most everyday PNGs are. All five row filters are undone, any number of `IDAT` chunks, every chunk's CRC checked; `gAMA`, `sRGB`, `iCCP`, text and the like are skipped. Real transparency (an alpha below 255, or a `tRNS` colour some pixel has), palette, greyscale, 16-bit and interlaced files are rejected by name. See [PNG](#png). |
| `.ppm`, `.pnm` | Netpbm portable pixmap | `P6` binary and `P3` ASCII are read; `P6` is written. Header comments are skipped and a `maxval` below 255 is rescaled. 16-bit samples are rejected. |
| `.npy` | NumPy array | The raw pixel matrix in NumPy's own container, for images that already live in an array. Read: `uint8` of shape `(height, width, 3)` as RGB, `(height, width)` or `(height, width, 1)` as greyscale, and `(height, width, 4)` as RGBA only when fully opaque. Other dtypes, other channel counts, transparency and CMYK are rejected by name; pickled files are refused from the header and never loaded. Written as `(height, width, 3)` `uint8`, so `numpy.load` reads it back as is. Channel order is RGB; a BGR array, as OpenCV produces, is `array[..., ::-1]`. |
| `.pkl`, `.pickle` | Pickled pixels | A pickled NumPy array, rows of `(r, g, b)` pixels, or a flat list of them with a declared size. Read through an allowlist, so **nothing in the file is ever executed**; rows of `(r, g, b)` tuples are written. See [Pickled pixels](#pickled-pixels). |
| `.pam` | Netpbm portable arbitrary map | `P7` with `DEPTH 3`, `TUPLTYPE RGB` (or none) and `MAXVAL` up to 255 is read and written; a lower `maxval` is rescaled. Header keys may come in any order, comment and blank lines are skipped. Greyscale, alpha, other tuple types and 16-bit samples are rejected by name. The writer's output is byte-identical to Netpbm's own `pamtopam`. |
| `.tif`, `.tiff` | Uncompressed baseline TIFF | Both byte orders and multi-strip files are read; little-endian single-strip is written. Only the uncompressed RGB 8-bit chunky profile is supported -- LZW, palette, CMYK, greyscale, 16-bit, planar and rotated files are rejected by name. |

Every reader presents the same in-memory view -- RGB pixels, top row first --
whatever the file itself stores. BMP is the awkward one on both counts, storing
blue-green-red samples in bottom-up rows, and `BMPImage` converts in each
direction. That shared contract is what makes cross-format conversion work.

### PNG

PNG is the one compressed format here, and the only one most people have
pictures in. It costs the package nothing: a PNG's pixel rows sit in a zlib
stream, the inflater is `zlib` in the standard library, and everything around
it (chunks, CRCs, the five row filters) is read by hand, so there is still no
image library behind `walsh`. The compression is lossless, so a PNG reaches the
transform as exactly the pixels a PPM of the same picture would:
`data/png/earth.png`, written by libpng, compresses to a `.cim` byte-identical
to the one from `data/ppm/earth.ppm`.

```
walsh compress photo.png out.cim
walsh extract  out.cim restored.png
```

Three of the five filters predict a byte from the pixel to its left, which was
itself predicted, so a row cannot be undone in one step and the textbook
decoder is a loop over every byte. Here the image is undone one *anti-diagonal*
at a time instead: a pixel needs only its left, upper and upper-left
neighbours, all of which lie on the two diagonals before its own, so each
diagonal is a single array operation. A 2000x2000 PNG from libpng loads in
about half a second, where a byte loop in Python takes five to twelve.

Transparency is refused, not flattened. Dropping an alpha channel means
choosing a background to put behind it, and nothing in the file says which, so
an RGBA file is read only when every pixel is opaque, and the message counts
the pixels that are not:

```
$ walsh compress logo.png out.cim
Error: PNG has real transparency: alpha is below 255 in 1840 of 65536 pixels; only RGBA that is opaque throughout is supported
```

The files written are 8-bit RGB with a filter chosen per row, the same choice
libpng makes, which keeps them 14% smaller on the samples here and half the
size on a smooth picture. Their *bytes* are not reproducible from one machine
to the next, because DEFLATE output may differ between zlib builds. Their
pixels are, and that is how `data/png/recreated.png` is pinned.

A small file cannot cost much memory: inflation stops at the size the header
declares, and nothing is allocated from the header, only from what the stream
delivers.

### Pickled pixels

`pickle.load` and `numpy.load(allow_pickle=True)` run the program a pickle
contains, with the power to import any module and call anything in it, so
opening an untrusted pickle is running untrusted code. This package never
does that. It reads the same files through an allowlist: lists, tuples, dicts,
numbers and bytes need no lookups at all, and the only names a file may refer
to are the handful NumPy's own pickles use to rebuild an array. Anything else
is refused by name before it is called:

```
$ walsh compress evil.pkl out.cim
Error: unsupported pickle: it refers to posix.system; only lists, tuples, integers and NumPy arrays are read, and nothing in a pickle is ever executed
```

What a `.pkl` or `.pickle` may hold:

| Content | Size comes from | Notes |
| --- | --- | --- |
| A NumPy array | its shape | The `.npy` rules: `uint8`; `(h, w, 3)` RGB, `(h, w)` or `(h, w, 1)` greyscale, `(h, w, 4)` RGBA only when fully opaque. Every pickle protocol, and pickles written under NumPy 1 and NumPy 2 alike, whichever is installed. |
| Rows of pixels, `[[(r, g, b), ...], ...]` | its structure | |
| `{"width": w, "height": h, "pixels": [...]}` | the dict | Around a flat list, or around anything above, which must then agree. |
| A flat list of pixels, `[(r, g, b), ...]` | `--width` and `--height`, or `Codec.with_input_size(w, h)` | Top row first. It does not say how wide the picture is and nothing here guesses: 160,000 pixels could be 400x400 or 200x800. |

Lists and tuples are interchangeable at every level. Samples must be integers
in 0-255, Python's or NumPy's; floats, booleans, out-of-range values and ragged
rows are refused by name. A declared size is never silently dropped: input that
carries its own size, in any format, must match it.

```python
from walsh import Codec

Codec().with_input_size(400, 300).compress(input="pixels.pkl", output="out.cim").run()
```

An `object` array saved by `numpy.save`, which only
`numpy.load(allow_pickle=True)` opens, is a pickle inside a `.npy` and is read
the same way. The writer produces rows of `(r, g, b)` tuples of plain `int` at
protocol 4: any Python loads that without NumPy, and its bytes do not depend on
which NumPy wrote it.

### The `.cim` container

`compress` writes a `.cim` file: an atypical, project-specific container, so
most commercial tools will not be able to read it. It stores the image
dimensions, three block-layout descriptions (Y, Cb, Cr), and the retained
Walsh-Hadamard coefficients as little-endian `int16`.

> **Note.** `.cim` files written by 0.1.x are not compatible with 0.2.0. The
> in-memory pixel contract changed, so an old file extracted with 0.2.0 comes
> back with red and blue swapped and vertically flipped. Re-compress from the
> source image instead.
