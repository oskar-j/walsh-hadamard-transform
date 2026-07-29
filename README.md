# Walsh-hadamard transform

[![PyPI](https://img.shields.io/pypi/v/walsh.svg)](https://pypi.org/project/walsh/)
[![Python versions](https://img.shields.io/pypi/pyversions/walsh.svg)](https://pypi.org/project/walsh/)
[![CI](https://github.com/oskar-j/walsh-hadamard-transform/actions/workflows/ci.yml/badge.svg)](https://github.com/oskar-j/walsh-hadamard-transform/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-%E2%89%A5%2090%25-brightgreen)](https://github.com/oskar-j/walsh-hadamard-transform/actions/workflows/ci.yml)
[![Downloads](https://img.shields.io/pepy/dt/walsh)](https://pepy.tech/project/walsh)
[![Stars](https://img.shields.io/github/stars/oskar-j/walsh-hadamard-transform)](https://github.com/oskar-j/walsh-hadamard-transform/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Compressing images with a Hadamard transform

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the checks CI
runs, and the conventions this codebase follows.

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

`uv.lock` is committed, so a checkout reproduces exactly the environment CI
uses:

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
walsh compress data/image.bmp data/transformed.cim
walsh extract  data/transformed.cim data/recreated.bmp
```

The format is taken from the filename suffix, so PPM works the same way, and a
picture can be compressed from one format and restored as another:

```
walsh compress photo.ppm out.cim
walsh extract  out.cim restored.bmp     # PPM in, BMP out
walsh extract  out.cim restored.tif     # or TIFF out
```

`compress` accepts `--packed-block-size` (how many low-frequency coefficients
per axis to keep -- lower is smaller and lossier), `--y-block-size`,
`--chroma-block-size` and `--coeff-removal`. Add `-v`/`-vv` for progress
logging, and see `walsh -h` for the full list.

`--coeff-removal` is the second, independent lossy knob: spectral coefficients
smaller than the given magnitude are zeroed. It does not change the `.cim`
file's size, because the format stores a fixed count of `int16` values whether
or not they are zero, but it makes the result far more compressible. On
`data/earth.ppm`:

| `--coeff-removal` | non-zero coefficients | gzipped `.cim` | PSNR |
| --- | --- | --- | --- |
| *(unset)* | 48,121 / 60,000 | 59,404 B | 25.07 dB |
| 5 | 29,049 | 45,080 B | 25.06 dB |
| 25 | 14,769 | 27,924 B | 24.71 dB |
| 50 | 8,917 | 19,285 B | 23.85 dB |

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
from walsh import Task

Task().with_action("compress").with_input("data/image.bmp").with_output("out.cim").run()
Task().with_action("extract").with_input("out.cim").with_output("back.bmp").run()
```

### Example

`examples/roundtrip.py` compresses the sample image, restores it, and plots
both images with their histograms side by side (needs the `demo` extra):

```
python examples/roundtrip.py
```

## Requirements

The package needs `numpy` and `click` -- BMP parsing is done by hand with
`struct`, and `click` powers the command line interface. `matplotlib` and
`Pillow` are needed only by the example script, and are declared as the `demo`
extra. Versions are pinned in `pyproject.toml`;
`requirements.txt`, `requirements-demo.txt` and `requirements-dev.txt` mirror
them for plain `pip install -r` workflows.

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
| `.ppm`, `.pnm` | Netpbm portable pixmap | `P6` binary and `P3` ASCII are read; `P6` is written. Header comments are skipped and a `maxval` below 255 is rescaled. 16-bit samples are rejected. |
| `.tif`, `.tiff` | Uncompressed baseline TIFF | Both byte orders and multi-strip files are read; little-endian single-strip is written. Only the uncompressed RGB 8-bit chunky profile is supported -- LZW, palette, CMYK, greyscale, 16-bit, planar and rotated files are rejected by name. |

Every reader presents the same in-memory view -- RGB pixels, top row first --
whatever the file itself stores. BMP is the awkward one on both counts, storing
blue-green-red samples in bottom-up rows, and `BMPImage` converts in each
direction. That shared contract is what makes cross-format conversion work.

### The `.cim` container

`compress` writes a `.cim` file: an atypical, project-specific container, so
most commercial tools will not be able to read it. It stores the image
dimensions, three block-layout descriptions (Y, Cb, Cr), and the retained
Walsh-Hadamard coefficients as little-endian `int16`.

> **Note.** `.cim` files written by 0.1.x are not compatible with 0.2.0. The
> in-memory pixel contract changed, so an old file extracted with 0.2.0 comes
> back with red and blue swapped and vertically flipped. Re-compress from the
> source image instead.

## Effects

![https://raw.githubusercontent.com/oskar-j/walsh-hadamard-transform/master/doc/sample_usage.jpg](https://raw.githubusercontent.com/oskar-j/walsh-hadamard-transform/master/doc/sample_usage.jpg)
