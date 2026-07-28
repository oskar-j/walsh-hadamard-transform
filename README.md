# Walsh-hadamard transform

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

## Acknowledgement

This code is partially based on the solution from [ktisha/python2012](https://github.com/ktisha/python2012/tree/dee4beda8e22f3a66a3e31384d4b72ab66102e88/avereshchagin)

## Installation

Requires Python 3.10 or newer.

```
pip install .
```

For a development checkout, with the example script, tests and linters:

```
pip install -e ".[demo,dev]"
```

## How to run

### Command line

```
walsh compress data/image.bmp data/transformed.cim
walsh extract  data/transformed.cim data/recreated.bmp
```

`compress` accepts `--packed-block-size` (how many low-frequency coefficients
per axis to keep -- lower is smaller and lossier), `--y-block-size`,
`--chroma-block-size` and `--coeff-removal`. Add `-v`/`-vv` for progress
logging, and see `walsh --help` for the full list.

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

The package itself needs only `numpy` -- BMP parsing is done by hand with
`struct`. `matplotlib` and `Pillow` are needed only by the example script, and
are declared as the `demo` extra. Versions are pinned in `pyproject.toml`;
`requirements.txt`, `requirements-demo.txt` and `requirements-dev.txt` mirror
them for plain `pip install -r` workflows.

## Development

```
pytest             # test suite
ruff check .       # lint
ruff format .      # format
mypy               # strict type check
```

## File format

`compress` writes a `.cim` file: an atypical, project-specific container, so
most commercial tools will not be able to read it. It stores the image
dimensions, three block-layout descriptions (Y, Cb, Cr), and the retained
Walsh-Hadamard coefficients as little-endian `int16`.

## Effects

![https://raw.githubusercontent.com/oskar-j/walsh-hadamard-transform/master/doc/sample_usage.jpg](https://raw.githubusercontent.com/oskar-j/walsh-hadamard-transform/master/doc/sample_usage.jpg)
