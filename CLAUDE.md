# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```
pip install -e ".[demo,dev]"   # dev setup; requires Python 3.10+
pytest                          # full suite
pytest tests/test_task.py -k roundtrip   # single test / pattern
ruff check . && ruff format .   # lint + format
mypy                            # strict; config selects the walsh package
python -m build                 # sdist + wheel
```

Run the codec end to end:

```
walsh compress data/image.bmp out.cim
walsh extract out.cim back.bmp
python examples/roundtrip.py    # same thing plus matplotlib plots
```

The local `.venv` is Python 3.12. `walsh` is installed as a console script
(`walsh.cli:main`).

## Layout

`src/` layout, package name `walsh`, built with setuptools. Tests import the
installed package, so an editable install must exist before `pytest` will work.
`tests/conftest.py` exposes a `write_bmp` helper plus `gradient_bmp` and
`sample_bmp` fixtures; test modules import the helper as `from conftest import ...`,
which works because pytest puts `tests/` on `sys.path`.

## Architecture

`walsh.task` orchestrates; the other modules are layers under it.

**`task.py`** — `Task` is a fluent builder (`with_action`, `with_input`,
`with_output`, `with_coeff_removal`, then `run()`). Actions are the `Action`
enum, dispatched through the `Task._ACTIONS` ClassVar; adding an action means
adding a method *and* an entry there. Block sizes are constructor kwargs
defaulting to the original values (Y 8, chroma 16, packed 4).

**`image.py`** — `BMPImage` hand-parses/writes 24-bit uncompressed BMP with
`struct`. `CustomizableImage` is the custom `.cim` container: `<II` dimensions,
three `<HHH` `BlockDescription` records (Y, Cb, Cr), then coefficients as
little-endian `int16`. Channel dicts are keyed `"y"`, `"cb"`, `"cr"` and rely on
dict insertion order matching the on-disk order.

**`transforms.py`** — `WalshHadamardTransform` builds an orthonormal Hadamard
matrix and sorts its rows by sign-change count for sequency (Walsh) ordering.
`h @ src @ h`; the matrix is symmetric so `inverse_transform` just calls
`transform`. Memoised via `@cached`.

**`colors.py`** — RGB ↔ YCbCr per-pixel conversion, clamped to 0-255.

**`decorators.py`** — `cached`, an unbounded memo keyed on arguments, falling
back to `repr()` for unhashable ones (which `functools.lru_cache` cannot do).

### Where the compression happens

The transform is lossless and involutive. The loss is in
`CustomizableImage.set_data`, which crops each block to its top-left
`packed_block_size` square — at the default 4 that is 16 of 64 luma
coefficients and 16 of 256 chroma coefficients. `Task._slice` zero-pads the
image up to a block multiple; `Task._merge` crops the padding back off.
`with_coeff_removal` is a second, independent lossy knob applied during matrix
construction, off by default.

## Known quirks — do not "fix" casually

- **`BMPImage` pixel triples are in file order, which for BMP is B, G, R.**
  `Task` treats them as R, G, B. Read and write use the same order so round
  trips are exact, but the YCbCr weights land on swapped channels. Correcting
  this changes every compressed output.
- **The `coeff` comparison in `WalshHadamardTransform._build_matrix` is
  one-sided** (`value - coeff < tol`, not on the magnitude). Preserved verbatim
  from the Python 2 original; in practice any non-`None` coeff above
  `-1/sqrt(2)**n` zeroes every entry that is ever negated.
- **Coefficients are rounded (`np.rint`) on write.** The Python 2 original
  relied on `struct.pack` implicitly truncating floats. Output therefore differs
  from the pre-port `data/recreated.bmp` by at most 2 per channel (mean 0.33).
- **`_KWARGS_MARKER` in `decorators.py` must stay module level.** A per-call
  sentinel would make every cache lookup miss.

## History

Ported from Python 2.7 in v0.1.0; the old `fal/` package and root
`transform.py` were removed. Partially based on
https://github.com/ktisha/python2012/tree/dee4beda8e22f3a66a3e31384d4b72ab66102e88/avereshchagin
