# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

uv-first; `uv.lock` is committed and CI syncs from it.

```
uv sync --group dev --all-extras         # dev setup; requires Python 3.10+
uv run pytest                            # full suite
uv run pytest --cov --cov-report=term-missing   # with coverage (floor 90%)
uv run pytest tests/test_task.py -k roundtrip   # single test / pattern
uv run ruff check . && uv run ruff format .
uv run mypy                              # strict; config selects the walsh package
uv build                                 # sdist + wheel
uv version --short                       # what release.yml reads
```

`dev` is a PEP 735 dependency group, not an extra — `pip install .[dev]` does
not work. `demo` is a real extra (matplotlib + Pillow, for
`examples/roundtrip.py`), so `--all-extras` is needed to run the example.

Run the codec end to end:

```
walsh compress data/image.bmp out.cim
walsh extract out.cim back.bmp
python examples/roundtrip.py    # same thing plus matplotlib plots
```

The local `.venv` is Python 3.12. `walsh` is installed as a console script
(`walsh.cli:main`).

`cli.py` uses **click**, not argparse. `main` is a `click.group` with `compress`
and `extract` subcommands, so tests drive it through `click.testing.CliRunner`
rather than calling `main(argv)` — it no longer returns an int.

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

Between `load` and `save` everything is numpy. `_pixels_to_array` and
`_array_to_pixels` are the one crossing from the raster contract's list of
tuples into an `(n, 3)` array and back (`np.fromiter` over a chained list and
`zip` over `tolist()` columns, each about twice as fast as the obvious call).
`_slice` and `_merge` are a single reshape each, and the block lists that flow
between layers are views into one array. Do not reintroduce a per-pixel or
per-block Python loop here: that was the whole cost of the codec before 0.3.3.

**`image/`** — a package, one submodule per format. `base.py` defines
`RasterImage` and **the contract that matters: RGB pixels, top row first,
whatever the file stores**. `bmp.py` converts both ways (BMP is blue-green-red
and bottom-up); `ppm.py` and `tiff.py` need no conversion.

`tiff.py` supports exactly one TIFF profile — uncompressed, RGB, 8-bit, chunky,
top-left — and rejects everything else by name rather than guessing. It reads
both byte orders (the header declares its own) and multi-strip files; it always
writes little-endian single-strip. Widening the profile means handling
`Compression`, `PlanarConfiguration` or `BitsPerSample` in `_validate` and the
strip reader, not loosening the checks. `cim.py` holds
`CustomizableImage`, the `.cim` container: `<II` dimensions, three `<HHH`
`BlockDescription` records (Y, Cb, Cr), then coefficients as little-endian
`int16` — its channel dicts are keyed `"y"`, `"cb"`, `"cr"` and rely on dict
insertion order matching the on-disk order. A channel is read with one `read`
and one `np.frombuffer` and written with one `tobytes`; a truncated file still
reports the index of the first incomplete block. `_io.py` has `align`,
`open_binary_read` and `open_binary_write` — separate rather than one
mode-string function, so `open()` gets a literal mode and the handle type is
known; `open_binary(source, mode)` remains as a delegate. `__init__.py` re-exports everything the pre-0.2.0 single module
exported, so old imports still work, and owns `reader_for`, which dispatches on
filename suffix.

Adding a format means a new submodule subclassing `RasterImage` plus an entry in
`SUFFIXES`. Honour the RGB top-down contract there, not in `Task`. The contract
is what makes the source format irrelevant to the output: `tests/test_task.py`
asserts that BMP, PPM and TIFF of one picture compress to byte-identical `.cim`.

**`transforms.py`** — `hadamard_matrix(size)` is a **module-level** `@cached`
function building the orthonormal matrix and sorting rows by sign-change count
for sequency (Walsh) ordering. It must stay module level: memoising the old
method pinned every `WalshHadamardTransform` instance forever (fixed in 0.2.1).
`WalshHadamardTransform._build_matrix` is now a thin delegate kept for callers.

`hadamard_matrix` is Sylvester's construction — `log2(size)` Kronecker
products with `[[1, 1], [1, -1]]`, then the sequency sort — and it rejects a
`size` that is not a power of two. The scale is applied by multiplication, not
division, so every entry is bit-identical to the pre-0.3.3 triple loop;
`tests/test_transforms.py` keeps that loop as an oracle and asserts byte
equality. Do not "simplify" to `h / np.sqrt(size)`.

`transform` is `h @ src @ h` then, if `coeff` is set, zeroing coefficients below
that magnitude. `inverse_transform` applies only the matrix — repeating the
threshold would discard reconstructed detail twice, so the two are no longer
the same call once `coeff` is set. Both accept a 3-D stack of blocks as well as
one block, and `transform_sequence` / `inverse_transform_sequence` stack
uniformly shaped blocks into one broadcast call, falling back to one call per
block only for mixed shapes. The `Transform` base class keeps the per-block
defaults for subclasses that know nothing of stacks.

**`colors.py`** — RGB ↔ YCbCr conversion. `rgb_to_ycbcr` and `ycbcr_to_rgb`
are the implementation and take whole `(n, 3)` arrays; the `ColorModel` classes
are the per-pixel interface and delegate to them one pixel at a time, so the
two cannot drift apart. The inverse truncates (`int()` semantics, via
`np.trunc`) *then* clamps to 0-255. Keep the arithmetic in the same order as
written: the same IEEE operations round the same way, which is what keeps the
checked-in sample outputs byte-identical.

**`decorators.py`** — `cached` returns a `Memo`, an unbounded memo keyed on
arguments, falling back to `repr()` for unhashable ones (which
`functools.lru_cache` cannot do). It is a class rather than a closure so
`cache_clear`/`cache_size` are typed members rather than bolted-on attributes.
**Never apply it to a method**: `self` would join the key by strong reference.
`Memo` is not a descriptor, so that misuse now raises `TypeError` at the call.

**`exceptions.py`** — `WalshError` (base), `UnsupportedFileFormatError`, and
`EXPECTED_ERRORS`, the tuple the CLI converts into a clean `ClickException`.
It must keep importing nothing from the rest of the package, so every module can
use it without a cycle; `image.py` re-exports `UnsupportedFileFormatError` for
back-compat. `EXPECTED_ERRORS` lists `struct.error` explicitly because that
derives from `Exception`, not `ValueError` — dropping it reintroduces a
traceback on truncated input.

### Where the compression happens

The transform is lossless and involutive. The loss is in
`CustomizableImage.set_data`, which crops each block to its top-left
`packed_block_size` square — at the default 4 that is 16 of 64 luma
coefficients and 16 of 256 chroma coefficients. `Task._slice` zero-pads the
image up to a block multiple; `Task._merge` crops the padding back off.
`with_coeff_removal` is a second, independent lossy knob, off by default. It
thresholds *spectral coefficients*, not matrix entries — every entry of an
orthonormal Hadamard matrix has the same magnitude, so a threshold on the
matrix can only ever be all-or-nothing. That was the 0.2.1 bug.

### Performance

No Python-level loop touches a pixel or a block between `load` and `save`
(0.3.3). On the 400×400 sample the codec core takes about 40 ms; the rest of
the wall time is the format readers and writers, which build and consume the
raster contract's list of pixel tuples, plus the two conversions across that
boundary. A numpy-backed `RasterImage` is the next step and a public-API
change, since `get_raw_data` returns that list.

Multiprocessing was considered for the matrix build and rejected: the build
takes about 0.1 ms at the codec's block sizes and runs once per process
(memoised), while spawning a pool costs ~170 ms, and the memo is per process
so each worker would rebuild it. Threads do not help either: the matrix
products are 8×8 and 16×16, far too small to amortise a handoff even though
numpy releases the GIL inside them.

## Known quirks — do not "fix" casually

- **Coefficients are rounded (`np.rint`) on write.** The Python 2 original
  relied on `struct.pack` implicitly truncating floats. Output therefore differs
  from the pre-port `data/recreated.bmp` by at most 2 per channel (mean 0.33).
- **`_KWARGS_MARKER` in `decorators.py` must stay module level.** A per-call
  sentinel would make every cache lookup miss.

## Coverage gate

`[tool.coverage.report] fail_under = 90` with branch coverage on. Currently 97%,
so there is headroom, but `--cov` is what applies the floor — a bare `pytest`
does not. CI passes `--cov` on all five Python versions and those `test` jobs
are required checks on `master`, so a coverage drop blocks the merge rather than
just going red. `release.yml` applies the same floor before it builds.

If you add a module that is genuinely untestable, prefer an `exclude_also`
entry in `[tool.coverage.report]` over lowering the floor.

## Release pipeline

`pyproject.toml`'s version is the single source of truth. `.github/workflows/release.yml`
fires on every push to `master`, reads `uv version --short`, and does nothing if
a `v<version>` GitHub Release already exists. Otherwise it extracts the matching
`## [x.y.z]` section from `CHANGELOG.md` (failing loudly if absent), runs the
tests, builds, creates the Release, and publishes to PyPI via Trusted Publishing
in the `pypi` GitHub environment — no API token in the repo.

So: cutting a release means bumping the version *and* adding its CHANGELOG
section in the same merge. The `## [x.y.z]` headings are load-bearing.

`MANIFEST.in` controls the sdist. It exists mainly because setuptools ships
`tests/test_*.py` but not `tests/conftest.py`, which would produce a sdist whose
tests all error on missing fixtures.

## Typing

`mypy --strict` over the `walsh` package, and **the codebase carries no
`type: ignore`**. If one seems necessary, the annotation or the design is
usually wrong: both historical suppressions came from a function attribute that
wanted to be a class member, and an `open()` call that wanted a literal mode.

## Docstring style

Google sections (`Args:` / `Returns:` / `Raises:`) on every method in
`src/walsh`, including private ones, matching the `thresher` repo. Include the
sections that apply and omit the ones that do not — no empty `Args:` on a
no-argument method.

## History

Ported from Python 2.7 in v0.1.0; the old `fal/` package and root
`transform.py` were removed. v0.1.1 added uv support and the release pipeline.
v0.1.2 added the coverage gate, v0.1.3 moved the CLI to click, and **v0.2.0
added PPM, split `image.py` into a package, and fixed the BGR/bottom-up quirk**
by making RGB top-down the shared in-memory contract. That last change makes
0.1.x `.cim` files incompatible: extracting one with 0.2.0 swaps red and blue
and flips the image. v0.3.3 vectorised the codec core with byte-identical
output.
Partially based on
https://github.com/ktisha/python2012/tree/dee4beda8e22f3a66a3e31384d4b72ab66102e88/avereshchagin
