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
rather than calling `main(argv)` — it no longer returns an int. `_EXPECTED_ERRORS`
is the tuple of exceptions turned into a clean `click.ClickException`; it has to
list `struct.error` explicitly, since that derives from `Exception` rather than
`ValueError`.

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

## History

Ported from Python 2.7 in v0.1.0; the old `fal/` package and root
`transform.py` were removed. v0.1.1 added uv support and the release pipeline.
Partially based on
https://github.com/ktisha/python2012/tree/dee4beda8e22f3a66a3e31384d4b72ab66102e88/avereshchagin
