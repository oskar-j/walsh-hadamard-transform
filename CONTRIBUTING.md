# Contributing

## Co-ordinator

Oskar Jarczyk (`oskar.jarczyk@gmail.com`)

## Adding features or fixing bugs

* Fork the repo
* Check out a feature or bug branch
* Add your changes
* Update the README and `CHANGELOG.md` when needed
* Submit a pull request to the upstream repo
* Add a description of your changes
* Ensure tests are passing
* Ensure the branch is mergeable

`master` is protected: pull requests only, and all seven checks (`lint`,
`build`, and `test` on Python 3.10 through 3.14) must be green before the merge
button unlocks.

## Setting up

The project is uv-first, and `uv.lock` is committed, so a checkout reproduces
exactly the environment CI uses. Python 3.10 or newer.

```
uv sync --group dev --all-extras
```

`--group dev` brings in pytest, ruff and mypy. `--all-extras` adds matplotlib
and Pillow, which only `examples/roundtrip.py` needs. Note `dev` is a
[PEP 735](https://peps.python.org/pep-0735/) dependency group and not an extra,
so `pip install .[dev]` does not work; without uv, use
`pip install -e ".[demo]" -r requirements-dev.txt`.

## Testing

Run everything CI runs, before pushing:

```
uv run pytest --cov --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

* Coverage must stay at or above 90%, measured with branch coverage on. CI
  enforces it on every supported Python version, so a pull request that drops
  below cannot be merged. It currently sits at ~98%, so there is headroom.
* Coverage is opt-in locally: a bare `uv run pytest` skips the floor entirely,
  because the threshold is applied by `--cov`. Pass it before you push, or CI
  will fail on something your green local run never checked.
* If you add code that is genuinely untestable, prefer an `exclude_also` entry
  in `[tool.coverage.report]` over lowering the floor.

Tests import the installed package, so an editable install has to exist before
pytest will work — `uv sync` handles that. `tests/conftest.py` provides
`write_bmp` and `write_ppm` helpers plus the `gradient_bmp`, `gradient_ppm`,
`sample_bmp` and `sample_ppm` fixtures.

## House conventions

* **Docstrings on every method**, Google style, with `Args:`, `Returns:` and
  `Raises:` sections wherever they apply — including private helpers. Omit the
  sections that do not apply rather than leaving them empty.
* **Type annotations throughout.** `mypy` runs in strict mode and the package
  ships a `py.typed` marker, so annotations are part of the public contract.
* Ruff handles formatting; do not hand-format around it.

## Things that will trip you up

Three invariants are load-bearing and easy to break by accident:

* **Raster images are RGB, top row first, in memory** — whatever the file
  itself stores. BMP is the awkward one, storing blue-green-red samples in
  bottom-up rows, and `BMPImage` converts in both directions. Honour this in
  the format class, never in `Task`, or cross-format conversion silently swaps
  channels or flips the picture.
* **Never apply `@cached` to a method.** `self` joins the cache key by strong
  reference, so every instance is kept alive forever. Make the function module
  level and key it on the values it actually depends on, as
  `walsh.transforms.hadamard_matrix` does.
* **Coefficient removal thresholds spectral coefficients, not matrix entries.**
  Every entry of an orthonormal Hadamard matrix has the same magnitude, so a
  threshold applied to the matrix can only ever be all-or-nothing.

`CLAUDE.md` documents the architecture and the remaining deliberate quirks in
more detail.

## Adding an image format

Add a submodule under `src/walsh/image/` with a class subclassing
`RasterImage`, implementing `load` and `save`, and add its filename suffix to
`SUFFIXES` in `src/walsh/image/__init__.py`. Convert to and from the in-memory
contract inside that class. Nothing in `Task` or the CLI needs to change.

## Releasing

Maintainers only. The version in `pyproject.toml` is the single source of
truth. Cutting a release means bumping it *and* adding the matching
`## [x.y.z]` section to `CHANGELOG.md` in the same merge — the heading is
load-bearing, and the release workflow fails loudly if it is missing. Merging
to `master` then tags `v<version>`, creates the GitHub Release with those notes
and publishes to PyPI via Trusted Publishing. Merges that do not change the
version are a no-op.

## What is wanted at the moment

* Check the "Issues" section for something to pick up
* Support for more uncompressed formats — PAM and headerless RAW are the
  obvious next ones, and the `image` package is laid out to make them
  self-contained additions. Widening the TIFF profile (PackBits, planar,
  16-bit) is another self-contained piece
* Entropy or run-length coding in the `.cim` container. `--coeff-removal`
  currently makes the spectrum much sparser without shrinking the file, since
  the format writes a fixed count of `int16` regardless of zeros; gzip recovers
  roughly half of it, so the gain is real but unclaimed
* Spread some word on social media about this package
* Stay positive :)
