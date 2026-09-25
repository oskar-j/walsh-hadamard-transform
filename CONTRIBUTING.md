# Contributing

## Co-ordinator

Oskar Jarczyk (`oskar.jarczyk@gmail.com`)

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md), version
2.1. It applies to issues, pull requests and every other project space. Report
unacceptable behaviour to the co-ordinator's address above, which is also the
enforcement contact in that document.

## Reporting a vulnerability

Privately, not in an issue or a pull request: [SECURITY.md](SECURITY.md) gives
the route, what a report needs, and what counts.

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
`pip install -e ".[demo]" -r requirements-dev.txt`, or on pip 25.1 or newer
`pip install -e ".[demo]" --group dev`. The requirements files mirror
`pyproject.toml` and a test fails if they drift.

## Testing

Run everything CI runs, before pushing:

```
uv run pytest --cov --cov-report=term-missing
uv run ruff check .
uv run ruff format --check .
uv run mypy
```

* CI installs with `uv sync --locked`, so a change to `pyproject.toml` must
  be committed together with the `uv.lock` that `uv lock` produces for it, and
  a version bump too. The `build` job also unpacks the sdist and runs its
  tests from inside, so `MANIFEST.in` must keep shipping `tests/conftest.py`.
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
`write_bmp`, `write_png`, `write_ppm`, `write_pam`, `write_tiff` and `write_npy` helpers plus a
`gradient_*` fixture (a synthetic 16x16 picture) and a `sample_*` fixture (the
checked-in 400x400 sample) for each format.

## House conventions

* **Docstrings on every method**, Google style, with `Args:`, `Returns:` and
  `Raises:` sections wherever they apply — including private helpers. Omit the
  sections that do not apply rather than leaving them empty.
* **Type annotations throughout.** `mypy` runs in strict mode and the package
  ships a `py.typed` marker, so annotations are part of the public contract.
* **No `type: ignore`.** There are none in the codebase and pull requests
  should not add one. Every suppression removed so far turned out to be a
  design smell rather than a type checker limitation.
* Ruff handles formatting; do not hand-format around it.

## Things that will trip you up

Four invariants are load-bearing and easy to break by accident:

* **Raster images are RGB, top row first, in memory** — whatever the file
  itself stores. BMP is the awkward one, storing blue-green-red samples in
  bottom-up rows, and `BMPImage` converts in both directions. Honour this in
  the format class, never in `Codec`, or cross-format conversion silently swaps
  channels or flips the picture.
* **Never apply `@cached` to a method.** `self` joins the cache key by strong
  reference, so every instance is kept alive forever. Make the function module
  level and key it on the values it actually depends on, as
  `walsh.transforms.hadamard_matrix` does.
* **Coefficient removal thresholds spectral coefficients, not matrix entries.**
  Every entry of an orthonormal Hadamard matrix has the same magnitude, so a
  threshold applied to the matrix can only ever be all-or-nothing.
* **Nothing writes into `data/`.** Its `recreated.*` files and
  `transformed_earth.cim` are the codec's contract, which `test_golden.py` and
  CI's wheel smoke compare against byte for byte. Send your own outputs
  elsewhere: a reference regenerated from a changed codec makes the test
  compare the codec with its own output, and a `git commit -a` then ships the
  change unannounced. A deliberate change to the codec regenerates them all in
  one commit and says so in the CHANGELOG. CI fails if the test suite leaves
  the checkout changed.

`CLAUDE.md` documents the architecture and the remaining deliberate quirks in
more detail.

## Adding an image format

Add a module to the family folder it belongs to under `src/walsh/image/` —
`raster/` for picture formats with their own header, `netpbm/` for the Netpbm
family, `arrays/` for pixel data that is not a picture format, or a new folder
for a new family — with a class subclassing `RasterImage`, implementing `load`
and `save`. Import it in `src/walsh/image/__init__.py`, add its filename suffix
to `SUFFIXES` there, and put its tests in the matching folder under
`tests/image/`. Test folders have no `__init__.py`, so give the test module a
name no other test module has. Convert to and from the in-memory
contract inside that class: decode with `np.frombuffer` and hand `set_array` a
`(height, width, 3)` `uint8` array, and serialise from `get_array()`. Never
build a list of tuples on the way — routing one through `np.asarray` is slower
than per-pixel Python. Nothing in `Codec` or the CLI needs to change. A sample
file goes in `data/<suffix>/`, which is where the `sample` fixture looks for it
from the file name alone.

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
* Support for more formats — TGA is the obvious next uncompressed one.
  Headerless RAW is largely covered by `.npy` since 0.4.8, which carries
  the shape and dtype a raw file lacks. The `image` package is laid out to
  make them self-contained additions; PAM in 0.4.1 and NPY in 0.4.8 are the
  templates. Widening a profile is another self-contained piece: PackBits,
  planar or 16-bit TIFF, and for PNG (0.5.0) palette and greyscale files,
  whose expansion to RGB is lossless, and Adam7 interlacing
* Entropy or run-length coding in the `.cim` container. `--coeff-removal`
  currently makes the spectrum much sparser without shrinking the file, since
  the format writes a fixed count of `int16` regardless of zeros; gzip recovers
  roughly half of it, so the gain is real but unclaimed
* Spread some word on social media about this package
* Stay positive :)
