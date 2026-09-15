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
rather than calling `main(argv)` — it no longer returns an int. The three
block-size options are `click.IntRange(min=1)` (0.4.7): zero once encoded a
header-only file that decoded to solid green. Under `CliRunner` an uncaught
exception lands in `result.exception` and is *not* printed, so a CLI test for
"no traceback" must assert `exit_code == 1` and `"Error:" in result.output`.
`_reject_writing_over_the_input` runs first in both subcommands: the pipelines
read the whole image before writing, so `walsh compress p.bmp p.bmp` used to
succeed and destroy the original. It compares by `os.path.samefile` when OUTPUT
exists, so aliases count. It belongs in the CLI, not in `Task`, whose
`FileSource` may legitimately be `None` for the stdin/stdout path.

## Layout

`src/` layout, package name `walsh`, built with setuptools. Tests import the
installed package, so an editable install must exist before `pytest` will work.
`tests/conftest.py` exposes `write_bmp`, `write_ppm`, `write_pam`, `write_tiff`
and `write_npy` helpers plus `gradient_*` and `sample_*` fixtures per format; test
modules import the helpers as `from conftest import ...`, which works because
pytest puts `tests/` on `sys.path`.

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
between layers are views into one array. `_merge` derives its row count from
the declared height, never from `len(blocks)`, and raises `ValueError` on a
count that cannot tile the plane (0.4.7). Do not reintroduce a per-pixel or
per-block Python loop here: that was the whole cost of the codec before 0.4.0.

**`image/`** — a package, one submodule per format. `base.py` defines
`RasterImage` and **the contract that matters: RGB pixels, top row first,
whatever the file stores**. `_check_complete` enforces the other half of that
contract, `width * height` pixels, and every `save()` calls it before writing a
byte (0.4.5, #23). It belongs at `save()`, not in the setters: the documented
`set_dimensions` then `set_raw_data` build is transiently inconsistent by
design. `CustomizableImage` is deliberately not a `RasterImage` and keeps no
such invariant. `bmp.py` converts both ways (BMP is blue-green-red
and bottom-up); `ppm.py`, `pam.py`, `tiff.py` and `npy.py` need no conversion.

`npy.py` (0.4.8) reads and writes NumPy's `.npy` container, the raw pixel
matrix for images that already live in an array. The array carries no colour
metadata, so the profile is declared, not detected: `uint8` only; `(h, w, 3)`
is RGB, `(h, w)` and `(h, w, 1)` are greyscale replicated to three channels,
`(h, w, 4)` is RGBA and accepted only when alpha is 255 throughout, since a
non-opaque fourth channel is either transparency (flattening invents a
background) or CMYK (a colour-space conversion, undefined without a profile,
and indistinguishable from RGBA by shape). Everything else is rejected by
name. The header is parsed with `np.lib.format.read_magic` /
`read_array_header_1_0` and validated **before** the body is read; a pickled
`object` array is refused from the header dtype and `np.load` is never called
with pickling enabled. Versions 1.0 and 2.0 are accepted; 3.0 exists only for
structured dtypes and is rejected by name. The writer is `np.save` of a
C-ordered `(h, w, 3)` `uint8` through the staged `open_binary_write`, so its
bytes are deterministic and byte-identical to `np.save` of the same array —
`data/earth.npy` was written by numpy from Pillow's array and the writer
reproduces it exactly, which is the foreign-writer check.

`_netpbm.py` holds what PPM and PAM share, since their rasters are identical
behind different headers: the one-byte sample decoder (one read, a lookup-table
rescale when `maxval` is below 255, tuples built by `zip` in C; samples above
`maxval` are rejected by name) and the encoder (`bytes` over a chained
iterator). `pam.py` supports exactly one profile — `DEPTH 3`, `TUPLTYPE RGB`
or none, `MAXVAL` ≤ 255 — and rejects the rest by name, like `tiff.py`. Its
writer emits the header in `pamtopam`'s order, so the two are byte-identical;
the Netpbm tools (`pamvalidate`, `pamtopnm`) are the independent check when
touching it.

`tiff.py` supports exactly one TIFF profile — uncompressed, RGB, 8-bit, chunky,
top-left — and rejects everything else by name rather than guessing. Its strip
reader clamps every read to the bytes still outstanding (0.4.5, #23): repeated
strip offsets otherwise let a small file cost hundreds of megabytes. Do not
"tidy" that into an up-front `sum(counts) > expected` rejection — `RowsPerStrip`
need not divide the height, so a padded final strip legitimately overshoots and
real files rely on the surplus being ignored. It reads
both byte orders (the header declares its own) and multi-strip files; it always
writes little-endian single-strip. Widening the profile means handling
`Compression`, `PlanarConfiguration` or `BitsPerSample` in `_validate` and the
strip reader, not loosening the checks. `cim.py` holds
`CustomizableImage`, the `.cim` container: `<II` dimensions, three `<HHH`
`BlockDescription` records (Y, Cb, Cr), then coefficients as little-endian
`int16` — its channel dicts are keyed `"y"`, `"cb"`, `"cr"` and rely on dict
insertion order matching the on-disk order. `number_of_blocks` is a `H`, so
`MAX_BLOCKS_PER_CHANNEL` is 65535 and the codec caps out near 4.2 MP at the
default 8-pixel luma block. `Task._check_fits_the_container` rejects an
oversized image up front, naming the channel and the block size that would fit
(0.4.6, #19); `set_descriptions` re-checks as a backstop. Widening the field
would raise the ceiling and break every existing `.cim`, so it is a format
decision rather than a fix. Do not "simplify" by deriving the count from the
dimensions: a zero count is the meaningful "empty channel, fill with a neutral"
state that `extract` relies on. **The reader validates the header before
allocating from it** (0.4.7, #22): `_validate_header` runs once all three
descriptions are read (so a short header is still "truncated", not
"inconsistent") and requires positive dimensions, a power-of-two block size, a
packed size in `1..block`, and a block count of `0` or exactly
`blocks_for(width, height, block)`. That count rule is exact — the encoder pads
to a block multiple — and `blocks_for` is the one implementation both `Task`
and the reader use. `.cim` has no signature, so this is also how a file that
is not a `.cim` at all is caught. Coefficients are read in bounded chunks
(`_read_up_to`) because `file.read(n)` allocates `n` bytes first. A channel
is decoded with one `np.frombuffer` and written with one `tobytes`; a
truncated file still reports the index of the first incomplete block. Do not
add `MemoryError` to `EXPECTED_ERRORS` to paper over an unbounded allocation:
its message is empty and it would relabel a genuine OOM as malformed input. `_io.py` has `align`,
`open_binary_read` and `open_binary_write` — separate rather than one
mode-string function, so `open()` gets a literal mode and the handle type is
known; `open_binary(source, mode)` remains as a delegate — and `read_up_to`,
the chunked read every reader uses for a size it took from a header (moved
there from `cim.py` in 0.4.8 so `npy.py` shares it). `__init__.py`
re-exports everything the pre-0.2.0 single module exported, so old imports
still work, and owns `reader_for`, which dispatches on filename suffix.

**`open_binary_write` stages every write** (0.4.2, #20): bytes go to a
`.walsh-*` temporary file in the destination's own directory and reach the
destination only through `os.replace`, once the caller has returned without
raising. That is what keeps a mid-write failure — a container field that
overflows, ENOSPC, EIO — from replacing a good file with a stub. The staging
file must stay a sibling of the destination: `os.replace` is atomic only
within one filesystem. Three paths exist to preserve prior behaviour and each
has a test: a symlink destination is `realpath`-resolved so the link survives,
a non-regular destination (`/dev/null`, a FIFO) is opened directly because it
cannot be replaced, and the mode is copied from the destination or derived
from the umask, since `mkstemp` creates `0o600`.

Adding a format means a new submodule subclassing `RasterImage` plus an entry in
`SUFFIXES`. Honour the RGB top-down contract there, not in `Task`. The contract
is what makes the source format irrelevant to the output: `tests/test_task.py`
asserts that BMP, PPM, PAM and TIFF of one picture compress to byte-identical
`.cim`.

**`transforms.py`** — `hadamard_matrix(size)` is a **module-level** `@cached`
function building the orthonormal matrix and sorting rows by sign-change count
for sequency (Walsh) ordering. It must stay module level: memoising the old
method pinned every `WalshHadamardTransform` instance forever (fixed in 0.2.1).
`WalshHadamardTransform._build_matrix` is now a thin delegate kept for callers.

`hadamard_matrix` is Sylvester's construction — `log2(size)` Kronecker
products with `[[1, 1], [1, -1]]`, then the sequency sort — and it rejects a
`size` that is not a power of two. The scale is applied by multiplication, not
division, so every entry is bit-identical to the pre-0.4.0 triple loop;
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
coefficients and 16 of 256 chroma coefficients. `Task._slice` pads the image
up to a block multiple **by replicating its last row and column**
(`mode="edge"`), and `Task._merge` crops the padding back off. The padding
shares its blocks with real pixels and the transform is low-pass, so whatever
fills it is smeared back over the last few real columns and rows: zero-filling
put a black-and-saturated seam there, off by up to 200 of 255 on flat colour
(fixed in 0.4.3, #18). Do not change this back, and note that `reflect` is not
a drop-in alternative — it raises when the pad is wider than the image, which
happens for anything under 16 pixels.
`with_coeff_removal` is a second, independent lossy knob, off by default. It
thresholds *spectral coefficients*, not matrix entries — every entry of an
orthonormal Hadamard matrix has the same magnitude, so a threshold on the
matrix can only ever be all-or-nothing. That was the 0.2.1 bug.

### Performance

No Python-level loop touches a pixel or a block between `load` and `save`
(0.4.0). On the 400×400 sample the codec core takes about 40 ms; the rest of
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
and flips the image. v0.4.0 vectorised the codec core with byte-identical
output. v0.4.1 added PAM, v0.4.2 made writes atomic, and v0.4.3 fixed the
zero-padded edge seam (#18), which changes encoder output for images that need
padding while leaving every existing `.cim` decoding unchanged. v0.4.4 added a
code of conduct, v0.4.5 closed three unchecked preconditions in the raster
layer (#23), v0.4.6 made the container's 4.2 MP block-count ceiling an
up-front, actionable error (#19), v0.4.7 made the `.cim` reader validate
its header before allocating from it (#22), and v0.4.8 added `.npy`, a bare
NumPy array as an image.
Partially based on
https://github.com/ktisha/python2012/tree/dee4beda8e22f3a66a3e31384d4b72ab66102e88/avereshchagin
