# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

uv-first; `uv.lock` is committed and CI syncs from it.

```
uv sync --group dev --all-extras         # dev setup; requires Python 3.10+
uv run pytest                            # full suite
uv run pytest --cov --cov-report=term-missing   # with coverage (floor 90%)
uv run pytest tests/codec/test_task.py -k roundtrip   # single test / pattern
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
rather than calling `main(argv)` — it no longer returns an int. The
atomic-write regression test injects an `ENOSPC` into `_write_blocks` via
`monkeypatch`: every header overflow that once served as a post-open failure
is caught up front now, so do not look for a real input that fails mid-write. The three
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
modules import the helpers as `from conftest import ...`, which works at any
depth because `[tool.pytest.ini_options] pythonpath = ["tests"]` says so.

Neither tree is flat (0.4.16). The tests mirror the source:

```
src/walsh/image/            tests/
  __init__.py  registry       conftest.py        helpers, fixtures (root, sample)
  base.py      RasterImage    image/             test_base, test_cim, test_io,
  _io.py       staged writes                     test_streams, test_layout
  cim.py       the container    raster/          test_bmp, test_tiff
  raster/      bmp, tiff        netpbm/          test_ppm, test_pam
  netpbm/      ppm, pam,        arrays/          test_npy, test_pickle
               _samples       codec/             test_task, test_transforms,
  arrays/      npy, pkl,                         test_custom_transform,
               _rules                            test_colors, test_golden
                              cli/               test_cli
                              support/           test_decorators, test_exceptions
                              project/           test_requirements_mirror,
                                                 test_readme_toc
```

Three rules keep that working. Test folders have **no `__init__.py`**, so
every test module's basename must be unique across the tree, and there must
be no second `conftest.py`. A test that needs a repository path takes the
`root` fixture, or `sample` for a file in `data/` (it skips when the sdist
has not shipped one), and **never counts `parent` hops from `__file__`**:
that broke for four files the day the folders appeared. `root` is pytest's
rootdir (`pytestconfig.rootpath`), anchored by the `[tool.pytest.ini_options]`
table in `pyproject.toml`, and it asserts that `pyproject.toml` is there,
because a wrong rootdir would otherwise turn every golden test into a quiet
skip (0.4.19). A path is therefore never a module-level constant in a test:
build it inside the test from the fixture. `test_readme_toc.py` also runs as
a script, where there is no pytest configuration, and takes the README from
the working directory or its argument. And **`walsh.image` is the
import path; where a class lives below it is an implementation detail.** The
flat paths of 0.4.15 and earlier (`walsh.image.bmp` and its siblings) were
dropped in the move rather than aliased: compatibility scaffolding would
have made the source narrate its own history, and every documented import
already went through `walsh.image` or `walsh`. `test_layout.py` pins the
exports, each path checked as the first import of a fresh interpreter, which
is where an import cycle between the family packages would show.

## Architecture

`walsh.task` orchestrates; the other modules are layers under it.

**`task.py`** — `Task` is a fluent builder (`with_action`, `with_input`,
`with_output`, `with_coeff_removal`, then `run()`). Actions are the `Action`
enum, dispatched through the `Task._ACTIONS` ClassVar; adding an action means
adding a method *and* an entry there. Block sizes are constructor kwargs
defaulting to the original values (Y 8, chroma 16, packed 4), and
`Task.__init__` validates them (0.4.11, #21): each edge a positive power of
two no larger than `MAX_BLOCK_SIZE` (`COEFF_MAX // 255` = 128, above which an
all-255 block's DC overflows `int16` and the clip on write silently halves
it), and the packed size between 1 and the smallest edge. Packed need *not*
be a power of two — 3 and 6 round-trip and are tested — and packed equal to
the edge is legal. `CustomizableImage.set_data` mirrors the packed check on
the write side. The CLI builds the `Task` inside `_run`'s guarded region for
this reason: `_run` takes a factory, not a task.

`Task(transform=...)` (0.4.13, #41) takes a `Transform` **instance** or, since
0.4.14, a **name** resolved by `transform_for` — `"walsh"`, `"dct"`, `"haar"`,
case-insensitive, an unknown one a `ValueError` listing them — default
`WalshHadamardTransform()`, used by both directions; anything else is a
`TypeError` at construction. It is for experiments — a DCT or Haar through the
same pipeline — and is **library only on purpose**: the `.cim` does not record
its transform, so a file written with a custom one decodes under the default
without complaint into a degraded picture (pinned by
`test_a_cim_does_not_record_its_transform`). Do not add a CLI option for it
without first giving the container a way to say which transform wrote it,
which is a format change. Task calls `transform_stack` /
`inverse_transform_stack`, never `transform` on a stack, because a subclass
only promises single blocks; it checks the returned shape and names the
class. Coefficient removal is applied by Task after the transform
(`remove_small_coefficients`, shared with `WalshHadamardTransform(coeff=...)`
so the two cannot drift), which is what makes it work for any transform, and
`with_coeff_removal` rejects a negative value when it is set.

Between `load` and `save` everything is numpy, and since 0.4.10 (#27) so is
the raster contract itself: `compress` reads `image.get_array().reshape(-1, 3)`
and `extract` ends in `image.set_array(...)`, with no list of tuples anywhere.
`_slice` returns the `(count, edge, edge)` stack it builds, `Task` hands it
to `transform_stack` whole (the built-in has taken a stack since 0.4.0), and
`_merge` takes the stack back through `np.asarray`, free for an array and a
stack for a list. `_merge` derives its row count from the declared height,
never from `len(blocks)`, and raises `ValueError` on a count that cannot tile
the plane (0.4.7). Do not reintroduce a per-pixel or per-block Python loop, a
`list(stack)`, or an `np.stack` of views here: pixel marshalling was half the
codec's wall time up to 0.4.9.

**`image/`** — a package, one module per format, grouped by family into
`raster/`, `netpbm/` and `arrays/` (see Layout). `base.py` defines
`RasterImage` and **the contract that matters: RGB pixels, top row first,
whatever the file stores**. The store is one `(n, 3)` `uint8` array (0.4.10,
#27); `get_array()` / `set_array()` are the primary accessors and hand out a
`(height, width, 3)` view of it, and `get_raw_data()` / `set_raw_data()` are
converters kept for callers (`get_raw_data` returns a fresh list — the
pre-0.4.10 "mutating it mutates the image" promise is gone on purpose). Two
rules for readers: decode with `np.frombuffer` and hand `set_array` the array;
and never route a list of tuples through `np.asarray` — measured, it is
*slower* than per-pixel Python. A list must go through `itertools.chain` into
`np.fromiter` with an explicit count, as `set_raw_data` does. `set_array`
checks `uint8` rather than casting, and sets the size through
`set_dimensions()`, not the attributes: BMP overrides it to derive its header
size fields, and bypassing it once wrote five zero bytes into a header.
`_check_complete` enforces `width * height` pixels, and every `save()` and
`get_array()` calls it (0.4.5, #23). It belongs there, not in the setters: the
documented `set_dimensions` then `set_raw_data` build is transiently
inconsistent by design. `CustomizableImage` is deliberately not a
`RasterImage`; it holds each channel as one `(count, edge, edge)` stack,
`get_stack(channel)` is the array form, and `get_y_data()` and friends return
views into it. `bmp.py` converts both ways (BMP is blue-green-red and
bottom-up, two reversed views); `ppm.py`, `pam.py`, `tiff.py` and `npy.py`
need no conversion.

`npy.py` (0.4.8) reads and writes NumPy's `.npy` container, the raw pixel
matrix for images that already live in an array. The array carries no colour
metadata, so the profile is declared, not detected: `uint8` only; `(h, w, 3)`
is RGB, `(h, w)` and `(h, w, 1)` are greyscale replicated to three channels,
`(h, w, 4)` is RGBA and accepted only when alpha is 255 throughout, since a
non-opaque fourth channel is either transparency (flattening invents a
background) or CMYK (a colour-space conversion, undefined without a profile,
and indistinguishable from RGBA by shape). Everything else is rejected by
name. The header is parsed with `np.lib.format.read_magic` /
`read_array_header_1_0` and validated **before** the body is read. `np.load`
is never called with pickling enabled; an `object` array, whose body is a
pickle, goes through `pkl.safe_loads` since 0.4.15 and is then judged as a
pickle of pixels would be. The array rules themselves live in `arrays/_rules.py`
(`validate_image_array`, `to_rgb`), shared with the pickle reader, and take a
label so each format's messages name it. Versions 1.0 and 2.0 are accepted; 3.0 exists only for
structured dtypes and is rejected by name. The writer is `np.save` of a
C-ordered `(h, w, 3)` `uint8` through the staged `open_binary_write`, so its
bytes are deterministic and byte-identical to `np.save` of the same array —
`data/earth.npy` was written by numpy from Pillow's array and the writer
reproduces it exactly, which is the foreign-writer check.

`pkl.py` (0.4.15) reads `.pkl` / `.pickle`: a pickled NumPy array, rows of
`(r, g, b)` pixels, a flat list of them, or the `{"width", "height",
"pixels"}` dict around any of those. **It must never call `pickle.load`,
`pickle.loads` or `np.load(allow_pickle=True)`**: those execute the file.
`safe_loads` unpickles through `_RestrictedUnpickler`, whose `find_class`
resolves names from the `_ALLOWED` table and nowhere else, importing nothing:
NumPy's `_reconstruct`, `_frombuffer` and `scalar` under both the `numpy.core`
(NumPy 1) and `numpy._core` (NumPy 2) spellings, `numpy.dtype`, and
`_codecs.encode` for protocols 0-2. Three deliberate details. `numpy.ndarray`
resolves to an inert token, not the class, because pickles only pass it as an
argument and the class would let a file call `ndarray(shape)` and allocate
anything; `_reconstruct` is wrapped to accept only that token with shape
`(0,)`, which is all NumPy writes; and the real functions are taken from what
NumPy itself emits (`np.zeros(1).__reduce__()[0]`), so no private module path
is imported. Widening the allowlist is a security decision: an entry must be
unable to import, open, or allocate from its arguments. `MemoryError` is
re-raised, not converted, for the reason given under `EXPECTED_ERRORS`. A
flat list has no dimensions and nothing guesses them (160,000 pixels are
400x400 or 200x800): the size comes from the dict or from
`RasterImage.declare_size`, which `Task.with_input_size` and the CLI's
`--width` / `--height` feed. `Task` verifies a declared size against every
format after `load`, so it is honoured or verified, never dropped, and
rejects one on `extract`. Lists are flattened with `itertools.chain` into
`np.fromiter`, per the rule above, through `_Reiterable` because the samples
are walked twice (types, then values) and a bare `chain` is one-shot; that
was a real bug during development, caught by mypy's complaint about the type.
Samples must be `int` or `np.integer`, never `bool` or float, and in 0-255,
checked before the cast so nothing wraps. The writer emits rows of tuples of
plain `int` at protocol 4: loadable without NumPy, and byte-stable across
NumPy versions, which a pickled array is not. `data/earth.pkl` is a pickled
array written under NumPy 2 and is in the golden test and the CI `cmp` loop.

`netpbm/_samples.py` holds what PPM and PAM share, since their rasters are identical
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

Adding a format means a new module in the family folder it belongs to (or a
new folder, for a new family), subclassing `RasterImage`, plus an import and an
entry in `SUFFIXES` in `image/__init__.py` and a test module in the matching
`tests/image/` folder. Honour the RGB top-down contract there, not in `Task`. The contract
is what makes the source format irrelevant to the output: `tests/codec/test_task.py`
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
`tests/codec/test_transforms.py` keeps that loop as an oracle and asserts byte
equality. Do not "simplify" to `h / np.sqrt(size)`.

`transform` is mathematically `h @ src @ h` then, if `coeff` is set, zeroing
coefficients below that magnitude. **Since 0.4.12 (#39) it is computed
exactly**, in `_spectrum`: the input is snapped to a multiple of `2 ** -f`
with `f = 36 - 2 * log2(n)`, multiplied on both sides by the `+-1` sign matrix
(`_hadamard_signs`, memoised like `hadamard_matrix` and taking its row order
from it), and scaled by `1 / n` last. Every one of those operations has an
exactly representable result for samples below `2 ** 15` at any edge up to
128, so the output is a function of the input bits alone: the same on every
platform and BLAS library, with or without fused multiply-add, and an
integer wherever the true value is an integer. Do not go back to
multiplying by the orthonormal entries — `+-1/sqrt(n)` is irrational, every
product rounds, and the rounding depends on the BLAS summation order, which
is what made 0.4.11's outputs differ between Accelerate and openblas and,
worse, made the inverse return `1.999999999999999` where the answer was `2`,
which the colour conversion then truncated a level low. The grid constants
`_SAMPLE_BITS` and `_MARGIN_BITS` are part of the codec's output: changing
them moves the handful of coefficients that sit within `1e-9` of a half.
A butterfly (fast Walsh-Hadamard transform) was measured for #39 and rejected
for the runtime: in numpy it is about eight times slower than the matrix
product at the codec's block sizes because every pass materialises
temporaries. It lives on in `tests/codec/test_transforms.py` as the oracle, since
it adds the same samples in a completely different order and can only agree
bit for bit with the matrix products when neither rounds anywhere. `inverse_transform` applies only the matrix — repeating the
threshold would discard reconstructed detail twice, so the two are no longer
the same call once `coeff` is set. Both accept a 3-D stack of blocks as well as
one block, and `transform_sequence` / `inverse_transform_sequence` stack
uniformly shaped blocks into one broadcast call, falling back to one call per
block only for mixed shapes. The `Transform` base class keeps the per-block
defaults for subclasses that know nothing of stacks, and the same goes for
`transform_stack` / `inverse_transform_stack` (0.4.13): the base loops over
blocks through `_per_block`, which compares shapes explicitly because
assignment would broadcast a scalar over a block, and `WalshHadamardTransform`
overrides both with its one broadcast product. That override is what keeps the
built-in path vectorised; `examples/compare_transforms.py` shows a subclass
doing the same, and `tests/codec/test_custom_transform.py` runs that example.
`examples/plot_transform.py` draws `doc/how_it_works.png` for the README
(0.4.17): regenerate it after any change to the transform or the sample, and
keep `pick_block`'s rule honest, since `tests/codec/test_plot_transform.py`
checks the block it chooses against the caption's claim.

`MatrixTransform` (0.4.14) is the public base for a separable orthonormal
transform given by one matrix: `m @ src @ m.T`, inverse by the transpose,
stacks in one broadcast product. `DiscreteCosineTransform` and `HaarTransform`
are built on it from the module-level memoised `dct_matrix` and `haar_matrix`,
whose arrays are read-only because every caller shares them. `TRANSFORMS` is
the name registry; adding a transform means a class and an entry there.
**Only Walsh-Hadamard is bit-exact.** The DCT and Haar matrices are
irrational, so their products round and the rounding follows the BLAS: never
pin their output byte for byte, and never add them to `tests/codec/test_golden.py`
or the CI `cmp` smoke. Their tests assert properties with margins instead
(orthonormality, round trip, the DCT beating Walsh by over 1 dB on a smooth
picture, Haar tying Walsh at kept sizes 2, 4 and 8 and parting at 3 and 6).
The names make the unrecorded-transform trap easier to fall into, which is why
the CLI still has no `--transform`: that needs a header field first.

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
use it without a cycle; `image/__init__.py` re-exports `UnsupportedFileFormatError` for
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

No Python-level object is created per pixel or per block anywhere between
`load` and `save` (0.4.0 for the core, 0.4.10 for the raster contract, #27).
On a 2000×2000 image, `compress` takes about 1.0 s and 490 MB peak and
`extract` about 1.0 s and 710 MB, down from 3.2 s / 740 MB and 2.3 s /
1035 MB; the transform arithmetic is now the largest item in the profile.
What remains of the memory is the float64 working set of the transform (the
YCbCr planes and the coefficient stacks at 8 bytes a sample), which is a
different matter from the contract and would need a dtype decision, not more
vectorisation. `tests/codec/test_golden.py` pins the checked-in outputs byte for byte; run it
first after any change on the pixel path. **Byte-identical means exactly
that since 0.4.12:** on every platform and BLAS library, in both directions,
because the transform's arithmetic is exact (see `transforms.py` above).
0.4.11 had measured a one-step difference in 2 of 60,000 coefficients
between Accelerate and openblas and relaxed the checks to a tolerance; that
machinery is gone and the CI wheel smoke compares with `cmp` on the Linux
runner, which is the standing cross-platform proof. A deliberate change to
the codec regenerates every `recreated.*` file and `transformed_earth.cim`
in the same commit and states the change in the CHANGELOG.
`data/transformed_earth.cim` is un-ignored in `.gitignore` explicitly — it
was silently absent from CI until 0.4.11.

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

## CI

Every job installs with `uv sync --locked` and the release runs tests with
`uv run --locked` (0.4.11, #25): a stale `uv.lock`, including one left behind
by a version bump without `uv lock`, fails with uv's own message rather than
being re-resolved in the runner. The `build` job also unpacks the sdist and
runs its tests from inside, the only check that can catch a `MANIFEST.in`
regression, and smoke-tests the wheel against the reference `.cim` for every
container and the reference decode, byte for byte. A `force_publish` retry of the release downloads the assets on the
existing GitHub Release rather than rebuilding, so PyPI gets the same bytes.
`tests/project/test_requirements_mirror.py` keeps `requirements*.txt` equal to
`pyproject.toml` (#24), and `tests/project/test_readme_toc.py` keeps the README's
table of contents equal to its headings: after renaming or adding a heading,
run `python tests/project/test_readme_toc.py` from the repository root to
regenerate the block between the `<!-- toc -->` markers.

## Coverage gate

`[tool.coverage.report] fail_under = 90` with branch coverage on. At 100%
since 0.4.18 (#26), with no `pragma: no cover` on any reachable line, so a
drop shows exactly which lines lost their test. `--cov` is what applies the
floor — a bare `pytest` does not. CI passes `--cov` on all five Python versions and those `test` jobs
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

`MANIFEST.in` controls the sdist. It exists mainly because setuptools on its
own ships only `tests/test_*.py`: not `tests/conftest.py`, and not the test
modules in subfolders, which would produce a sdist whose tests are mostly
missing and whose remainder errors on absent fixtures. `recursive-include
tests *.py` covers both, and CI's in-sdist test run is what would notice.

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
its header before allocating from it (#22), v0.4.8 added `.npy`, a bare NumPy
array as an image, v0.4.9 fixed the CLI help and the stale docstrings (#28,
#29), v0.4.10 made the raster contract array-backed (#27), three times
faster on large images with byte-identical output, and v0.4.11 validated the
block geometry (#21), fixed the requirements mirrors (#24), made CI verify
the lock, the sdist and the release assets (#25), tracked the reference
`.cim` that had been ignored all along, and stated byte identity across
platforms precisely. v0.4.12 made the transform's arithmetic exact (#39), so
output is byte-identical on every platform and the decode no longer truncates
a level low where the true value is an integer. v0.4.13 added
`Task(transform=...)` (#41) so other block transforms can reuse the pipeline
for experiments, v0.4.14 shipped a DCT-II and a Haar transform selectable
by name, and v0.4.15 added pickled arrays and pixel lists as input, read
through an allowlist so nothing in the file is executed, plus a generated
table of contents in the README. v0.4.16 grouped the format modules and the
tests into folders; the flat module paths such as `walsh.image.bmp` went with
it, and `walsh.image` is the import path. v0.4.17 added the README's
explanatory figure, tagline sentences and a typed badge. v0.4.18 closed #26 with the
regression tests it asked for and took coverage to 100%. v0.4.19 made the
tests take the repository root from pytest's rootdir rather than from
`__file__`.
Partially based on
https://github.com/ktisha/python2012/tree/dee4beda8e22f3a66a3e31384d4b72ab66102e88/avereshchagin
