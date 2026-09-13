# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The `## [x.y.z]` headings are load-bearing: the release workflow extracts the
section matching the version in `pyproject.toml` and uses it as the GitHub
Release notes.

## [0.4.4]

### Added

- `CODE_OF_CONDUCT.md`: the [Contributor Covenant](https://www.contributor-covenant.org)
  version 2.1, verbatim from the upstream source with only the reporting
  placeholder filled in. GitHub recognises the file at the repository root and
  surfaces it in the Community Standards checklist and on the issue and pull
  request forms.

  The enforcement contact is the co-ordinator address `CONTRIBUTING.md` already
  publishes, rather than a second inbox nobody watches. `CONTRIBUTING.md` and
  the README now link to the document, and `MANIFEST.in` ships it in the sdist
  alongside `CONTRIBUTING.md` and `CHANGELOG.md`.

  The Contributor Covenant is licensed CC BY 4.0; the upstream Attribution
  section is kept intact, which is what that licence asks for. It covers the
  document only and does not affect this project's MIT licence.

### Notes

- No code changed, so the codec is untouched: all five checked-in sample
  outputs still reproduce byte for byte.

## [0.4.3]

### Fixed

- **Images whose dimensions are not a multiple of the block size no longer come
  back with a corrupted edge.** Closes #18.

  `Task._slice` padded each plane up to a whole number of blocks with zeros.
  The padding shares its blocks with real pixels and the transform is low-pass,
  so whatever fills the padding is smeared back over the last few real columns
  and rows. Zero is black in luma and fully saturated in chroma, so it showed
  up as a coloured seam. Solid orange 17 pixels wide came back with a pure
  green final column, off by 200 of 255. A 1x1 image, smaller than every block
  and therefore entirely padding, was destroyed outright.

  The padding now replicates the last row and column (`mode="edge"`), which is
  what JPEG does for the same reason: the fill carries no content of its own,
  so it cannot introduce an edge that was not in the picture.

  Measured on a flat colour at every width from 16 to 24, the worst channel
  error falls from 200 to 1, which is what an exactly aligned image already
  scored. On the Blue Marble sample cropped off-alignment:

  | size | PSNR before | PSNR after |
  | --- | --- | --- |
  | 400x400 (aligned) | 25.07 dB | 25.07 dB |
  | 399x400 | 25.00 dB | 25.06 dB |
  | 398x398 | 24.72 dB | 25.03 dB |
  | 397x395 | 24.76 dB | 24.99 dB |

  The mean understates it, because the damage is confined to the seam: at
  398x398 the last column's mean absolute error drops from 24.08 to 0.68 and
  its worst pixel from 101 to 5, against an interior mean of 7.9.

  This affected roughly 15 of every 16 arbitrary sizes, on the default path,
  with no unusual options. It survived to 0.4.2 because nothing could see it:
  the only test for non-multiple dimensions used 20x20, and a reconstruction is
  constant on 4-wide tiles, so a pad boundary landing on a multiple of 4 is
  exactly representable whatever the padding holds. `data/earth.ppm` is 400x400
  and equally clean. That test now uses 18x18 and asserts on pixels rather than
  only on the dimensions, and a flat colour is round-tripped at 17x16, 18x18,
  16x19, 23x21 and 1x1.

### Notes

- **This changes `.cim` bytes for images that need padding, and only those.**
  The encoder is what changed: `_slice` is called only from `compress`, so
  **every existing `.cim` file still decodes to exactly the same pixels**,
  verified by decoding a 0.4.2-written file under both versions and comparing
  byte for byte. All five checked-in sample outputs also reproduce byte for
  byte, since they derive from a 400x400 source that needs no padding at all.
  Images that are a multiple of 16 on both axes are unaffected; anything else
  re-encodes differently, and better.
- 254 to 259 tests. The five new ones and the two tightened ones all fail
  against the old padding, checked.

## [0.4.2]

### Fixed

- **A failed write no longer destroys the file it was writing over.** Closes #20.

  `open_binary_write` truncated the destination the moment it opened it, so any
  failure after that point -- an image too large for a container field, a full
  disk, an I/O error -- replaced the previous contents with a stub while the CLI
  printed a clean `Error:` line that read like a safe abort. Compressing a
  4.19 MP or larger image over an existing `.cim` left an 8-byte file behind: a
  well-formed `<II` header, so a later extract reported a *format* error,
  indistinguishable from a file that was never valid.

  Writes are now staged. Bytes go to a temporary file in the destination's own
  directory and are moved onto it with `os.replace` only once the caller has
  finished without raising; a failure unlinks the staging file and leaves the
  destination exactly as it was. `os.replace` is atomic within a filesystem,
  which is why the staging file is a sibling rather than a file in the system
  temporary directory.

  Three behaviours are preserved deliberately, each with a test: a symlinked
  destination is resolved first, so the link survives and its target is
  replaced, as writing through it did before; a destination that is not a
  regular file, such as `/dev/null` or a FIFO, cannot be replaced and is
  written directly, as before; and the finished file keeps an existing
  destination's permissions, or gets `0o666` less the umask when it is new,
  rather than the `0o600` `mkstemp` creates. One behaviour does change: the
  destination gets a new inode, so a pre-existing hard link to it keeps the old
  contents. That is inherent to replace-based atomic writes.

- **`walsh compress photo.bmp photo.bmp` is refused instead of silently
  destroying the photo.** Both pipelines read the whole image before writing
  anything, so naming one file twice did not fail: it exited 0 having replaced
  the original with its own lossy reconstruction, mean absolute error 12.6 on
  the sample image. The CLI now rejects it with exit code 2 before reading a
  byte. An existing destination is compared with `os.path.samefile`, so a hard
  link or a symlink pointing back at the input is caught too; a path that does
  not exist yet is compared by resolved name. Overwriting a *different*
  pre-existing output is still allowed -- re-running into the same target is
  ordinary use.

### Changed

- The stream-helper tests moved from `tests/test_decorators.py`, where they had
  been since 0.3.2 for want of a better home, into a new `tests/test_io.py`
  alongside the staging tests.

### Notes

- **Codec output is unchanged.** Only where the bytes land first changed, not
  what they are: all five checked-in sample outputs still reproduce byte for
  byte, and PPM, PAM and TIFF of one picture still compress to an identical
  `.cim`.
- 237 to 254 tests; `cli.py` and `image/_io.py` are both at 100%, total
  coverage 98.4%.
- The `<HHH` block-count ceiling that triggers the compress failure above is
  still there, and is still ~4.19 MP at the default block size. Staging makes
  that failure non-destructive; it does not raise the ceiling. That is #19.

## [0.4.1]

### Added

- **PAM support**, read and written, under `.pam`. PAM (`P7`) is Netpbm's
  general container: the same raw raster as PPM behind a header of
  `KEY value` lines closed by `ENDHDR`, which also serves bitmaps, greyscale
  and alpha. Exactly one profile is supported, `DEPTH 3` with `TUPLTYPE RGB`
  (or no `TUPLTYPE`, which by convention means the same) and `MAXVAL` at most
  255; greyscale, alpha, other tuple types and 16-bit samples are rejected by
  name, as the TIFF reader does. A `maxval` below 255 is rescaled on load.
  Header keys may come in any order, comment and blank lines are skipped, and
  CRLF line endings are accepted.

  The writer emits the header in the order Netpbm's own `pamtopam` uses, so
  the two produce byte-identical files. Checked against Netpbm 11.2: the
  reader loads `pamtopam`'s output and a `pamdepth 15` file correctly and
  rejects a `ppmtopgm` greyscale file by name; the writer's output passes
  `pamvalidate`, and `pamtopnm` turns it back into the source PPM exactly.
- `data/earth.pam` and `data/recreated.pam`: the Blue Marble sample in a third
  container, and the codec's output for it. Compressing it gives the same
  `.cim` as the PPM and the TIFF, byte for byte, and the suite checks that.
- `PAMImage` is exported from `walsh` and `walsh.image`, with `PAM_MAGIC`,
  `PAM_DEPTH` and `PAM_TUPLTYPE`. `tests/test_pam.py` covers the round trip,
  the header layout, field order, comments and whitespace, CRLF, optional and
  multi-line `TUPLTYPE`, rescaling, every rejected profile by name, and eleven
  malformed headers. 203 to 237 tests; `pam.py` and the shared decoder are at
  100%.

### Changed

- PPM and PAM share one raster decoder and encoder in `image/_netpbm.py`,
  since the two formats differ only in their headers. The decoder is
  vectorised: one read, a lookup-table rescale when `maxval` is below 255,
  and tuples built in C. Loading the 400x400 sample PPM drops from about
  46 ms to 15 ms. The encoder consumes the pixel list through `bytes()`
  directly, about twice as fast as the generator it replaces.
- Netpbm samples larger than the declared `maxval`, which the formats forbid,
  are now rejected with a named error in both the binary and ASCII PPM paths
  rather than passed through as out-of-range values.
- The CLI help lists every supported suffix. It had still said "BMP or PPM"
  since before TIFF arrived.

## [0.4.0]

### Changed

- **The codec core is vectorised end to end.** Every per-element Python loop
  between loading a raster image and saving one is now a whole-array numpy
  operation. On the 400x400 sample, `compress` drops from about 246 ms to
  86 ms and `extract` from about 185 ms to 53 ms; the codec itself, meaning
  everything except the format reader and writer, runs roughly five times
  faster (200 ms to 40 ms, and 166 ms to 34 ms). **Output is unchanged**: all
  four checked-in sample outputs reproduce byte for byte, and new tests pin
  the matrix, the batched transform and the colour conversion to their
  previous scalar forms bit for bit.

  - `hadamard_matrix` uses Sylvester's construction, `log2(size)` Kronecker
    products with `[[1, 1], [1, -1]]`, instead of a triple loop over bits,
    rows and columns. At the codec's own block sizes both take about 0.1 ms,
    once per process, since the result is memoised; the loop was never where
    the time went. But it was `size ** 2 * log2(size)` Python iterations and
    now scales: 82 ms to 0.4 ms at size 256. The scale is applied by
    multiplication so entries are bit-identical to the loop's. It also
    rejects a `size` that is not a power of two with `ValueError`, which the
    construction requires, instead of returning a wrong-shaped matrix.
  - `WalshHadamardTransform.transform` and `inverse_transform` accept a 3-D
    stack of blocks, and `transform_sequence` and `inverse_transform_sequence`
    use that to send every block of a channel through one broadcast matrix
    product. Mixed block shapes, which the codec never produces, fall back to
    one call per block. The `Transform` base class is unchanged.
  - `colors.py` gains `rgb_to_ycbcr` and `ycbcr_to_rgb`, which convert an
    `(n, 3)` array in a handful of operations. The per-pixel `ColorModel`
    classes remain and now delegate to them one pixel at a time, so the two
    cannot disagree.
  - `Task._slice` and `_merge` cut and reassemble blocks with a single
    reshape each rather than a split or stack per row and per block, and the
    pipelines keep planes as arrays rather than lists between stages.
  - `.cim` block I/O reads and decodes a whole channel with one `read` and
    one `np.frombuffer`, and writes it with one `tobytes`, instead of one
    `struct.unpack` or `write` per block. A truncated file still names the
    first incomplete block.

- **Why not multiprocessing.** The question that prompted this release was
  whether the matrix loop could be parallelised. It could, but spawning a
  worker pool costs about 170 ms on macOS, three orders of magnitude more
  than the 0.1 ms build it would parallelise, and the memo is per process so
  every worker would rebuild it anyway. Threads would not help either: numpy
  releases the GIL inside a matrix product, but the codec's products are 8x8
  and 16x16, far too small to amortise a handoff. The cost was interpreter
  overhead per element, and vectorising removes it with no new dependency, no
  pickling, and no start-method differences between platforms.

- What remains of the wall time is the raster contract itself: the format
  readers build a Python list of pixel tuples and the writers consume one, and
  crossing that boundary into numpy and back is now about half of the codec
  core. A numpy-backed `RasterImage` is the next step and a larger one, since
  `get_raw_data` is public.

### Added

- Tests pinning `hadamard_matrix` to the pre-0.4.0 loop byte for byte at
  every size from 1 to 64, batched transforms to per-block results exactly,
  and the array colour conversion to the scalar formula exactly; plus the
  power-of-two check, mixed and empty block sequences, the slice and merge
  round trip on a padded plane, the pixel list and array conversion, multi-block
  `.cim` channels, truncation reporting on a later block, and extracting a
  file whose channels hold no blocks. 170 to 203 tests.

## [0.3.2]

### Changed

- **No `type: ignore` remains anywhere in the codebase.** All three were fixed
  at the root rather than moved or broadened.

  - `decorators.py` suppressed `attr-defined` twice, because `cache_clear` and
    `cache_size` were attributes bolted onto a closure. The memo is now a
    `Memo` class, so both are real typed methods. `cached` returns
    `Memo[P, R]`, which keeps the wrapped signature: mypy now rejects a wrong
    argument type, a wrong use of `cache_size()`'s result, *and* a call to an
    attribute that does not exist -- the last being exactly what the
    suppression was hiding.
  - `image/_io.py` suppressed `misc`, because `open()` given a `str` mode
    returns `IO[Any]`. Split into `open_binary_read` and `open_binary_write`,
    each opening with a literal mode, so the handle's type is known. This also
    removes the `"r" in mode` string sniffing that decided between stdin and
    stdout. `open_binary(source, mode)` is kept as a thin delegate, so no
    caller breaks.

- `Memo` is not a descriptor, so applying `@cached` to a method now raises
  `TypeError` at the call rather than silently pinning every instance -- the
  0.2.1 leak becomes structurally hard to reintroduce.

### Added

- `tests/test_decorators.py`, covering memoisation, the unhashable-argument
  fallback, keyword handling and ordering, metadata preservation, the
  not-a-descriptor guarantee, and both stream helpers including their stdin and
  stdout paths. `decorators.py` and `image/_io.py` are now at 100%; total
  coverage rises from 96.9% to 97.7%.

## [0.3.1]

### Added

- `data/earth.tiff` and `data/recreated.tiff`: the Blue Marble sample as an
  uncompressed TIFF, and the codec's output for it. `earth.tiff` is the same
  picture as `earth.ppm` in a different container, so the pair doubles as a
  check that the source format does not affect the result -- compressing either
  produces a byte-identical `.cim`, and the two reconstructions are
  pixel-identical.
- A `sample_tiff` fixture and tests that exercise the checked-in samples.
- An explanation of PSNR in the README, covering the formula, why the
  logarithmic scale makes small-looking differences significant, what ranges
  mean in practice, and the caveat that it measures arithmetic difference
  rather than perceived quality.

### Changed

- `data/recreated.bmp` regenerated with the current codec. It had been produced
  before the 0.2.0 pixel-order fix, so `data/README.md` carried a note telling
  readers not to treat it as current output. Regenerating it retires the note:
  70% of channels changed, by at most 11 and 1.13 on average. PSNR against the
  source is unchanged at 23.09 dB, which is consistent with the 0.2.0 finding
  that the channel and row fix corrected the pipeline without measurably
  changing reconstruction quality.
- `data/README.md` rewritten around a table of every sample and its result,
  with the provenance and licence of the Blue Marble source kept in full.

## [0.3.0]

### Added

- **Uncompressed TIFF support**, read and written, under `.tif` and `.tiff`.

  TIFF is a container rather than a single layout, so the reader handles one
  profile and names what it will not take rather than guessing: uncompressed
  (`Compression = 1`), RGB (`PhotometricInterpretation = 2`), 8 bits per
  sample, 3 samples per pixel, chunky, top-left orientation. LZW, PackBits,
  palette, CMYK, greyscale, 16-bit, planar and rotated files are each rejected
  with a message naming the actual value.

  Both byte orders are read, since a TIFF header declares its own endianness.
  Multi-strip files are read. Output is always little-endian and single-strip,
  in the minimal 140-byte-header layout -- byte-for-byte the same size as
  Pillow produces for the same picture.

  Verified against Pillow in both directions: this reads TIFFs Pillow writes
  (single and multi-strip) pixel-for-pixel, and Pillow reads what this writes.

- A `gradient_tiff` fixture and `write_tiff`/`build_tiff` test helpers. The
  builder can emit the unsupported profiles on demand, which Pillow will not
  do, so the rejection paths are covered by real files rather than mocks.

### Notes

TIFF joins BMP and PPM under the same in-memory contract, so the source format
cannot change the result: compressing one picture from `.bmp`, `.ppm` and
`.tif` produces byte-identical `.cim` output, and a `.cim` can be extracted to
any of the three. Both are asserted in the test suite.

## [0.2.2]

### Added

- `CONTRIBUTING.md`, covering the uv-first setup, the checks CI runs, the
  coverage floor and how it is enforced, the docstring and typing conventions,
  how to add an image format, and the release process.

  It also writes down the three invariants that are easy to break by accident:
  raster images are RGB top-row-first in memory whatever the file stores,
  `@cached` must never be applied to a method, and coefficient removal
  thresholds spectral coefficients rather than matrix entries. Each of those
  was a real bug at some point in 0.1.x-0.2.1.

## [0.2.1]

### Fixed

- **`--coeff-removal` is now a usable quality dial.** It applied its threshold
  to the *Hadamard matrix*, where every entry has the same magnitude
  (`1/sqrt(2)**n`), so any threshold was necessarily all-or-nothing: below
  `-1/sqrt(2)**n` it did nothing, at or above it zeroed every negated entry.
  On `data/earth.ppm`, `--coeff-removal 0.5` took PSNR from 25.07 dB to
  7.35 dB, and no setting did anything useful in between.

  The threshold now applies to the **spectral coefficients**, which is what
  "coefficient removal" means, and degrades smoothly:

  | `--coeff-removal` | non-zero coefficients | gzipped `.cim` | PSNR |
  | --- | --- | --- | --- |
  | *(unset)* | 48,121 / 60,000 | 59,404 B | 25.07 dB |
  | 5 | 29,049 | 45,080 B | 25.06 dB |
  | 25 | 14,769 | 27,924 B | 24.71 dB |
  | 50 | 8,917 | 19,285 B | 23.85 dB |

  The `.cim` file itself does not shrink, because the format stores a fixed
  count of `int16` values whether or not they are zero. The sparsity is real
  though: gzipped, a threshold of 25 is 2.1x smaller for 0.36 dB.

  A negative threshold is now rejected rather than silently keeping
  everything, and `inverse_transform` no longer reapplies the threshold, which
  would have discarded reconstructed detail a second time. **Output with no
  `--coeff-removal` is byte-for-byte unchanged from 0.2.0.**

- **A malformed `.cim` raised `struct.error`.** That derives from `Exception`
  rather than `WalshError`, so a library caller catching this package's own
  exceptions missed it. `BMPImage` gained truncation checks in 0.2.0 and the
  `.cim` reader did not. It now raises `UnsupportedFileFormatError` for a
  truncated header, a truncated block description, missing block data, and a
  description whose packed size exceeds its block size. The CLI already
  reported these cleanly, so only library callers see a difference.

- **The matrix memo pinned every transform instance for the life of the
  process.** `cached` was applied to a method, so `self` was part of the key
  and held by a strong reference: instances were never collected and the cache
  grew without bound, roughly 4.6 KB of retained matrices per compress run.
  Matrix construction moved to the module-level `walsh.transforms.hadamard_matrix`,
  memoised on `size` alone, so the cache now holds one entry per distinct block
  size no matter how many transforms are created. `WalshHadamardTransform._build_matrix`
  remains as a thin delegate. `cached` now documents that it must not be
  applied to methods.

## [0.2.0]

### Added

- **PPM support.** Both Netpbm pixmap variants are read -- `P6` binary and `P3`
  ASCII -- and `P6` is written. Header comments are skipped, and a `maxval`
  below 255 is rescaled to the full range. 16-bit samples are rejected with a
  clear message rather than misread.
- Format dispatch by filename suffix (`.bmp`, `.ppm`, `.pnm`) via
  `walsh.image.reader_for`, so `walsh compress photo.ppm out.cim` and
  `walsh extract out.cim photo.ppm` work with no extra flags. A picture can be
  compressed from one format and extracted to another.
- `walsh.image.base.RasterImage`, the abstract base that states the in-memory
  contract every raster format honours.
- Top-down BMP support: a negative header height is now understood instead of
  being read as a huge unsigned number.
- Method-level docstrings across `src/walsh`, with `Args:`, `Returns:` and
  `Raises:` sections wherever they apply.
- `data/earth.ppm`, a 400x400 P6 sample: NASA's Apollo 17 "Blue Marble"
  photograph, public domain, from Wikimedia Commons. Provenance and the
  conversion recipe are recorded in `data/README.md`. Like the other samples it
  is excluded from the sdist.

### Changed

- **`walsh.image` is now a package** rather than a single module, one submodule
  per format: `base`, `bmp`, `ppm`, `cim`, and the shared `_io` helpers. Every
  name the old module exported is re-exported from `walsh.image`, so existing
  imports keep working.
- **Breaking: the in-memory pixel contract is now RGB, top row first**, for
  every format. `BMPImage` previously handed out pixels in BMP's own
  blue-green-red order and bottom-up row order, and `Task` treated them as if
  they were RGB top-down. Both are now converted on read and write.

  This was harmless while BMP was the only format, since the same swap applied
  on the way out. With a second format it is not: without this change,
  BMP -> `.cim` -> PPM would come back with red and blue exchanged and the
  image upside down.

  Consequences:
  - `BMPImage.get_raw_data()` returns different tuples than in 0.1.x. Code
    reading pixels directly needs no change if it treated them as RGB, which
    is what it now genuinely gets.
  - **Compressed output changed.** A `.cim` written by 0.1.x, extracted with
    0.2.0, comes back with red and blue swapped and vertically flipped.
    Re-compress from the source image rather than converting old `.cim` files.
  - Reconstruction quality is unchanged in practice: on `data/image.bmp` the
    PSNR moves from 23.10 dB to 23.09 dB. The fix is about correctness and
    cross-format interoperability, not quality.

  Verified against Pillow as an independent decoder: `BMPImage` now reads
  `data/image.bmp` pixel-for-pixel identically to `PIL.Image.open`.

## [0.1.3]

### Added

- PyPI downloads and GitHub stars badges in the README.
- A `walsh.exceptions` module holding the package's exception types and the
  `EXPECTED_ERRORS` grouping, so that policy lives in one place rather than
  inside the CLI. It imports nothing from the rest of the package, so any
  module can use it without a cycle.
- `WalshError`, a base class for every error the package raises on its own
  behalf. `UnsupportedFileFormatError` now derives from it, so callers can
  catch all package errors without also catching unrelated failures.

### Changed

- The command line interface moved from `argparse` to
  [click](https://click.palletsprojects.com/). `click>=8.1,<9` is now a runtime
  dependency. The command surface is unchanged -- `walsh compress` and
  `walsh extract` take the same arguments and options and produce byte-identical
  output -- but there are three visible differences:
  - `-h` now works as well as `--help`.
  - A missing input file exits `2` (a usage error) rather than `1`, and is
    reported by click before the task starts.
  - Error messages are prefixed `Error:` rather than `walsh:`.
- `walsh.cli.main` is now a click group rather than a `main(argv) -> int`
  function. Programmatic callers should use `click.testing.CliRunner`, or call
  the `walsh.Task` API directly.
- `UnsupportedFileFormatError` moved from `walsh.image` to `walsh.exceptions`.
  Both `from walsh.image import UnsupportedFileFormatError` and
  `from walsh import UnsupportedFileFormatError` keep working.

### Fixed

- An input file too short to hold a header raised `struct.error`, which was not
  caught and printed a traceback. `struct.error` derives from `Exception` rather
  than `ValueError`, so it slipped past the handler. It now reports a clean
  error and exits `1`. This affected the `argparse` interface too.

## [0.1.2]

### Added

- Test coverage measurement with `pytest-cov`, branch coverage on, and a
  `fail_under = 90` floor in `[tool.coverage.report]`.
- CI runs the suite with `--cov` on every supported Python version, so a pull
  request that drops coverage below 90% fails. The `test` jobs are required
  checks on `master`, which makes that failure block the merge.
- The release workflow applies the same floor before building, so nothing
  under-tested can reach PyPI.
- A coverage badge in the README. It states the enforced floor rather than a
  measured number, which is true by construction and cannot drift.
- Coverage reports (`coverage.xml`) uploaded as CI artifacts per Python version.

Coverage at the time of this release is 97% with branch coverage enabled.

## [0.1.1]

### Added

- `uv` support: a `uv.lock` committed to the repository, and a `dev`
  [PEP 735](https://peps.python.org/pep-0735/) dependency group, so
  `uv sync --group dev` reproduces the full development environment.
- Release automation. A version bump landing on `master` tags `v<version>`,
  creates a GitHub Release with these notes attached, and publishes the sdist
  and wheel to [PyPI](https://pypi.org/project/walsh/) via Trusted Publishing.
- CI on every pull request: ruff, `ruff format --check`, strict mypy, the test
  suite on Python 3.10 through 3.14, and a packaging job that builds both
  distributions and smoke-tests the wheel in a clean environment.
- Python 3.14 to the supported classifiers.
- `Issues` and `Changelog` project URLs.

### Changed

- The `dev` extra became the `dev` dependency group. `pip install .[dev]` no
  longer works; use `uv sync --group dev`, `pip install --group dev .`
  (pip 25.1+), or `pip install -r requirements-dev.txt`.

## [0.1.0]

### Added

- `walsh` console script with `compress` and `extract` subcommands.
- Test suite covering the transform, colour models, both container formats, the
  task pipelines and the CLI.
- Type annotations throughout, with a `py.typed` marker; clean under
  `mypy --strict`.

### Changed

- Ported from Python 2.7 to Python 3.10+ and restructured as an installable
  package under `src/walsh`. The old `fal` package and root `transform.py` were
  removed; the demo moved to `examples/roundtrip.py`.
- `numpy` is now the only runtime dependency. `matplotlib` and `Pillow` were
  only ever used by the demo script and became the `demo` extra.
- Spectral coefficients are rounded with `np.rint` on write. Python 2 relied on
  `struct.pack` implicitly truncating floats, so output differs from the
  pre-port result by at most 2 per channel (mean 0.33).
- `np.matrix` replaced with `ndarray` and the `@` operator.
- Requirements files carry explicit version bounds.

### Known issues

Both predate the port and are preserved deliberately, because changing either
alters every compressed output:

- `BMPImage` reads pixel triples in BMP's native B, G, R order while `Task`
  treats them as R, G, B. Round trips are exact, but the YCbCr weights land on
  swapped channels.
- The `coeff_removal` comparison is one-sided rather than on the magnitude, so
  any non-`None` coefficient above `-1/sqrt(2)**n` zeroes essentially the whole
  spectrum.

[0.3.2]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.3.2
[0.3.1]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.3.1
[0.3.0]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.3.0
[0.2.2]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.2.2
[0.2.1]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.2.1
[0.2.0]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.2.0
[0.1.3]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.1.3
[0.1.2]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.1.2
[0.1.1]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.1.1
[0.1.0]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.1.0
