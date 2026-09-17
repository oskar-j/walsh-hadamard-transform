# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The `## [x.y.z]` headings are load-bearing: the release workflow extracts the
section matching the version in `pyproject.toml` and uses it as the GitHub
Release notes.

## [0.4.16]

### Changed

- `src/walsh/image` is no longer flat. The format modules are grouped by
  family: `raster/` (BMP, TIFF), `netpbm/` (PPM, PAM and the sample code they
  share) and `arrays/` (`.npy`, pickles and the array rules they share).
  `base.py`, `_io.py` and `cim.py` stay at the top, since everything depends
  on them. Nothing public moved: every class, constant and function is still
  exported from `walsh.image` and `walsh`, and codec output is unchanged.
- **The old module paths keep working.** `walsh.image.bmp`, `.tiff`, `.ppm`,
  `.pam`, `.npy` and `.pkl` remain importable as aliases of the new modules,
  so `from walsh.image.bmp import BMPImage` is unaffected. New code should
  import from `walsh.image`. The two private modules, `_netpbm` and `_arrays`,
  moved to `netpbm/_samples` and `arrays/_rules` without an alias.
- `tests/` mirrors the source: `image/` with `raster/`, `netpbm/` and
  `arrays/` inside it, `codec/`, `cli/`, `support/` and `project/`. The
  520-line `test_image.py`, which mixed three subjects, is split along the
  same lines into `image/raster/test_bmp.py`, `image/test_cim.py` and
  `image/test_base.py`, with its one `align` test joining `image/test_io.py`.
  The split was done on the syntax tree, so each test kept its decorators and
  comments, and the count is unchanged: 608 before, 608 after.

### Added

- `tests/image/test_layout.py`: the old and new import paths, each checked as
  the first import of a fresh interpreter, which is the only place an import
  alias can honestly be tested; and that the family packages and
  `walsh.image` export the same objects.
- `pythonpath = ["tests"]` in the pytest configuration, so `from conftest
  import ...` works at any depth, and a `ROOT` in `conftest.py` for tests that
  need a repository path. Four tests counted `parent` hops and broke the
  moment the folders appeared.

### Notes

- `MANIFEST.in` gains the `exclude data/*.pkl` line 0.4.15 should have added
  beside the other sample suffixes. It changes nothing in practice: the sdist
  ships no sample data at all, which was checked against the built archive.
- 608 to 623 tests, coverage unchanged at 99.12%. Run locally on 3.10 through
  3.14.

## [0.4.15]

### Added

- `.pkl` and `.pickle` as an image format: pickled pixels in, and out. What a
  file may hold: a NumPy array, under the `.npy` rules and at every pickle
  protocol; rows of pixels, `[[(r, g, b), ...], ...]`; a flat list of pixels,
  `[(r, g, b), ...]`; or `{"width": w, "height": h, "pixels": [...]}` around
  any of them. Lists and tuples are interchangeable at every level. The writer
  produces rows of `(r, g, b)` tuples of plain `int` at protocol 4, which any
  Python loads without NumPy and whose bytes do not depend on the NumPy that
  wrote them.
- **Nothing in a pickle is ever executed.** `pickle.load` and
  `numpy.load(allow_pickle=True)` run the program a pickle contains, so a
  `walsh compress untrusted.pkl` built on them would run whatever the file
  said. The same files are read here through an allowlist
  (`walsh.image.safe_loads`): containers and numbers need no lookups, and the
  only names a file may refer to are the ones NumPy's own pickles use to
  rebuild an array. Anything else is refused by name before it is called.
  `numpy.ndarray` resolves to an inert token rather than the class, so a file
  cannot call it to allocate, and the array rebuilder accepts only the empty
  array NumPy itself starts from. A pickle written under NumPy 1 loads under
  NumPy 2 and the reverse, which `numpy.load` itself does not manage.
- `--width` and `--height` on `walsh compress`, and `Task.with_input_size`,
  for the one input that cannot say how large it is. A flat list of 160,000
  pixels is 400x400 or 200x800 and nothing here guesses; without a size it is
  refused with a message showing the three ways to give one. For every other
  input, in any format, a declared size is checked against the file, so it is
  honoured or verified and never silently dropped. `RasterImage.declare_size`
  is the hook underneath.
- `.npy` files holding an `object` array, which NumPy stores as a pickle and
  only `numpy.load(allow_pickle=True)` opens, are read through the same
  allowlist. Up to 0.4.14 they were refused from the header. `numpy.load` is
  still never called with pickling enabled.
- A table of contents in the README, generated from its headings.
  `tests/test_readme_toc.py` fails when the two drift and regenerates the
  block when run as a script.
- Samples `data/earth.pkl`, a pickled array written under NumPy 2, and
  `data/recreated.pkl`. Both are in the golden test, and `earth.pkl` is in
  CI's byte-for-byte wheel smoke.

### Changed

- The array rules the `.npy` reader applied (dtype, shape, greyscale, opaque
  RGBA) moved to `walsh.image._arrays` and are shared with the pickle reader.
  Every `.npy` message is unchanged.

### Notes

- Samples must be integers in 0-255, Python's or NumPy's. Floats, booleans,
  out-of-range values and ragged rows are refused by name rather than coerced:
  a float could be scaled 0-1 or 0-255, and a cast would wrap 256 to 0.
- The allowlist closes code execution. It does not make unpickling cheap: a
  pickle of tuples costs about 8 bytes and one Python object per pixel, and a
  genuine `MemoryError` is raised as itself, never relabelled as a bad file.
- 485 to 608 tests, coverage 98.95% to 99.12%. Run locally on 3.10 through
  3.14, and the pickle, `.npy` and golden tests under NumPy 1.26 as well as 2.

## [0.4.14]

### Added

- `Task(transform="dct")` and `Task(transform="haar")`: the transform keyword
  takes a name as well as an instance. `"walsh"` names the default, names are
  case-insensitive, and an unknown one is a `ValueError` at construction that
  lists the known names. `walsh.transforms.transform_for(name)` does the
  lookup and `TRANSFORMS` is the registry.
- `DiscreteCosineTransform` (the orthonormal DCT-II, the transform inside
  JPEG) and `HaarTransform`, moved into the package from the 0.4.13 example,
  with their memoised, read-only matrices `dct_matrix` and `haar_matrix`.
- `MatrixTransform`, a public base class for any separable orthonormal
  transform: a subclass supplies `matrix(size)` and gets both directions and
  the one-product stack methods. The example now selects the three shipped
  transforms by name and defines a fourth, a Hartley transform, this way; it
  trails the others because its matrix puts half of its low frequencies in
  its last rows, while the codec keeps the top-left corner of each spectrum.

### Notes

- **Only the Walsh-Hadamard transform is bit-exact across platforms.** The
  DCT and Haar matrices are irrational, so their products round and the
  rounding follows the BLAS library. They are accurate to rounding and meant
  for comparison; nothing pins their output byte for byte.
- The warning from 0.4.13 matters more now that the names are this
  convenient: **the `.cim` does not record which transform wrote it**, and a
  file written with `"dct"` or `"haar"` decodes under the default without
  complaint into a degraded picture. The `walsh` command still takes no
  transform for that reason; giving the container a field for it is a format
  change and would come first.
- The default's output is unchanged and still byte-identical to the
  checked-in samples. 447 to 485 tests, coverage 98.90% to 98.95%. Run locally
  on 3.10 through 3.14.

## [0.4.13]

### Added

- `Task(transform=...)`: the block transform is a constructor keyword taking a
  `Transform` instance, so a DCT, a Haar transform or anything else can reuse
  the whole pipeline — colour conversion, padding, the crop to the packed
  corner, the container — with only the transform swapped (#41). The default
  is the Walsh-Hadamard transform and its output is byte-identical to 0.4.12.
  Anything that is not a `Transform` instance is a `TypeError` at
  construction, before any file is opened. **Library only, on purpose: the
  `.cim` does not record which transform wrote it**, so a file written with a
  custom one must be extracted by a `Task` given the same one; the `walsh`
  command decodes it without complaint into a degraded picture (16 dB worse
  in the test that pins this). The keyword is for experiments.
- `Transform.transform_stack` and `Transform.inverse_transform_stack`, which
  are what `Task` calls, with every block of a channel at once. A subclass
  only has to implement the two single-block methods: the defaults loop over
  the blocks and reject a block that comes back another shape, by name. The
  built-in transform overrides both with its one broadcast product, so its
  path is as vectorised as before.
- `walsh.transforms.remove_small_coefficients`, the coefficient-removal
  threshold as one function shared by `Task` and `WalshHadamardTransform`.
- `examples/compare_transforms.py`: Walsh-Hadamard, DCT-II and Haar over any
  image at five kept sizes, numpy only. The byte count depends on the
  geometry alone, so each row is like for like. On the Blue Marble sample the
  DCT leads by 0.7 to 2.8 dB, and Haar ties Walsh-Hadamard exactly wherever
  the kept size is a power of two, because the first 2, 4 or 8 functions of
  each span the same piecewise-constant subspace. The table is in the README
  and the test suite runs the example.

### Changed

- Coefficient removal is applied by `Task` to the output of whichever
  transform it was given, rather than by handing the threshold to the
  built-in transform, so `with_coeff_removal` works for any transform. The
  bytes written are unchanged, and `WalshHadamardTransform(coeff=...)` still
  works on its own; a test holds the two routes to the same output.
- `Task.with_coeff_removal` rejects a negative threshold when it is set. It
  used to be rejected only when `run()` built the transform. The command line
  reports it the same way either way.

### Notes

- 426 to 447 tests, coverage 98.86% to 98.90%. Run locally on 3.10 through
  3.14.

## [0.4.12]

### Changed

- The Walsh-Hadamard transform's arithmetic is exact, so the codec's output is
  byte-identical on every platform and BLAS library, in both directions
  (#39). The input is snapped to a binary grid, multiplied by the `+-1` sign
  matrix on both sides, and scaled by `1 / n` last; each step has an exactly
  representable result for any sample the codec produces at any block edge it
  accepts, so the platform's summation order cannot change a bit. Proven in
  tests by bit equality against a butterfly transform, which adds the same
  samples in an unrelated order, and against rational arithmetic; and in CI,
  where the Linux runner's openblas must now reproduce the macOS reference
  files with `cmp`.
- **Output changes, both regenerated in this commit.** Encoding: 3 of 60,000
  coefficients of the Blue Marble sample move by one, the half-way cases the
  old arithmetic rounded by accident (one of them is the coefficient where
  Linux already disagreed with the macOS reference in 0.4.11). Decoding: 1,228
  of 480,000 samples move by one level, 1,196 of them up, because the
  orthonormal matrix products returned `1.999999999999999` where the answer
  was `2` and the colour conversion truncates. Every existing `.cim` still
  decodes; about 0.25% of its samples come back one level higher, which is the
  mathematically correct value. PSNR figures are unchanged to two decimals.
- A butterfly transform was #39's proposal and was measured first: in numpy it
  is about eight times slower than the matrix products at the codec's block
  sizes, because every pass materialises temporaries. The exact matrix product
  costs nothing measurable: 0.92 s compress and 1.06 s extract on a 2000x2000
  image, against 1.00 s and 1.00 s before, within run-to-run noise.

### Removed

- The cross-platform tolerance added in 0.4.11 to `tests/test_golden.py` and
  the wheel smoke (header identical, every coefficient within one, at most
  0.1% differing). Both compare byte for byte again.

### Notes

- 410 to 426 tests, coverage 98.85% to 98.86%. Run locally on 3.10 through
  3.14; the byte-for-byte checks pass on macOS Accelerate locally and on
  Linux openblas in CI.
- Follow-up filed: #41, a `Task(transform=...)` keyword so other block
  transforms can reuse the pipeline for experiments.

## [0.4.11]

Three issues and two findings made along the way. Codec output is unchanged
on every platform; see the note on byte identity below for what that now
means precisely.

### Fixed

- **Block geometry is validated once, in `Task.__init__`, before any file is
  touched.** Closes #21. Three of the four failure modes were silent:
  `--packed-block-size 16`, the flag the README tells users to raise, exited 0
  and wrote a 960 KB file `extract` refuses forever, because numpy clamped the
  crop to the whole block while the header recorded 16; `--packed-block-size
  0` from the library exited 0 at both ends and returned a flat green image;
  and a block edge of 256 saturated the `int16` coefficients so a white
  picture came back mid-grey, again at exit 0. A library caller passing an
  edge of 0 got a bare `ZeroDivisionError`, which is outside
  `EXPECTED_ERRORS`. Now: each edge must be a positive power of two no larger
  than `MAX_BLOCK_SIZE` (128, derived as `COEFF_MAX // 255`, the last edge
  whose DC coefficient fits `int16`), and the packed size must lie between 1
  and the smallest edge. The packed size is deliberately **not** required to
  be a power of two, and 16/16/16 stays legal: both compress and extract
  cleanly today and a test pins each. `CustomizableImage.set_data` mirrors
  the reader's guard on the write side, which is the half that closes the
  corruption path for callers that build a container directly. The CLI now
  builds the `Task` inside its guarded region, so all of these are `Error:`
  lines rather than tracebacks.

- **The requirements mirrors say what `pyproject.toml` says.** Closes #24.
  `requirements.txt` never gained `click` after 0.1.3 made it a runtime
  dependency and `requirements-dev.txt` never gained `pytest-cov` or `build`,
  so the documented non-uv setup could not run the coverage gate CI enforces.
  Both are corrected, `tests/test_requirements_mirror.py` fails if they ever
  drift again, and the README and CONTRIBUTING mention the `--group dev`
  route pip 25.1 offers instead.

- **CI verifies what it claims to.** Closes #25.
  - `uv sync --locked` in every job, and `uv run --locked` in the release
    workflow: a stale `uv.lock` is now a failure with uv's own message rather
    than a silent re-resolve in the runner. Verified on a throwaway branch:
    lint and all five test jobs fail at the install step.
  - The `build` job unpacks the sdist and runs its own tests from inside,
    which is what a downstream packager does and the only thing that can
    catch a `MANIFEST.in` mistake, since the wheel is built from
    `packages.find` and `twine check` never opens the payload. Verified on a
    throwaway branch with `tests/conftest.py` dropped from the sdist: every
    other check stays green and only this step fails, with the collection
    errors. The release workflow runs the same step against the artifacts it
    is about to publish.
  - The wheel smoke test now compresses every sample container and checks
    the result against the reference `.cim`, so packaging cannot silently
    drop a format module.
  - A forced `force_publish` retry downloads the assets attached to the
    existing GitHub Release instead of rebuilding: setuptools output is not
    reproducible, the build backend is unpinned, and a dispatched retry runs
    on the branch tip while the release was pinned to a commit. Build, metadata
    check and the sdist test are gated on the release not existing yet.

### Notes

- **The reference `.cim` was never actually checked in.** Every document since
  0.3.1 called `data/transformed_earth.cim` the checked-in reference, and
  0.4.10's golden test was written against it, but `.gitignore`'s `*.cim`
  rule had kept it out of git the whole time. In CI the golden test skipped
  45 of its 50 cases. The new sdist and smoke steps failed on its absence
  within minutes, which is the whole point of #25. It is tracked now.
- **Byte identity, stated precisely.** With the reference finally present in
  CI, the compress side turned out not to be byte-identical *across BLAS
  implementations*: on Linux with scipy-openblas, 2 of the 60,000
  coefficients differ from the macOS Accelerate reference, each by exactly
  one, where a coefficient lands on an exact half and `np.rint` rounds the
  other way; the decoded picture differs in 44 pixels by one level. Within a
  platform the output is byte-identical, across containers it is
  byte-identical everywhere, and the decode side is byte-identical
  everywhere too, verified by decoding the macOS reference on Linux. The
  golden test and the smoke test now assert exactly that: header identical,
  every coefficient within one, at most 0.1% differing. Deliberately shifting
  100 coefficients or one coefficient by two is still rejected. A
  BLAS-independent transform, the fast Walsh-Hadamard butterflies, would make
  the encode side exact everywhere and is filed as #39.
- The atomic-write regression test in `tests/test_cli.py` was re-pointed a
  second time: every header-field overflow that served as its post-open
  failure is now caught up front (#19, #22, #21), so it injects an `ENOSPC`
  into the block writer after the header has gone out, which is the fault
  the staged write exists to survive.
- 391 to 410 tests, coverage 98.83% to 98.85%. Run locally on 3.10 through
  3.14, and, for the first time with the reference present, in CI.

## [0.4.10]

### Changed

- **The raster contract is array-backed.** Closes #27, the last piece of the
  0.4.0 vectorisation. `RasterImage` now holds its pixels as one `uint8`
  array, and `get_array()` / `set_array()` are the primary accessors: a
  `(height, width, 3)` RGB view of the image's own storage in, and out. Every
  reader decodes straight into it and every writer serialises straight from
  it, so no Python object per pixel is created anywhere between `load` and
  `save`. `get_raw_data()` / `set_raw_data()` remain as converters, so every
  existing caller keeps working.

  Measured on a 2000x2000 image (12 MB of pixels), the same commands, before
  and after:

  | | wall time | peak RSS |
  | --- | --- | --- |
  | `walsh compress` | 3.18 s -> 0.98 s | 742 MB -> 492 MB |
  | `walsh extract` | 2.33 s -> 1.00 s | 1035 MB -> 709 MB |

  The pixel store itself goes from 291 MB as a list of tuples (72.7 bytes per
  pixel) to 12 MB as an array. In the profile of one round trip, function
  calls fall from 563,000 to 95,000 and the transform arithmetic is now the
  largest single item, where before it was 8% behind pixel marshalling at
  49%. What remains of the resident memory is the float64 working set of the
  transform, a different matter from the raster contract.

- **Block stacks flow through the pipeline as one array.** `Task._slice`
  returns the `(count, edge, edge)` stack it always built instead of
  splitting it into a list; `Task` calls the transform on the stack directly,
  which it has accepted since 0.4.0; `_merge` passes an array through with
  `np.asarray` rather than re-stacking it; and `CustomizableImage` holds each
  channel as a stack, with a new `get_stack(channel)` accessor beside the
  unchanged `get_y_data()` and friends, which now return views into it. The
  crops `set_data` makes are stacked once there rather than in the writer.
  `transform_sequence` / `inverse_transform_sequence` keep their documented
  list return; `Task` simply no longer needs them.

- The two remaining per-pixel decoders, BMP and TIFF, are vectorised: one
  `np.frombuffer` and a reshape that drops the row padding, with BMP's
  blue-green-red and bottom-up order handled by two reversed views. BMP's
  writer is a pad and a `tobytes()`; TIFF's body is a `tobytes()`. The BMP
  reader now sizes its read from the header through the bounded
  `read_up_to`, like the `.cim` and `.npy` readers, and still reports the
  first incomplete row.

### Notes

- **Codec output is byte-identical.** A new `tests/test_golden.py` pins it
  for good: every `earth.*` container compresses to the checked-in
  `transformed_earth.cim` (sha256 `29942c84...`), that file extracts to
  every checked-in `recreated.*`, and the BMP sample round-trips to its
  reconstruction. It was written and passing **before** the rewrite, then
  re-run after it, and caught one real defect on the way: `set_array` first
  assigned the dimensions directly and bypassed BMP's `set_dimensions`
  override, which derives the two header size fields, so five header bytes
  came out zero. It now goes through the method.
- **One documented promise changed, deliberately.** `get_raw_data()` used to
  return the live internal list, so mutating it mutated the image. It now
  builds a fresh list per call, and its docstring says so; nothing in the
  package, tests or example relied on the aliasing. `get_array()` is the live
  view now.
- **A trap for future readers, recorded in `CLAUDE.md`.** Routing a list of
  tuples through `np.asarray` is slower than the old per-pixel code, not
  faster. Any list-to-array hop must go through `itertools.chain` into
  `np.fromiter` with an explicit count, which is what `set_raw_data` does.
- `set_array` checks the dtype rather than casting it: a silent cast is how
  a float or a 300 would become a wrong pixel with no error. It keeps a
  reference to a contiguous `uint8` input rather than copying, and copies a
  read-only buffer so `get_array` is always writable.
- 370 to 391 tests, coverage 98.81% to 98.83%; `base.py`, `bmp.py`,
  `npy.py`, `_netpbm.py` and `task.py` are at 100%. Run locally on 3.10
  through 3.14.

## [0.4.9]

### Fixed

Two documentation issues, both closed. Closes #28 and #29. No code path
changes, so codec output is byte-identical.

- **The three `--help` screens no longer print the docstrings' `Args:` and
  `Raises:` sections** (#28). Click reflowed them into one unreadable run,
  complete with RST double-backticks and the internal `click.ClickException`
  class name. A form feed in each docstring now marks where the help stops;
  the Google sections stay in the source verbatim, for readers of it. The
  docstrings must remain plain strings: click splits on the form-feed
  *character*, and a raw string prints a literal `\f` instead (verified on
  click 8.4). A test asserts each screen is free of all four leaks and still
  carries its summary.

- **`--coeff-removal` is described as what it does** (#28). The help said
  "Zero Hadamard matrix entries at or below this threshold", which is the
  0.2.1 bug restated as documentation: the option thresholds *spectral
  coefficients*, strictly below the value, and never touches the matrix. The
  wording was harmful, not just stale. Every entry of the orthonormal matrix
  shares one magnitude, 0.354 at block 8, so a user who read the help and
  picked a value at that scale got a byte-identical file -- `--coeff-removal
  0.35` and no threshold produce the same `.cim`. `Task.with_coeff_removal`
  said the same thing and added "during construction"; both now describe the
  comparison, that it is strict, that only `compress` applies it, and that a
  negative value is rejected at `run()`. The README repoints "the full list"
  from `walsh -h`, which lists only `-v` and `--version`, to `walsh compress
  -h`.

- **The threshold's dependence on block size is documented** (#28). It is
  an absolute magnitude, and the README's tuning table was measured at the
  default 8/16 blocks only. Re-measured for this release: the same
  `--coeff-removal 25` zeroes 79.3% of coefficients at block 8 but 51.0% at
  block 64, so a value tuned against the table and then combined with a
  larger `--y-block-size` quietly stops doing most of its work. The README
  now carries that table, and `WalshHadamardTransform.__init__` and the CLI
  help say to retune with the block size. A normalised threshold would be the
  cleaner design and is deliberately not done: it would change output bytes
  for every existing `--coeff-removal` invocation.

- **`UnsupportedFileFormatError` is documented for every format** (#29). Its
  docstring, unchanged since the original port, said "Raised for BMP files
  that are not 24-bit, single-plane, uncompressed". It is raised from 61
  sites across nine modules and is exported from `walsh`, so that sentence
  was what `help()` showed for a public symbol. It now says what it covers.

- **The stdin "shell pipeline" claims are corrected** (#29). Two comments
  advertised piping through the codec. It is not reachable from the CLI,
  whose arguments are paths, and it is not a pipeline even from the library:
  BMP, TIFF and ASCII PPM seek while parsing, so a real pipe fails with
  `io.UnsupportedOperation` for them; only a `< file` redirect, which is
  seekable, works. Binary PPM, PAM, `.npy` and `.cim` read forward only. The
  `FileSource` and `DEFAULT_RASTER` comments and every reader's `load`
  docstring now say which. A new `tests/test_streams.py` feeds each reader a
  genuinely non-seekable stream and pins that table, which the existing stdin
  test could not, since it substitutes a seekable `io.BytesIO`. Deliberately
  **not** fixed by `allow_dash=True`, which would ship a `-` that dies on
  every real pipe; real pipe support means buffering stdin and is a feature
  decision.

- `CLAUDE.md` said `image.py` re-exports the error; it has been the `image/`
  package since 0.2.0.
- The README's before-and-after image now sits directly under the tagline,
  where a first-time reader sees it, instead of in an "Effects" section at the
  very end; that section is gone, and the image has a real alt text instead
  of its own URL.

### Notes

- 357 to 370 tests. The five new help and docstring assertions fail against
  the old text, checked. Coverage unchanged at 98.81%.

## [0.4.8]

### Added

- **`.npy` support: a bare NumPy array as an image**, read and written under
  the `.npy` suffix. `.npy` is NumPy's own container -- a magic string, a
  one-line header giving shape, dtype and memory order, then the array's
  bytes -- and it is the natural way to hand the codec an image that already
  lives in an array, from Pillow, OpenCV, a camera or a numerical pipeline,
  without converting to a picture format first. It needs no dependency this
  package did not already have, and a headerless raw file's one shortcoming,
  that nothing records its shape, is exactly what the header supplies.

  An array carries no colour metadata, so the profile is declared rather
  than detected. Accepted: `uint8` arrays of shape `(height, width, 3)` as
  RGB; `(height, width)` and `(height, width, 1)` as greyscale, replicated
  across the channels; and `(height, width, 4)` as RGBA only when every alpha
  is 255, in which case the channel is dropped. Rejected by name: any other
  dtype, because a float array could be scaled 0-1 or 0-255 and guessing is
  how silent corruption starts; any other channel count; and a fourth channel
  that is not fully opaque, which is either transparency, only flattenable by
  inventing a background, or CMYK, a colour-space conversion undefined
  without a profile that the shape alone cannot tell apart from RGBA. Channel
  order is RGB by definition; a BGR array as OpenCV produces is
  `array[..., ::-1]`, and a test proves that round trip. Both header versions
  NumPy writes for plain arrays are read (1.0 and 2.0); Fortran-ordered
  bodies decode correctly.

  The header is validated before a byte of the body is read, so a 144-byte
  file declaring a 12.9 GB array is refused at baseline memory. A pickled
  object array is refused from the header's dtype alone and `numpy.load` is
  never invoked with pickling enabled, which would execute code from the
  file. The writer always produces `(height, width, 3)` `uint8` in C order,
  through the staged atomic write, and its bytes are identical to
  `numpy.save` of the same array.

  Verified against numpy itself and Pillow: a file numpy wrote from Pillow's
  array of the Blue Marble sample loads pixel-for-pixel equal to the PPM
  reader, compresses to the checked-in `transformed_earth.cim` byte for byte,
  and this package's writer reproduces that numpy-written file exactly.
  `data/earth.npy` and `data/recreated.npy` are that file and the codec's
  output for it.

- `NPYImage` is exported from `walsh` and `walsh.image`, with `NPY_DTYPE` and
  `NPY_CHANNELS`. `tests/test_npy.py` covers the round trip, the numpy
  reference, greyscale and opaque-RGBA handling, transparency, every rejected
  dtype and channel count by name, the pickled file, the bomb, Fortran order,
  format versions, malformed and truncated files, and dispatch. `.npy` joins
  the cross-format identity tests: BMP, PPM, PAM, TIFF and NPY of one picture
  compress to the same `.cim`, and the sample agrees across all four
  containers.

### Changed

- The chunked `read_up_to` moved from `cim.py` to `image/_io.py`, so the
  `.cim` and `.npy` readers share one implementation for any read sized from
  a header field. No behaviour change.
- The CLI help lists `.npy` among the suffixes.

### Notes

- **Codec output is unchanged.** All five previous sample outputs reproduce
  byte for byte and the new container agrees with them.
- 320 to 357 tests, coverage 98.72% to 98.81%; `npy.py` and `image/_io.py`
  are at 100%. Run locally on 3.10 through 3.14.

## [0.4.7]

### Fixed

- **A `.cim` header is checked for self-consistency before anything is
  allocated from it.** Closes #22.

  The container has no signature, so any 26 bytes parse as a header, and every
  consumer downstream allocated, divided and reshaped on the raw fields.
  `walsh extract data/image.bmp out.bmp`, an ordinary argument mix-up on the
  repository's own sample, decoded the BMP as a 1396067650x7 image, ran for
  62 seconds behind a 78 GB allocation, and then blamed a block size. A
  crafted 26-byte file asked for 2 PiB and died with a bare `MemoryError`
  traceback. A zero width was a bare `ZeroDivisionError`; a zero height exited
  0 and wrote a 12-byte stub; and `--packed-block-size 0` encoded a header-only
  file that decoded, exit 0, to solid green.

  `_read_header` now validates the geometry it just read, after all three
  block descriptions so a file cut short in the header is still reported as
  truncated: both dimensions positive; per channel, the block size a positive
  power of two (the transform requires it anyway), the packed size between 1
  and the block size, and the block count either 0 -- the "empty channel"
  state `extract` fills with a neutral -- or exactly what a plane of those
  dimensions is cut into. That last rule is exact, not a heuristic: the
  encoder pads to a block multiple, so a valid file always carries precisely
  that many. Every rejection names the channel and the field:

  ```
  Error: invalid .cim y block description: block size 0 is not a positive power of two
  Error: invalid .cim header: dimensions must be positive, got 16x0
  Error: invalid .cim y block description: 4 blocks declared, but a 16x64 image in 8-pixel blocks has 16
  ```

  The BMP-as-`.cim` case now fails in 0.2 seconds at baseline memory. The
  blocks-in-a-plane arithmetic moved to `walsh.image.blocks_for`, which both
  `Task` and the reader use, so encoder and reader cannot disagree about it.

- **The reader no longer asks the stream for the declared coefficient total
  in one call.** `file.read(n)` allocates `n` bytes before reading, so a
  header declaring terabytes exhausted memory before the first byte arrived,
  even when the header was otherwise plausible. Coefficients are now read in
  bounded chunks and cost only what the file holds; a short file still
  reports the first incomplete block as before.

- **`Task._merge` takes its row count from the declared height**, not from
  the number of blocks it was handed. It used to drop surplus blocks silently
  and fail a shortfall with a numpy reshape error naming no dimension. It now
  raises `ValueError` naming the plane, the block and the count it needed.
  For any file the reader accepts this is unreachable; it is for direct
  callers.

- **`--packed-block-size`, `--y-block-size` and `--chroma-block-size` refuse
  values below 1 before any work is done** (`click.IntRange(min=1)`, exit
  code 2). Zero produced the solid-green file above; a negative value failed
  by accident inside `struct.pack` with a message naming neither the option
  nor the limit.

### Notes

- **Codec output and the container format are unchanged.** This is read-side
  validation only: no byte of the format moves, deliberately not a magic
  number, which would shift every offset and invalidate every existing file.
  All five checked-in sample outputs reproduce byte for byte; PPM, PAM and
  TIFF sources still agree; `data/transformed_earth.cim` still decodes to
  `data/recreated.ppm` exactly; and a test round-trips five encoder
  configurations, including a 13x7 image and `--packed-block-size 3`, through
  the hardened reader.
- `MemoryError` was deliberately **not** added to `EXPECTED_ERRORS`. Its
  message is empty, so the CLI would print a bare `Error:`, and it would
  relabel a genuine out-of-memory on a legitimately large image as malformed
  input. The allocations are bounded instead.
- Two test fixtures encoded geometrically impossible headers and were
  corrected (16x16 with 5 blocks of 8 became 40x8; 4x2 with 3 blocks of 2
  became 6x2), each keeping the truncation message it was pinning.
- One bound remains open by design: a header that is self-consistent but
  describes an enormous image still allocates in proportion to that image,
  as any decoder must. A block-size ceiling shared by encoder and reader would
  close it and belongs with #21, since both sides have to agree on it.
- 296 to 320 tests, 21 of the new assertions fail against the unfixed code.
  Coverage 98.68% to 98.72%; `cli.py` and `task.py` at 100%.

## [0.4.6]

### Fixed

- **An image too large for the `.cim` container is now refused with a message
  that says so.** Closes #19.

  The container stores each channel's block count in a 16-bit field, so at the
  default 8-pixel luma block the codec caps out at 65535 blocks, about 4.2
  megapixels. That is below any phone photo or 4K frame. Larger input died
  with `Error: 'H' format requires 0 <= number <= 65535` -- naming no file,
  dimension, channel or limit -- and only after the whole image had been read,
  colour-converted and transformed.

  The block count follows from the dimensions and the configured block sizes
  alone, so `Task.compress` now checks it as soon as the dimensions are known,
  before any of that work. A 12 megapixel photo is refused in about a second
  rather than failing after the full pipeline:

  ```
  Error: image is too large for the .cim container: a 4032x3024 image needs
  190512 luma blocks of 8 pixels, and the format stores at most 65535 per
  channel, which is 4.2 megapixels at this block size. Retry with
  --y-block-size 16 or larger, or scale the image down.
  ```

  The suggested block size is computed, not guessed, and a test asserts that
  compressing with it actually succeeds. Chroma is checked the same way and
  names `--chroma-block-size` instead.

- `CustomizableImage.set_descriptions` raises `ValueError` for a block count
  the field cannot hold, as a backstop for callers building a container
  directly rather than through `Task`. `MAX_BLOCKS_PER_CHANNEL` is exported
  from `walsh.image` so the limit has a name.

### Notes

- **The container format is unchanged, deliberately.** Widening the count
  field to 32 bits would raise the ceiling but break every existing `.cim`
  exactly as 0.2.0 did. That is a separate decision, not a bug fix, so this
  release reports the limit rather than moving it. `--y-block-size 16` remains
  the escape hatch and is now discoverable, since the error names it.
- **Codec output is unchanged.** Every image that compresses today is under
  the limit and takes the same path: all five checked-in sample outputs
  reproduce byte for byte, and BMP, PPM, PAM and TIFF of one picture still
  produce an identical `.cim`.
- `tests/test_cli.py`'s atomic-write regression test was re-pointed. It used
  this overflow as its example of a failure occurring *after* the output is
  opened, and the new up-front check means that no longer reaches the writer --
  its own assertion caught that. It now uses an oversized `--packed-block-size`,
  and was re-checked against a non-atomic writer to confirm it still fails
  there.
- 281 to 296 tests, coverage 98.64% to 98.68%; `task.py` is back to 100%.

## [0.4.5]

### Fixed

Three preconditions the raster layer documented or implied but never checked.
Closes #23.

- **A BMP declaring a width or height of zero is rejected** rather than
  accepted silently. A zero width makes the row stride zero, so the truncation
  guard in `_read_data` compares `0 < 0` and can never fire: the reader looped
  over the whole declared height against a file holding no pixel data, and the
  codec then wrote a `.cim` that `walsh extract` could not read back, failing
  with `range() arg 3 must not be zero`. A declared height of two million took
  1.7 s and 173 MB to read nothing. PPM and PAM already rejected a dimension of
  zero; this brings BMP in line. A *negative* height is still legal and still
  means the rows are stored top-down.

- **TIFF strip reads are bounded by the pixels the header declares.** Nothing
  stopped several strip entries pointing at the same region, so memory grew as
  strips times strip size with no cap and no complaint: a 180 KB file made an
  8x8 image cost 385 MB and still loaded, reporting no error. Each read is now
  clamped to the bytes still outstanding, and a strip left with nothing to
  contribute is reported by name. The same file now costs nothing and is
  refused.

  The obvious check, rejecting `sum(counts) > expected` up front, was
  deliberately **not** used: `RowsPerStrip` need not divide the height, so a
  final strip padded to a whole number of rows legitimately overshoots, and
  such files load correctly today. Clamping returns byte-identical pixels for
  every file that was readable before, because the strip arrays are in image
  order however the strips sit on disk. Both cases now have a test.

- **`save()` refuses a pixel count that contradicts the dimensions.**
  `RasterImage` documents that it holds `width * height` pixels, but nothing
  enforced it. Too few pixels wrote a file no reader in this package can load,
  reporting success; too many dropped the surplus silently, and left a BMP
  whose own size fields contradicted its body. `RasterImage._check_complete`
  now raises `ValueError`, and every `save()` calls it **before writing any
  header**, so a rejected save leaves nothing behind.

  The check is at `save()` rather than in the setters, because the documented
  two-call build, `set_dimensions` then `set_raw_data`, is transiently
  inconsistent by design. It is scoped to `RasterImage`, so
  `CustomizableImage`, which legitimately holds dimensions with no blocks, is
  untouched.

### Notes

- **Codec output is unchanged.** All five checked-in sample outputs reproduce
  byte for byte, and BMP, PPM, PAM and TIFF of one picture still compress to an
  identical `.cim`. No valid file changes behaviour; only malformed ones, which
  now fail by name instead of silently.
- No CLI user could reach the third defect: `Task.extract` always supplies
  exactly `width * height` pixels. It is hardening of a public, `py.typed` API
  that `CONTRIBUTING.md` tells format authors to subclass.
- 259 to 281 tests, coverage 98.45% to 98.64%; `bmp.py` and `base.py` reach
  100%. All 22 new assertions fail against the unfixed readers, checked.

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
