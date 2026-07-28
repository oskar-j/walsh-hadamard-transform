# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

The `## [x.y.z]` headings are load-bearing: the release workflow extracts the
section matching the version in `pyproject.toml` and uses it as the GitHub
Release notes.

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

[0.1.2]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.1.2
[0.1.1]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.1.1
[0.1.0]: https://github.com/oskar-j/walsh-hadamard-transform/releases/tag/v0.1.0
