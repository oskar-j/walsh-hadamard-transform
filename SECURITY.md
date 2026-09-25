# Security policy

`walsh` reads image files, and the files it is pointed at can come from
anyone. A file that makes it run code, write somewhere it was not asked to,
crash the interpreter, or spend memory or time out of all proportion to the
file's size is a vulnerability. Please report it privately, so that a fix can
ship before the details are public.

## Reporting a vulnerability

Use GitHub's private reporting: **Report a vulnerability** on the repository's
[Security tab](https://github.com/oskar-j/walsh-hadamard-transform/security),
which opens [this form](https://github.com/oskar-j/walsh-hadamard-transform/security/advisories/new).
Only the maintainer sees what you send. If you cannot use it, email the
maintainer at `oskar.jarczyk@gmail.com` with "walsh security" in the subject.

Please do not open a public issue, pull request or discussion about a
vulnerability.

A useful report has:

- the versions: `walsh --version`, Python, and NumPy;
- the smallest input that shows the problem, or a short script that writes it;
- the command or call that reads it, and what happens: code run, a file
  written, a crash, or the memory and time it takes.

## What happens next

The report is confirmed or declined in its private advisory. A confirmed
problem is fixed in a new release, since every fix ships as the next patch
version. The advisory is published with the fix and credits you, unless you
would rather it did not.

## Supported versions

Only the latest release on [PyPI](https://pypi.org/project/walsh/). There are
no maintenance branches, and fixes are not backported, so upgrading is how a
fix reaches you.

## Scope

In scope are the boundaries the code already treats as boundaries:

- **The pickle reader**: `.pkl` and `.pickle` files, and a `.npy` file holding
  an object array, which is a pickle inside. It never calls `pickle.load`. It
  resolves names from a short allowlist, and no entry on that list may import,
  open or allocate from its arguments. Anything that makes it run, import or
  open something, or allocate at a file's direction, is a vulnerability.
- **Resource use driven by a header.** A reader should cost memory and time in
  proportion to what the file holds, or to a picture its format can describe:
  a `.cim` channel holds at most 65,535 blocks of at most 128x128. A small file
  that costs gigabytes of memory or minutes of CPU is in scope.
- **A crash of the interpreter itself**, such as a segmentation fault, from
  reading any file.
- **Writes.** Output is staged in a temporary file beside the destination and
  then moved over it. A way to make `walsh` write or replace any file other
  than the output it was given is in scope.

Not in scope, but welcome as an ordinary issue:

- a malformed file refused with a traceback rather than an `Error:` line;
- a large picture costing memory in proportion to its size;
- output quality: the codec is lossy by design, and a `.cim` does not record
  which transform wrote it, which the README documents;
- anything that needs control of the environment `walsh` runs in, such as its
  installed packages or `PYTHONPATH`.
