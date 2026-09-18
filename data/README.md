# Sample images

Test inputs for the codec, and the codec's own output for each. They are
excluded from the sdist and wheel (see `MANIFEST.in`), because at roughly half a
megabyte each they would dominate the distribution, and the tests that use them
skip when they are absent.

The files sit in a folder per file type, named after the suffix (0.5.0):
`ppm/earth.ppm`, `png/recreated.png`, `cim/transformed_earth.cim`. A new
format brings its own folder, and nothing else has to be told about it: the
tests find a sample from its name alone, and `MANIFEST.in` prunes the whole
of `data/` rather than listing suffixes.

Every `recreated.*` file and `cim/transformed_earth.cim` is current output,
regenerated whenever the codec changes (last: 0.4.12, which made the
transform's arithmetic exact, so these bytes are what the codec produces on
every platform). Reproduce any of them with:

```
walsh compress data/<type>/<source> /tmp/out.cim
walsh extract  /tmp/out.cim data/<type>/<recreated>
```

`png/recreated.png` is the one exception to "these bytes": a PNG is a zlib
stream, and DEFLATE output may differ from one zlib build to the next, so that
file is pinned by the pixels it decodes to, which are exactly those of
`ppm/recreated.ppm`.

## The Blue Marble sample

`earth.ppm`, `earth.tiff`, `earth.pam`, `earth.npy`, `earth.pkl` and
`earth.png` are the
same 400x400 picture in six containers, so they also serve as a check that the source
format does not affect the result: compressing any of them produces a
byte-identical `.cim`.

- **Source:** [The Earth seen from Apollo 17](https://commons.wikimedia.org/wiki/File:The_Earth_seen_from_Apollo_17.jpg)
  on Wikimedia Commons — the "Blue Marble" photograph taken on 7 December 1972.
- **Credit:** NASA / Apollo 17 crew; taken by either Harrison Schmitt or Ron Evans.
- **Licence:** Public domain. Works produced by NASA are not subject to
  copyright protection in the United States, so no attribution is required —
  the credit above is recorded as good practice, not obligation.
- **Modifications:** the 3000x3002 JPEG original was centre-cropped to square,
  resampled to 400x400 with Lanczos filtering, and written as binary P6.
  Wikimedia Commons does not host Netpbm or TIFF files, so a conversion step is
  unavoidable. `earth.tiff` was then written from `earth.ppm` by this package,
  as an uncompressed little-endian single-strip TIFF, and so was `earth.pam`,
  which is byte-identical to what Netpbm's `pamtopam` produces from
  `earth.ppm`. `earth.npy` is `numpy.save` of the `(400, 400, 3)` `uint8` array
  Pillow reads from `earth.ppm`; this package's own `.npy` writer reproduces
  it byte for byte, which is the check that the two agree on the layout.
  `earth.pkl` is `pickle.dump` of that same array at protocol 4, written under
  NumPy 2, and `numpy.load(allow_pickle=True)` reads it; this package reads it
  through an allowlist instead, executing nothing, under NumPy 1 or 2.
  `earth.png` is `pnmtopng earth.ppm` from Netpbm 11.2, which is libpng's
  encoder: 37 `IDAT` chunks, with Sub, Average and Paeth rows chosen
  adaptively. It is deliberately not this package's own output, so that
  reading it is a check against a foreign writer, and the one sample that is
  both compressed and filtered on the way in.

Reproduce the PPM with:

```python
from PIL import Image

im = Image.open("The_Earth_seen_from_Apollo_17.jpg").convert("RGB")
side = min(im.size)
left, top = (im.width - side) // 2, (im.height - side) // 2
im.crop((left, top, left + side, top + side)).resize((400, 400), Image.LANCZOS).save(
    "earth.ppm", format="PPM"
)
```

## Files

| File | Role | Result |
| --- | --- | --- |
| `image.bmp` | The original 400x400 sample carried over from the Python 2 project | — |
| `recreated.bmp` | `image.bmp` round-tripped at default settings | 23.09 dB |
| `earth.ppm` | Blue Marble, binary P6 pixmap | — |
| `recreated.ppm` | `earth.ppm` round-tripped | 25.07 dB |
| `earth.tiff` | Blue Marble, uncompressed TIFF | — |
| `recreated.tiff` | `earth.tiff` round-tripped | 25.07 dB |
| `earth.pam` | Blue Marble, Netpbm PAM (`P7`, RGB) | — |
| `recreated.pam` | `earth.pam` round-tripped | 25.07 dB |
| `earth.npy` | Blue Marble, bare NumPy array `(400, 400, 3)` `uint8` | — |
| `recreated.npy` | `earth.npy` round-tripped | 25.07 dB |
| `earth.pkl` | Blue Marble, the same array pickled at protocol 4 | — |
| `recreated.pkl` | `earth.pkl` round-tripped, as rows of `(r, g, b)` tuples | 25.07 dB |
| `earth.png` | Blue Marble, 8-bit RGB PNG written by libpng | — |
| `recreated.png` | `earth.png` round-tripped; pinned by its pixels, not its bytes | 25.07 dB |

Each is in the folder named after its suffix, and `transformed_earth.cim`,
what every `earth.*` compresses to, is in `cim/`.

The six Blue Marble reconstructions are pixel-identical, as they must be:
the codec sees the same picture whichever container it arrives in.

See the [main README](../README.md#reading-the-psnr-figures) for what the dB
figures mean.
