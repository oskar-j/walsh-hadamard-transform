# Sample images

Test inputs for the codec, and the codec's own output for each. They are
excluded from the sdist and wheel (see `MANIFEST.in`), because at roughly half a
megabyte each they would dominate the distribution, and the tests that use them
skip when they are absent.

Every `recreated.*` file is current output, regenerated whenever the codec
changes. Reproduce any of them with:

```
walsh compress data/<source> /tmp/out.cim
walsh extract  /tmp/out.cim data/<recreated>
```

## The Blue Marble sample

`earth.ppm` and `earth.tiff` are the same 400x400 picture in two containers, so
they also serve as a check that the source format does not affect the result:
compressing either produces a byte-identical `.cim`.

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
  as an uncompressed little-endian single-strip TIFF.

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

The two Blue Marble reconstructions are pixel-identical, as they must be: the
codec sees the same picture whichever container it arrives in.

See the [main README](../README.md#reading-the-psnr-figures) for what the dB
figures mean.
