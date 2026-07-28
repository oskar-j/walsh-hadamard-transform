# Sample images

Test inputs for the codec. They are excluded from the sdist and wheel (see
`MANIFEST.in`), because at roughly half a megabyte each they would dominate the
distribution, and the tests that use them skip when they are absent.

## `earth.ppm`

A 400x400 binary (P6) portable pixmap, added in 0.2.0 as the worked example for
PPM support.

- **Source:** [The Earth seen from Apollo 17](https://commons.wikimedia.org/wiki/File:The_Earth_seen_from_Apollo_17.jpg)
  on Wikimedia Commons — the "Blue Marble" photograph taken on 7 December 1972.
- **Credit:** NASA / Apollo 17 crew; taken by either Harrison Schmitt or Ron Evans.
- **Licence:** Public domain. Works produced by NASA are not subject to
  copyright protection in the United States, so no attribution is required —
  the credit above is recorded as good practice, not obligation.
- **Modifications:** the 3000x3002 JPEG original was centre-cropped to square,
  resampled to 400x400 with Lanczos filtering, and written as P6. Wikimedia
  Commons does not host Netpbm files, so a conversion step is unavoidable.

Reproduce it with:

```python
from PIL import Image

im = Image.open("The_Earth_seen_from_Apollo_17.jpg").convert("RGB")
side = min(im.size)
left, top = (im.width - side) // 2, (im.height - side) // 2
im.crop((left, top, left + side, top + side)).resize((400, 400), Image.LANCZOS).save(
    "earth.ppm", format="PPM"
)
```

## `recreated.ppm`

`earth.ppm` after a compress/extract round trip at the default settings, so
current 0.2.0 output: 4.00x on the compressed intermediate, reconstructed at
25.07 dB PSNR.

## `image.bmp`, `recreated.bmp`

The original 400x400 sample carried over from the Python 2 project, and an
example of its reconstruction. `recreated.bmp` predates 0.2.0, so it was
produced before the pixel-order fix; regenerate it with
`python examples/roundtrip.py` rather than treating it as current output.
