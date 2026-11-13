"""Cut square patches around the spots of a slide."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

# pathology encoders are usually trained on 20x tiles, about half a micron per pixel
TARGET_MPP = 0.5
OUT_PX = 224


def crop_side(mpp: float, target_mpp: float = TARGET_MPP, out_px: int = OUT_PX) -> int:
    """Side of the square to cut, in image pixels, so that a resize to out_px gives target_mpp."""
    return int(round(out_px * target_mpp / mpp))


class SlideImage:
    """An RGB image that can be cropped around a point.

    The pixels can be a memory map, so a 1.7 GB TIFF is never read in full. Positions are given
    in full resolution pixels and `scale` takes them to the pixels of this image.
    """

    def __init__(self, pixels: np.ndarray, scale: float = 1.0):
        self.pixels = pixels
        self.scale = scale

    @classmethod
    def from_tiff(cls, path: str | Path) -> "SlideImage":
        return cls(tifffile.memmap(path))

    @classmethod
    def from_png(cls, path: str | Path, scale: float) -> "SlideImage":
        return cls(np.asarray(Image.open(path).convert("RGB")), scale)

    def crop(self, x: float, y: float, side: int) -> np.ndarray:
        """`side` x `side` pixels around (x, y). What falls outside the image is white."""
        cx, cy = int(round(x * self.scale)), int(round(y * self.scale))
        top, left = cy - side // 2, cx - side // 2
        height, width = self.pixels.shape[:2]
        out = np.full((side, side, 3), 255, dtype=np.uint8)
        y0, y1 = max(top, 0), min(top + side, height)
        x0, x1 = max(left, 0), min(left + side, width)
        if y0 < y1 and x0 < x1:
            out[y0 - top : y1 - top, x0 - left : x1 - left] = self.pixels[y0:y1, x0:x1]
        return out


def extract_patches(
    image: SlideImage,
    centres: np.ndarray,
    mpp: float,
    out_px: int = OUT_PX,
    target_mpp: float = TARGET_MPP,
) -> Iterator[np.ndarray]:
    """Yield one (out_px, out_px, 3) uint8 patch per centre, at target_mpp microns per pixel.

    `mpp` is the pixel size of the full resolution image.
    """
    side = crop_side(mpp / image.scale, target_mpp, out_px)
    for x, y in centres:
        patch = image.crop(x, y, side)
        if side != out_px:
            patch = np.asarray(Image.fromarray(patch).resize((out_px, out_px), Image.BICUBIC))
        yield patch
