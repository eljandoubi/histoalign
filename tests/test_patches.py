import numpy as np

from histoalign.patches import OUT_PX, SlideImage, crop_side, extract_patches
from histoalign.visium import read_slide
from tests.slides import GRID


def test_crop_side():
    assert crop_side(0.5) == 224
    assert crop_side(0.31) == 361
    assert crop_side(1.0, target_mpp=0.5, out_px=100) == 50


def test_crop_is_centred(fake_slide):
    image = SlideImage.from_tiff(next(fake_slide.root.glob("*_image.tif")))
    patch = image.crop(450, 450, 11)
    assert patch.shape == (11, 11, 3)
    # red carries x and green carries y modulo 256
    assert tuple(patch[5, 5]) == (450 % 256, 450 % 256, 128)
    assert tuple(patch[5, 6]) == (451 % 256, 450 % 256, 128)


def test_tiff_is_memory_mapped(fake_slide):
    image = SlideImage.from_tiff(next(fake_slide.root.glob("*_image.tif")))
    assert isinstance(image.pixels, np.memmap)


def test_border_is_white():
    pixels = np.zeros((50, 50, 3), dtype=np.uint8)
    patch = SlideImage(pixels).crop(2, 2, 11)
    assert patch[0, 0].tolist() == [255, 255, 255]
    assert patch[5, 5].tolist() == [0, 0, 0]
    assert patch[10, 10].tolist() == [0, 0, 0]


def test_far_outside_is_all_white():
    pixels = np.zeros((50, 50, 3), dtype=np.uint8)
    assert (SlideImage(pixels).crop(500, 500, 8) == 255).all()


def test_scale_moves_the_window():
    pixels = np.zeros((100, 100, 3), dtype=np.uint8)
    pixels[20, 30] = (1, 2, 3)
    # (x, y) = (60, 40) in full resolution is (30, 20) in a half size image
    patch = SlideImage(pixels, scale=0.5).crop(60, 40, 5)
    assert patch[2, 2].tolist() == [1, 2, 3]


def test_patches_have_the_encoder_size(fake_slide):
    slide = read_slide(fake_slide.root)
    image = SlideImage.from_tiff(slide.image_path)
    patches = list(extract_patches(image, slide.adata.obsm["spatial"], slide.mpp))
    assert len(patches) == GRID * GRID
    assert all(p.shape == (OUT_PX, OUT_PX, 3) and p.dtype == np.uint8 for p in patches)
