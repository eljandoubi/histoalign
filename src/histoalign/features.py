"""Turn a slide into cached image features and a spot table."""

from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from histoalign.encoders import FrozenEncoder, embed_batches, save_features
from histoalign.patches import SlideImage, extract_patches
from histoalign.visium import Slide


def spot_table(slide: Slide) -> pd.DataFrame:
    """One row per spot: array position and pixel position in the full resolution image."""
    adata = slide.adata
    x, y = adata.obsm["spatial"].T
    return pd.DataFrame(
        {
            "barcode": adata.obs_names.to_numpy(),
            "slide": slide.sample_id,
            "array_row": adata.obs["array_row"].to_numpy(),
            "array_col": adata.obs["array_col"].to_numpy(),
            "x": x,
            "y": y,
        }
    )


def with_progress(items: Iterable, total: int, label: str, every: int = 500) -> Iterator:
    for i, item in enumerate(items, 1):
        yield item
        if i % every == 0 or i == total:
            print(f"{label}: {i}/{total} patches", flush=True)


def extract(
    slide: Slide,
    encoder: FrozenEncoder,
    out_dir: str | Path,
    batch_size: int = 64,
    hires: bool = False,
) -> Path:
    """Embed the patch around every spot and write <sample>.h5 and <sample>.parquet in out_dir.

    With `hires` the small PNG that Space Ranger writes is used instead of the full resolution
    TIFF. It is quick, but a patch then holds far fewer real pixels.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if hires:
        image = SlideImage.from_png(slide.hires_path, slide.hires_scale)
    else:
        image = SlideImage.from_tiff(slide.image_path)

    spots = spot_table(slide)
    centres = slide.adata.obsm["spatial"]
    patches = extract_patches(image, centres, slide.mpp, out_px=encoder.image_size)
    patches = with_progress(patches, len(centres), slide.sample_id)
    embedding = embed_batches(encoder, patches, batch_size)

    save_features(
        out_dir / f"{slide.sample_id}.h5",
        list(spots["barcode"]),
        embedding,
        encoder=encoder.name,
        patch_px=encoder.image_size,
        mpp=slide.mpp,
        source="hires" if hires else "tiff",
    )
    spots.to_parquet(out_dir / f"{slide.sample_id}.parquet", index=False)
    return out_dir / f"{slide.sample_id}.h5"
