"""A frozen image encoder and the HDF5 cache of its features."""

from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import torch
from transformers import AutoModel

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_tensor(patches: np.ndarray) -> torch.Tensor:
    """(B, H, W, 3) uint8 to a normalised (B, 3, H, W) float tensor."""
    x = (patches.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(0, 3, 1, 2).copy())


class FrozenEncoder:
    """A vision transformer from the Hub used as a feature extractor.

    The feature of a patch is the CLS token of the last layer. Nothing here is trained.
    """

    def __init__(self, name: str = "owkin/phikon", device: str | None = None, model=None):
        self.name = name
        self.device = torch.device(device or pick_device())
        model = model if model is not None else AutoModel.from_pretrained(name)
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.image_size = self.model.config.image_size

    @torch.no_grad()
    def embed(self, patches: np.ndarray) -> np.ndarray:
        x = to_tensor(patches).to(self.device)
        out = self.model(pixel_values=x).last_hidden_state[:, 0]
        return out.float().cpu().numpy()


def embed_batches(
    encoder: FrozenEncoder, patches: Iterable[np.ndarray], batch_size: int = 64
) -> np.ndarray:
    """Run the encoder over a stream of patches, `batch_size` at a time."""
    chunks, batch = [], []
    for patch in patches:
        batch.append(patch)
        if len(batch) == batch_size:
            chunks.append(encoder.embed(np.stack(batch)))
            batch = []
    if batch:
        chunks.append(encoder.embed(np.stack(batch)))
    return np.concatenate(chunks)


def save_features(path: str | Path, barcodes: list[str], embedding: np.ndarray, **attrs) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("barcode", data=np.array(barcodes, dtype="S"))
        f.create_dataset("embedding", data=embedding.astype(np.float32), compression="gzip")
        for key, value in attrs.items():
            f.attrs[key] = value


def load_features(path: str | Path) -> tuple[list[str], np.ndarray, dict]:
    with h5py.File(path, "r") as f:
        barcodes = [b.decode() for b in f["barcode"][:]]
        return barcodes, f["embedding"][:], dict(f.attrs)
