"""A frozen image encoder and the HDF5 cache of its features."""

from collections.abc import Iterable
from pathlib import Path

import h5py
import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer
from torch import nn
from transformers import AutoModel

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ViT-S weights pre-trained on TCGA patches, published as GitHub release files by Lunit
LUNIT_URL = (
    "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
)
LUNIT = {"lunit/dino-vit-s16": "dino_vit_small_patch16_ep200.torch"}


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


class ClsToken(nn.Module):
    """Wraps a Hugging Face ViT so that it returns the CLS token of the last layer."""

    def __init__(self, vit):
        super().__init__()
        self.vit = vit

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vit(pixel_values=x).last_hidden_state[:, 0]


class FrozenEncoder:
    """Any module that maps (B, 3, H, W) images to (B, D) features, used as it is.

    Nothing here is trained: the parameters are switched off and the module stays in eval mode.
    """

    def __init__(
        self, model: nn.Module, name: str, image_size: int = 224, device: str | None = None
    ):
        self.name = name
        self.image_size = image_size
        self.device = torch.device(device or pick_device())
        self.model = model.eval().to(self.device)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def embed(self, patches: np.ndarray) -> np.ndarray:
        x = to_tensor(patches).to(self.device)
        return self.model(x).float().cpu().numpy()


def lunit_vit_small() -> VisionTransformer:
    """The ViT-S/16 trunk the Lunit weights were trained with. It returns the CLS token."""
    return VisionTransformer(
        img_size=224, patch_size=16, embed_dim=384, depth=12, num_heads=6, num_classes=0
    )


def load_encoder(name: str = "lunit/dino-vit-s16", device: str | None = None) -> FrozenEncoder:
    """`name` is one of the Lunit ViT-S weights, or a Hugging Face model id or local folder
    (owkin/phikon, for instance)."""
    if name in LUNIT:
        model = lunit_vit_small()
        state = torch.hub.load_state_dict_from_url(
            f"{LUNIT_URL}/{LUNIT[name]}", progress=False, map_location="cpu"
        )
        model.load_state_dict(state)
        return FrozenEncoder(model, name, 224, device)
    vit = AutoModel.from_pretrained(name)
    return FrozenEncoder(ClsToken(vit), name, vit.config.image_size, device)


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
