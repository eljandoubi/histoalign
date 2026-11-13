"""Train the two projection heads on cached features."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from histoalign.encoders import pick_device
from histoalign.losses import info_nce, sigmoid_loss
from histoalign.models import TwoTower


@dataclass
class Config:
    out_dim: int = 256
    hidden: int = 512
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    epochs: int = 60
    batch_size: int = 256
    loss: str = "infonce"  # or "sigmoid"
    temperature: float = 0.07
    learn_temperature: bool = True
    neighbour_radius: float = 0.0  # in spot pitches, 0 keeps every other spot as a negative
    seed: int = 0


@dataclass
class Pairs:
    """Spots that went through both sides: frozen image features and scaled expression."""

    image: np.ndarray  # (n, image_dim)
    expr: np.ndarray  # (n, n_genes)
    coords: np.ndarray  # (n, 2) pixels
    slide: np.ndarray  # (n,) integer id of the slide each spot comes from

    def __len__(self) -> int:
        return len(self.image)

    def take(self, index) -> "Pairs":
        return Pairs(self.image[index], self.expr[index], self.coords[index], self.slide[index])


def neighbour_mask(coords: torch.Tensor, slide: torch.Tensor, radius_px: float) -> torch.Tensor:
    """True where two spots are on the same slide and no further apart than radius_px."""
    close = torch.cdist(coords, coords) <= radius_px
    return close & (slide[:, None] == slide[None, :])


def spatial_block_split(
    coords: np.ndarray, slide: np.ndarray, frac: float = 0.2, gap_px: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the right-hand strip of every slide.

    Spots within `gap_px` of the strip are dropped from training as well, so that no training
    spot is a neighbour of a held out one. Returns (train mask, held out mask).
    """
    held = np.zeros(len(coords), dtype=bool)
    buffer = np.zeros(len(coords), dtype=bool)
    for s in np.unique(slide):
        here = slide == s
        cut = np.quantile(coords[here, 0], 1 - frac)
        held |= here & (coords[:, 0] >= cut)
        buffer |= here & (coords[:, 0] >= cut - gap_px) & (coords[:, 0] < cut)
    return ~(held | buffer), held


def train(
    pairs: Pairs,
    cfg: Config,
    radius_px: float = 0.0,
    device: str | None = None,
    log=print,
) -> tuple[TwoTower, list[float]]:
    """Fit the two heads. Returns the model in eval mode and the mean loss of each epoch."""
    if cfg.loss not in ("infonce", "sigmoid"):
        raise ValueError(f"unknown loss {cfg.loss!r}")
    device = torch.device(device or pick_device())
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    model = TwoTower(
        pairs.image.shape[1],
        pairs.expr.shape[1],
        out_dim=cfg.out_dim,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        temperature=cfg.temperature,
        learn_temperature=cfg.learn_temperature,
    ).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    batch_size = min(cfg.batch_size, len(pairs))
    steps_per_epoch = len(pairs) // batch_size
    total = cfg.epochs * steps_per_epoch
    warmup = max(1, total // 20)

    def lr_at(step: int) -> float:
        return (
            min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total)))
        )

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    image = torch.from_numpy(pairs.image).to(device)
    expr = torch.from_numpy(pairs.expr).to(device)
    coords = torch.from_numpy(pairs.coords.astype(np.float32)).to(device)
    slide = torch.from_numpy(pairs.slide).to(device)

    history = []
    for epoch in range(cfg.epochs):
        model.train()
        order = rng.permutation(len(pairs))
        running = 0.0
        for b in range(steps_per_epoch):
            idx = torch.from_numpy(order[b * batch_size : (b + 1) * batch_size]).to(device)
            a, e = model(image[idx], expr[idx])
            ignore = neighbour_mask(coords[idx], slide[idx], radius_px) if radius_px > 0 else None
            if cfg.loss == "infonce":
                loss = info_nce(a, e, model.scale, ignore)
            else:
                loss = sigmoid_loss(a, e, model.scale, model.bias, ignore)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
        history.append(running / steps_per_epoch)
        if log and (epoch + 1) % 10 == 0:
            scale = model.scale.item()
            log(f"epoch {epoch + 1}/{cfg.epochs} loss {history[-1]:.4f} scale {scale:.1f}")
    return model.eval(), history


@torch.no_grad()
def encode(
    model: TwoTower, image: np.ndarray, expr: np.ndarray, chunk: int = 2048
) -> tuple[np.ndarray, np.ndarray]:
    """Unit norm embeddings of both sides."""
    device = next(model.parameters()).device
    a, e = [], []
    for start in range(0, len(image), chunk):
        ia = torch.from_numpy(image[start : start + chunk]).to(device)
        ie = torch.from_numpy(expr[start : start + chunk]).to(device)
        out_a, out_e = model(ia, ie)
        a.append(out_a.cpu().numpy())
        e.append(out_e.cpu().numpy())
    return np.concatenate(a), np.concatenate(e)
