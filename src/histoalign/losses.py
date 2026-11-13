"""Contrastive losses on two batches of embeddings: row i of `a` is paired with row i of `b`."""

import torch
import torch.nn.functional as F


def _off_diagonal(ignore: torch.Tensor) -> torch.Tensor:
    """The positive pair of a row is never left out, whatever the mask says."""
    eye = torch.eye(len(ignore), dtype=torch.bool, device=ignore.device)
    return ignore & ~eye


def info_nce(
    a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, ignore: torch.Tensor | None = None
) -> torch.Tensor:
    """Symmetric InfoNCE, the CLIP loss. `a` and `b` are L2 normalised and `scale` multiplies
    their cosine similarities.

    `ignore` is a symmetric (n, n) boolean matrix. Pairs marked True are taken out of the
    denominators: two neighbouring spots look alike and should not be pushed apart.
    """
    logits = scale * a @ b.T
    if ignore is not None:
        logits = logits.masked_fill(_off_diagonal(ignore), float("-inf"))
    target = torch.arange(len(a), device=a.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


def sigmoid_loss(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
    ignore: torch.Tensor | None = None,
) -> torch.Tensor:
    """The pairwise sigmoid loss of SigLIP: each pair is its own binary problem and only the
    diagonal is positive. The sum over pairs is divided by the batch size."""
    n = len(a)
    logits = scale * a @ b.T + bias
    labels = 2 * torch.eye(n, device=a.device) - 1
    loss = -F.logsigmoid(labels * logits)
    if ignore is not None:
        loss = loss * (~_off_diagonal(ignore))
    return loss.sum() / n
