"""Two small towers on top of frozen features, and the temperature that goes with them."""

import math

import torch
import torch.nn.functional as F
from torch import nn


class Head(nn.Module):
    """LayerNorm, one hidden layer, then a linear map into the shared space.

    `input_dropout` switches off a share of the input features, which is a cheap regulariser for
    noisy vectors such as the expression of one spot. Outputs have unit norm.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 256,
        hidden: int = 512,
        dropout: float = 0.1,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(input_dropout),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class TwoTower(nn.Module):
    """A head for the image features and a head for the expression vector.

    `log_scale` is the inverse temperature in log space, as in CLIP, and `bias` is only used by
    the sigmoid loss. Both are learned unless `learn_temperature` is False.
    """

    def __init__(
        self,
        image_dim: int,
        expr_dim: int,
        out_dim: int = 256,
        hidden: int = 512,
        dropout: float = 0.1,
        temperature: float = 0.07,
        learn_temperature: bool = True,
        input_dropout: float = 0.0,
    ):
        super().__init__()
        self.image = Head(image_dim, out_dim, hidden, dropout, input_dropout)
        self.expr = Head(expr_dim, out_dim, hidden, dropout, input_dropout)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1 / temperature)))
        self.bias = nn.Parameter(torch.tensor(-10.0))
        self.log_scale.requires_grad_(learn_temperature)
        self.bias.requires_grad_(learn_temperature)

    @property
    def scale(self) -> torch.Tensor:
        # CLIP clips the inverse temperature at 100
        return self.log_scale.exp().clamp(max=100.0)

    def forward(self, image: torch.Tensor, expr: torch.Tensor):
        return self.image(image), self.expr(expr)
