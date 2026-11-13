import math

import torch
import torch.nn.functional as F

from histoalign.losses import info_nce, sigmoid_loss


def unit(n, d, seed):
    g = torch.Generator().manual_seed(seed)
    return F.normalize(torch.randn(n, d, generator=g), dim=-1)


def test_info_nce_at_zero_scale_is_log_n():
    a, b = unit(8, 16, 0), unit(8, 16, 1)
    loss = info_nce(a, b, torch.tensor(0.0))
    assert abs(loss.item() - math.log(8)) < 1e-6


def test_info_nce_is_low_when_pairs_match():
    a = unit(8, 16, 0)
    matched = info_nce(a, a, torch.tensor(30.0))
    shuffled = info_nce(a, a.roll(1, dims=0), torch.tensor(30.0))
    assert matched.item() < 0.01
    assert shuffled.item() > 5


def test_info_nce_is_symmetric():
    a, b = unit(6, 8, 2), unit(6, 8, 3)
    scale = torch.tensor(10.0)
    assert abs(info_nce(a, b, scale).item() - info_nce(b, a, scale).item()) < 1e-6


def test_ignoring_a_near_duplicate_lowers_the_loss():
    a = unit(4, 8, 4)
    b = a.clone()
    b[1] = a[0]  # sample 1 looks exactly like sample 0
    scale = torch.tensor(20.0)
    ignore = torch.zeros(4, 4, dtype=torch.bool)
    ignore[0, 1] = ignore[1, 0] = True
    assert info_nce(a, b, scale, ignore).item() < info_nce(a, b, scale).item()


def test_the_diagonal_cannot_be_ignored():
    a, b = unit(5, 8, 5), unit(5, 8, 6)
    everything = torch.ones(5, 5, dtype=torch.bool)
    # only the positive stays in each denominator, so there is nothing left to get wrong
    assert info_nce(a, b, torch.tensor(10.0), everything).item() == 0.0


def test_sigmoid_loss_prefers_matching_pairs():
    a = unit(8, 16, 7)
    scale, bias = torch.tensor(20.0), torch.tensor(-10.0)
    assert (
        sigmoid_loss(a, a, scale, bias).item()
        < sigmoid_loss(a, a.roll(1, dims=0), scale, bias).item()
    )


def test_sigmoid_loss_ignore_removes_terms():
    a, b = unit(6, 8, 8), unit(6, 8, 9)
    scale, bias = torch.tensor(10.0), torch.tensor(-5.0)
    ignore = torch.zeros(6, 6, dtype=torch.bool)
    ignore[0, 1] = ignore[1, 0] = True
    assert sigmoid_loss(a, b, scale, bias, ignore).item() < sigmoid_loss(a, b, scale, bias).item()


def test_gradients_reach_both_sides():
    a = unit(6, 8, 10).requires_grad_()
    b = unit(6, 8, 11).requires_grad_()
    scale = torch.tensor(5.0, requires_grad=True)
    info_nce(a, b, scale).backward()
    assert a.grad.abs().sum() > 0 and b.grad.abs().sum() > 0 and scale.grad is not None
