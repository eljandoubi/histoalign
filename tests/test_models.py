import pytest
import torch

from histoalign.losses import info_nce
from histoalign.models import Head, TwoTower


def test_head_outputs_unit_vectors():
    head = Head(32, out_dim=16, hidden=24)
    out = head(torch.randn(5, 32))
    assert out.shape == (5, 16)
    assert torch.allclose(out.norm(dim=-1), torch.ones(5), atol=1e-5)


def test_input_dropout_only_acts_in_training():
    head = Head(32, out_dim=16, hidden=24, dropout=0.0, input_dropout=0.5)
    x = torch.randn(5, 32)
    head.train()
    assert not torch.allclose(head(x), head(x))
    head.eval()
    assert torch.allclose(head(x), head(x))


def test_towers_share_the_output_size_only():
    model = TwoTower(image_dim=40, expr_dim=70, out_dim=16, hidden=32)
    img, expr = model(torch.randn(4, 40), torch.randn(4, 70))
    assert img.shape == expr.shape == (4, 16)


def test_initial_temperature():
    model = TwoTower(8, 8, temperature=0.05)
    assert model.scale.item() == pytest.approx(20.0, rel=1e-5)


def test_scale_is_clipped():
    model = TwoTower(8, 8)
    model.log_scale.data.fill_(10.0)
    assert model.scale.item() == 100.0


def test_temperature_can_be_frozen():
    model = TwoTower(8, 8, learn_temperature=False)
    assert not model.log_scale.requires_grad and not model.bias.requires_grad
    assert TwoTower(8, 8).log_scale.requires_grad


def test_a_few_steps_lower_the_loss_on_paired_inputs():
    torch.manual_seed(0)
    latent = torch.randn(64, 6)
    image = latent @ torch.randn(6, 20) + 0.1 * torch.randn(64, 20)
    expr = latent @ torch.randn(6, 30) + 0.1 * torch.randn(64, 30)
    model = TwoTower(20, 30, out_dim=8, hidden=32, dropout=0.0)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    def loss():
        a, b = model(image, expr)
        return info_nce(a, b, model.scale)

    before = loss().item()
    for _ in range(60):
        opt.zero_grad()
        loss().backward()
        opt.step()
    assert loss().item() < 0.5 * before
