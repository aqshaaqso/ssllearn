from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from src.losses import NTXentLoss


def test_nt_xent_loss_is_finite() -> None:
    loss_fn = NTXentLoss(temperature=0.5)
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)

    loss = loss_fn(z1, z2)

    assert torch.isfinite(loss)
    assert loss.ndim == 0

