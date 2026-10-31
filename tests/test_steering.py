"""Tests for the steering primitives (#75)."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from hprobes.steering import (
    SteeringController,
    SteeringSpec,
    contrastive_direction,
    median_residual_norm,
    write_vector,
)


class _FakeMLP(nn.Module):
    def __init__(self, d_model: int = 4, d_ff: int = 6, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(self.up_proj(x))


class _FakeLayer(nn.Module):
    def __init__(self, d_model: int = 4, d_ff: int = 6, seed: int = 0):
        super().__init__()
        self.mlp = _FakeMLP(d_model=d_model, d_ff=d_ff, seed=seed)

    def forward(self, x):
        return x + self.mlp(x)


class _FakeModel(nn.Module):
    def __init__(self, n_layers: int = 1, d_model: int = 4, d_ff: int = 6, vocab: int = 20):
        super().__init__()
        torch.manual_seed(0)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [_FakeLayer(d_model=d_model, d_ff=d_ff, seed=i) for i in range(n_layers)]
        )
        self.embed = nn.Embedding(vocab, d_model)

    def forward(self, input_ids):
        hidden = self.embed(input_ids)
        for layer in self.model.layers:
            hidden = layer(hidden)
        return SimpleNamespace(logits=hidden)


def _base_logits(model, input_ids):
    with torch.inference_mode():
        return model(input_ids=input_ids).logits.clone()


def test_write_vector_shape():
    model = _FakeModel()
    vector = write_vector(model, 0, 2)
    assert vector.shape == (4,)
    assert torch.isfinite(vector).all()
    assert vector.norm() > 0


def test_neuron_write_noop_is_bit_exact():
    model = _FakeModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    base = _base_logits(model, input_ids)

    with SteeringController(
        model,
        [SteeringSpec(kind="neuron_write", layer=0, neuron=2, dose=0.0, scale=1.0)],
    ) as controller:
        zero = _base_logits(model, input_ids)
    assert torch.equal(base, zero)
    assert controller.calls > 0
    assert controller.injected_norms == [0.0]


def test_neuron_write_changes_output_and_norm_is_dose():
    model = _FakeModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    base = _base_logits(model, input_ids)
    assert base.shape == (1, 4, 4)

    with SteeringController(
        model,
        [SteeringSpec(kind="neuron_write", layer=0, neuron=2, dose=0.5, scale=2.0)],
    ) as controller:
        steered = _base_logits(model, input_ids)

    assert not torch.equal(base, steered)
    assert controller.injected_norms[0] == pytest.approx(1.0)


def test_direction_noop_and_dose():
    model = _FakeModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    base = _base_logits(model, input_ids)
    vector = torch.ones(4)

    with SteeringController(
        model,
        [SteeringSpec(kind="direction", layer=0, vector=vector, dose=0.0)],
    ) as controller:
        zero = _base_logits(model, input_ids)
    assert torch.equal(base, zero)

    with SteeringController(
        model,
        [SteeringSpec(kind="direction", layer=0, vector=vector, dose=1.0)],
    ) as controller:
        steered = _base_logits(model, input_ids)
    assert not torch.equal(base, steered)
    assert controller.injected_norms[0] == pytest.approx(1.0)


def test_alpha_scale_and_ablation():
    model = _FakeModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    base = _base_logits(model, input_ids)

    with SteeringController(
        model,
        [SteeringSpec(kind="alpha_scale", layer=0, neuron=2, dose=1.0)],
    ):
        same = _base_logits(model, input_ids)
    assert torch.equal(base, same)

    with SteeringController(
        model,
        [SteeringSpec(kind="ablation", layer=0, neuron=2)],
    ):
        ablated = _base_logits(model, input_ids)
    with SteeringController(
        model,
        [SteeringSpec(kind="alpha_scale", layer=0, neuron=2, dose=0.0)],
    ):
        zeroed = _base_logits(model, input_ids)
    assert torch.equal(ablated, zeroed)
    assert not torch.equal(base, ablated)


def test_positions_last_only_changes_last_token():
    model = _FakeModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    base = _base_logits(model, input_ids)
    vector = torch.ones(4)

    with SteeringController(
        model,
        [SteeringSpec(kind="direction", layer=0, vector=vector, positions="last")],
    ):
        steered = _base_logits(model, input_ids)

    assert torch.equal(base[:, :-1, :], steered[:, :-1, :])
    assert not torch.equal(base[:, -1, :], steered[:, -1, :])


def test_hook_cleanup_restores_model():
    model = _FakeModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    base = _base_logits(model, input_ids)

    controller = SteeringController(
        model,
        [SteeringSpec(kind="neuron_write", layer=0, neuron=2, dose=1.0)],
    )
    assert controller.handles
    controller.close()
    assert not controller.handles
    assert torch.equal(base, _base_logits(model, input_ids))


def test_median_residual_norm():
    model = _FakeModel()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    norms = median_residual_norm(model, input_ids, layers=[0])
    assert set(norms) == {0}
    assert norms[0] > 0


def test_contrastive_direction():
    high = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    low = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    direction = contrastive_direction(high, low)
    assert torch.allclose(direction, torch.tensor([2.0, 3.0]))
    unit = contrastive_direction(high, low, normalize=True)
    assert unit.norm() == pytest.approx(1.0)


def test_batch_noop_is_bit_exact():
    model = _FakeModel()
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    base = _base_logits(model, input_ids)
    with SteeringController(
        model,
        [SteeringSpec(kind="direction", layer=0, vector=torch.ones(4), dose=0.0)],
    ):
        zero = _base_logits(model, input_ids)
    assert torch.equal(base, zero)


def test_invalid_specs_rejected():
    with pytest.raises(ValueError, match="requires neuron"):
        SteeringSpec(kind="neuron_write", layer=0)
    with pytest.raises(ValueError, match="requires vector"):
        SteeringSpec(kind="direction", layer=0)
    with pytest.raises(ValueError, match="sign"):
        SteeringSpec(kind="direction", layer=0, vector=torch.ones(4), sign=2)
