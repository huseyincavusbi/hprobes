"""Tests for hprobes.cett — CETT metric and hook utilities."""

import pytest
import torch
import torch.nn as nn

from hprobes.cett import (
    _batched_token_rows,
    _get_transformer_layers,
    _sequence_rows,
    _token_rows,
    available_layers,
    forward_cett,
    forward_cett_span,
    get_mlp_down_proj,
    precompute_col_norms,
    scale_h_neurons,
)

# ---------------------------------------------------------------------------
# Minimal mock models
# ---------------------------------------------------------------------------

_H, _I, _V, _L = 8, 16, 32, 4


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(_H, _I, bias=False)
        self.down_proj = nn.Linear(_I, _H, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate(x)))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()

    def forward(self, x):
        return x + self.mlp(x)


class _CausalLM(nn.Module):
    def __init__(self, n=_L):
        super().__init__()
        torch.manual_seed(0)

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Block() for _ in range(n)])

        self.model = _Inner()
        self.lm_head = nn.Linear(_H, _V, bias=True)

    def forward(self, input_ids=None, **kw):
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, _H)
        for block in self.model.layers:
            x = block(x)

        class _Out:
            pass

        out = _Out()
        out.logits = self.lm_head(x)
        return out


class _MultimodalLM(nn.Module):
    """Mimics MedGemma-4B: model.model.language_model.layers."""

    def __init__(self):
        super().__init__()

        class _Lang(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Block() for _ in range(2)])

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.language_model = _Lang()

        self.model = _Inner()


M = _CausalLM()


class _SharedExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(_I, _H, bias=False)


class _MoEMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(_H, 2, bias=False)
        self.experts = nn.ModuleList([_MLP() for _ in range(2)])
        self.shared_expert = _SharedExpert()


class _MoEBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MoEMLP()


class _MoECausalLM(nn.Module):
    """Mimics Qwen3.5/3.8 MoE: mlp has experts + shared_expert, no down_proj."""

    def __init__(self, n=2):
        super().__init__()

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_MoEBlock() for _ in range(n)])

        self.model = _Inner()


def _tok(text="ABCDE"):
    ids = torch.tensor([[ord(c) for c in text]], dtype=torch.long)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


# ---------------------------------------------------------------------------


class TestArchitectureDetection:
    def test_standard_causal_lm(self):
        assert len(_get_transformer_layers(M)) == _L

    def test_multimodal_wrapper(self):
        assert len(_get_transformer_layers(_MultimodalLM())) == 2

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            _get_transformer_layers(nn.Linear(4, 4))

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            get_mlp_down_proj(M, 999)


class TestMoESharedExpert:
    def test_shared_expert_down_proj_found(self):
        model = _MoECausalLM()
        for layer_idx in range(2):
            down = get_mlp_down_proj(model, layer_idx)
            assert down is model.model.layers[layer_idx].mlp.shared_expert.down_proj

    def test_shared_expert_col_norms(self):
        model = _MoECausalLM()
        norms = precompute_col_norms(model, [0, 1])
        assert set(norms.keys()) == {0, 1}
        assert norms[0].shape == (_I,)


class TestFlattenedActivations:
    def test_token_rows_3d(self):
        t = torch.arange(2 * 3 * 4).float().reshape(2, 3, 4)
        assert torch.equal(_token_rows(t, -1), t[0, -1, :])

    def test_token_rows_2d(self):
        t = torch.arange(3 * 4).float().reshape(3, 4)
        assert torch.equal(_token_rows(t, -1), t[-1, :])

    def test_sequence_rows_2d(self):
        t = torch.arange(3 * 4).float().reshape(3, 4)
        assert torch.equal(_sequence_rows(t), t)

    def test_batched_token_rows_2d(self):
        seq_len, dim = 3, 4
        t = torch.arange(2 * seq_len * dim).float().reshape(2 * seq_len, dim)
        batch_idx = torch.tensor([0, 1])
        token_pos = torch.tensor([2, 0])
        rows = _batched_token_rows(t, batch_idx, token_pos, seq_len)
        assert torch.equal(rows[0], t[2])
        assert torch.equal(rows[1], t[seq_len])


class TestColNorms:
    def test_shape_and_sign(self):
        norms = precompute_col_norms(M, available_layers(M))
        for li, v in norms.items():
            assert v.shape == (_I,)
            assert (v >= 0).all()


class TestForwardCett:
    def setup_method(self):
        self.layers = available_layers(M)
        self.norms = precompute_col_norms(M, self.layers)

    def test_output_shapes(self):
        cett, logits = forward_cett(M, _tok(), self.layers, self.norms)
        assert cett.shape == (len(self.layers) * _I,)
        assert (cett >= 0).all()  # Verify abs(z) fix
        assert logits.shape == (_V,)

    def test_last_token_equals_explicit(self):
        toks = _tok("ABC")
        seq = toks["input_ids"].shape[1]
        c1, _ = forward_cett(M, toks, self.layers, self.norms, token_position=-1)
        c2, _ = forward_cett(M, toks, self.layers, self.norms, token_position=seq - 1)
        assert torch.allclose(c1, c2)


class TestForwardCettSpan:
    def setup_method(self):
        self.layers = available_layers(M)
        self.norms = precompute_col_norms(M, self.layers)

    def test_output_shape(self):
        result = forward_cett_span(M, _tok("ABCDE"), 1, 3, self.layers, self.norms)
        assert result.shape == (len(self.layers) * _I,)

    def test_max_gte_mean(self):
        toks = _tok("ABCDE")
        mean = forward_cett_span(M, toks, 1, 4, self.layers, self.norms, "mean")
        mx = forward_cett_span(M, toks, 1, 4, self.layers, self.norms, "max")
        assert (mx >= mean - 1e-5).all()


class TestScaleHNeurons:
    def setup_method(self):
        self.layers = available_layers(M)
        self.norms = precompute_col_norms(M, self.layers)

    def test_alpha_one_is_identity(self):
        toks = _tok("XY")
        _, baseline = forward_cett(M, toks, self.layers, self.norms)
        scaled = scale_h_neurons(M, toks, [(0, 1), (2, 5)], 1.0, self.layers)
        assert torch.allclose(baseline, scaled, atol=1e-5)

    def test_alpha_zero_suppresses(self):
        toks = _tok("XY")
        # Use all neurons in layer 0 to guarantee at least one fires
        neurons = [(0, i) for i in range(_I)]
        l1 = scale_h_neurons(M, toks, neurons, 1.0, self.layers)
        l0 = scale_h_neurons(M, toks, neurons, 0.0, self.layers)
        assert not torch.allclose(l1, l0)
