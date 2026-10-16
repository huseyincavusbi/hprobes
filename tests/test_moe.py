"""Tests for active-only per-expert MoE capture (issue #41, phase 0)."""

import numpy as np
import torch
from torch import nn

from hprobes.moe import MoECapture, MoESketch


def _toy_experts(num_experts=4, d_in=8, d_inter=6):
    """Minimal module matching the family contract."""

    class ToyExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = num_experts
            self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * d_inter, d_in) * 0.1)
            self.down_proj = nn.Parameter(torch.randn(num_experts, d_in, d_inter) * 0.1)
            self.act_fn = nn.SiLU()

        def forward(self, hidden_states, top_k_index, top_k_weights):
            final = torch.zeros_like(hidden_states)
            for e in range(self.num_experts):
                mask = top_k_index == e
                token_idx, slot_idx = torch.where(mask)
                if token_idx.numel() == 0:
                    continue
                x = hidden_states[token_idx]
                gate, up = torch.nn.functional.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)
                act = self.act_fn(gate) * up
                out = torch.nn.functional.linear(act, self.down_proj[e])
                out = out * top_k_weights[token_idx, slot_idx, None]
                final.index_add_(0, token_idx, out)
            return final

    return ToyExperts()


class TestMoESketch:
    def test_active_only_stats(self):
        sketch = MoESketch(num_experts=3)
        a1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        sketch.update(0, 1, a1, torch.tensor([0.5, 0.5]))
        stats = sketch.expert_stats(0, 1)
        assert stats["n_active"] == 2
        assert stats["mean_abs"] == (1 + 2 + 3 + 4) / 4
        assert stats["gate_mean"] == 0.5
        untouch = sketch.expert_stats(0, 0)
        assert untouch["n_active"] == 0 and untouch["mean_abs"] == 0.0

    def test_insertion_order_determinism(self):
        """Same rows in the same order produce bit-identical sketch state.

        This is the property the masked bootstrap path relies on: rows are
        appended in a fixed order, so resuming/replaying yields the same stats.
        """
        rng = np.random.default_rng(0)
        rows = rng.normal(size=(7, 5)).astype(np.float32)
        gates = rng.random(7).astype(np.float32)
        one_pass = MoESketch(num_experts=1)
        one_pass.update(0, 0, torch.from_numpy(rows), torch.from_numpy(gates))
        replay = MoESketch(num_experts=1)
        replay.update(0, 0, torch.from_numpy(rows[:4]), torch.from_numpy(gates[:4]))
        replay.update(0, 0, torch.from_numpy(rows[4:]), torch.from_numpy(gates[4:]))
        f, p = one_pass.per_layer[0][0], replay.per_layer[0][0]
        assert f["n"] == p["n"] == 7
        assert torch.equal(f["mean"], p["mean"])
        assert torch.equal(f["M2"], p["M2"])
        assert f["gate_sum"] == p["gate_sum"]

    def test_variance_matches_numpy(self):
        rng = np.random.default_rng(1)
        rows = torch.from_numpy(rng.normal(size=(20, 4)).astype(np.float32))
        sketch = MoESketch(num_experts=1)
        sketch.update(0, 0, rows, torch.ones(20))
        stats = sketch.expert_stats(0, 0)
        assert stats["var_mean"] == float(rows.var(dim=0, unbiased=True).mean())


class TestMoECapture:
    def setup_method(self):
        torch.manual_seed(0)
        self.d_in, self.num_experts = 8, 4
        self.experts = _toy_experts(self.num_experts, self.d_in)
        self.hidden = torch.randn(6, self.d_in)
        k = 2
        self.top_k_index = torch.stack([torch.randperm(self.num_experts)[:k] for _ in range(6)])
        w = torch.rand(6, k)
        self.top_k_weights = w / w.sum(dim=-1, keepdim=True)

    def test_wrapped_forward_bit_identical_and_removable(self):
        base = self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        cap = MoECapture(num_experts=self.num_experts, layer=0)
        handle = cap.attach(self.experts)
        wrapped = self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        assert torch.equal(wrapped, base)
        handle.remove()
        again = self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        assert torch.equal(again, base)

    def test_records_match_routing_mask(self):
        cap = MoECapture(num_experts=self.num_experts, layer=3)
        handle = cap.attach(self.experts)
        self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        handle.remove()
        per_expert_tokens = {}
        for rec in cap.records:
            per_expert_tokens[rec["expert"]] = sorted(rec["tokens"].tolist())
        for e in range(self.num_experts):
            expected = sorted((self.top_k_index == e).nonzero(as_tuple=False)[:, 0].tolist())
            assert per_expert_tokens.get(e, []) == expected
        summary = cap.summary()
        total = sum(row["n_active"] for row in summary)
        assert total == self.top_k_index.numel()

    def test_routing_and_gate_tables(self):
        cap = MoECapture(num_experts=self.num_experts, layer=0)
        handle = cap.attach(self.experts)
        self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        handle.remove()
        assert torch.equal(cap.routing_table(), self.top_k_index)
        assert torch.allclose(cap.gate_table(), self.top_k_weights)

    def test_unknown_family_contract_raises(self):
        cap = MoECapture(num_experts=2, layer=0)

        def bad_forward(hidden_states):
            return hidden_states

        self.experts.forward = bad_forward
        cap.attach(self.experts)
        try:
            self.experts(self.hidden)
        except TypeError:
            return  # missing routing args: family contract not met
        raise AssertionError("expected TypeError for non-MoE contract")
