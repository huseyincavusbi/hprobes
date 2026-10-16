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


class _Recorder:
    """Lightweight capture double: no model, direct records."""

    def __init__(self, layer, num_experts, n_tokens):
        self.layer = layer
        self.num_experts = num_experts
        self.records = []
        self._n_tokens = n_tokens
        self._routing = None
        self._gates = None
        self.sketch = MoESketch(num_experts=num_experts)
        self._token_norm = None

    def routing_table(self):
        return self._routing

    def gate_table(self):
        return self._gates

    def set_routing(self, routing, gates):
        self._routing = routing
        self._gates = gates

    def add_record(self, expert, tokens, gates, acts):
        acts_t = torch.as_tensor(acts, dtype=torch.float)
        gates_t = torch.tensor(gates, dtype=torch.float) if not torch.is_tensor(gates) else gates
        self.sketch.update(self.layer, expert, acts_t, gates_t)
        self.records.append(
            {
                "layer": self.layer,
                "expert": expert,
                "tokens": torch.tensor(tokens),
                "slots": torch.zeros(len(tokens), dtype=torch.long),
                "gates": gates_t,
                "token_norm": torch.ones(len(tokens)),
                "acts": acts_t,
            }
        )

    def _ctx(self):
        return {}


class TestMoESparseDataset:
    def _setup(self, score="mean_abs"):
        cap = _Recorder(layer=0, num_experts=2, n_tokens=4)
        cap.set_routing(
            torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]]),
            torch.tensor([[0.6, 0.4], [0.6, 0.4], [0.7, 0.3], [0.5, 0.5]]),
        )
        # expert 0 active on tokens [0,1,3] (slot 0) and [2] (slot 1)
        cap.add_record(0, [0, 1, 3], [0.6, 0.6, 0.5], [[2.0, 0.0], [4.0, 1.0], [6.0, 2.0]])
        cap.add_record(1, [0, 1, 2], [0.4, 0.4, 0.3], [[1.0, 1.0], [3.0, 2.0], [5.0, 4.0]])
        return cap

    def test_selection_scores_and_budget(self):
        from hprobes.moe import MoESparseDataset

        cap = self._setup()
        ds = MoESparseDataset(captures=[cap], per_expert_top_n=2, top_k=4)
        feats = ds.resolve_features()
        assert len(feats) == 4
        # stage-1 keeps only top-2 neurons per expert
        assert sum(1 for f in feats if f["expert"] == 0) == 2

    def test_min_active_floor_drops_rare_experts(self):
        from hprobes.moe import MoESparseDataset

        cap = _Recorder(layer=0, num_experts=2, n_tokens=3)
        cap.set_routing(torch.tensor([[0], [0], [1]]), torch.ones(3, 1))
        cap.add_record(0, [0, 1], [1.0, 1.0], [[1.0], [2.0]])
        cap.add_record(1, [2], [1.0], [[9.0]])
        ds = MoESparseDataset(captures=[cap], per_expert_top_n=4, top_k=10, min_active=2)
        feats = ds.resolve_features()
        assert all(f["expert"] == 0 for f in feats)

    def test_gate_weighted_values_and_structural_zeros(self):
        from hprobes.moe import MoESparseDataset

        cap = self._setup()
        ds = MoESparseDataset(captures=[cap], per_expert_top_n=1, top_k=10, gate_weighted=True)
        ds.resolve_features()
        out = ds.assemble_rows()
        rows = out["rows"]
        assert rows.shape == (4, len(ds.features_))
        # expert-0 top neuron; token 0 value = gate 0.6 * activation of that neuron
        f0 = next(f for f in ds.features_ if f["expert"] == 0)
        expected = 0.6 * float(
            torch.as_tensor([[2.0, 0.0], [4.0, 1.0], [6.0, 2.0]])[0, f0["neuron"]]
        )
        fi = ds.features_.index(f0)
        assert abs(float(rows[0, fi]) - expected) < 1e-6
        # expert-1 features are zero on token 3 (expert 1 routed only to 0,1,2)
        f1s = [i for i, f in enumerate(ds.features_) if f["expert"] == 1]
        assert all(float(rows[3, i]) == 0.0 for i in f1s)

    def test_contribution_normalization_uses_total_write(self):
        from hprobes.moe import MoESparseDataset

        cap = _Recorder(layer=0, num_experts=1, n_tokens=2)
        cap.set_routing(torch.tensor([[0], [0]]), torch.ones(2, 1))
        cap.add_record(0, [0, 1], [1.0, 1.0], [[1.0], [3.0]])
        cap._token_norm = torch.tensor([2.0, 6.0])  # explicit override
        ds = MoESparseDataset(captures=[cap], per_expert_top_n=1, top_k=1)
        ds.resolve_features()
        out = ds.assemble_rows()
        norms = out["token_norms"]
        assert float(norms[0]) == 2.0
        assert float(norms[1]) == 6.0
