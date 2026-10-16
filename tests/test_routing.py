"""Tests for routing-pattern analysis (issue #42)."""

import numpy as np
import pytest
import torch

from hprobes.routing import routing_stats


def _planted(n=120, n_layers=2, k=2, n_experts=8, seed=0):
    """Expert 0 is over-selected on incorrect samples at layer 0."""
    rng = np.random.default_rng(seed)
    labels = np.array([i % 2 for i in range(n)])
    routings = []
    for i in range(n):
        r = rng.choice(n_experts, size=(n_layers, k), replace=False)
        if labels[i] == 1:
            r[0, 0] = 0
        routings.append(torch.as_tensor(r))
    return routings, labels


def test_usage_delta_flags_planted_expert():
    routings, labels = _planted()
    stats = routing_stats(routings, labels, n_experts=8, n_perm=200, seed=0)
    layer0 = stats["per_layer"][0]
    assert layer0["usage_delta"][0] > 0.5
    assert layer0["perm_p_abs_delta"][0] < 0.01
    flagged = [f["expert"] for f in layer0["top_flagged"]]
    assert 0 in flagged
    # layer 1 has no planted signal: expert 0's delta must stay small there
    # (significance at alpha=0.05 alone is not a valid negative control --
    # with 8 experts per layer a spurious flag is statistically expected)
    layer1_delta0 = stats["per_layer"][1]["usage_delta"][0]
    assert abs(layer1_delta0) < 0.5


def test_jaccard_within_and_between_groups():
    labels = np.array([0] * 10 + [1] * 10)
    routings = []
    for i in range(20):
        r = np.array([[0, 1], [0, 1]]) if labels[i] == 0 else np.array([[2, 3], [2, 3]])
        routings.append(torch.as_tensor(r))
    stats = routing_stats(routings, labels, n_experts=8, n_perm=50, seed=0)
    j = stats["jaccard"]
    assert j["within_correct"] == 1.0
    assert j["within_incorrect"] == 1.0
    assert j["between"] == 0.0


def test_gate_entropy_detects_peakedness():
    n = 60
    labels = np.array([i % 2 for i in range(n)])  # 1 = incorrect
    routings, gates = [], []
    for i in range(n):
        routings.append(torch.as_tensor(np.array([[0, 1], [2, 3]])))
        if labels[i] == 1:
            g = np.array([[0.99, 0.01], [0.99, 0.01]])
        else:
            g = np.array([[0.5, 0.5], [0.5, 0.5]])
        gates.append(torch.as_tensor(g, dtype=torch.float))
    stats = routing_stats(routings, labels, gates=gates, n_experts=6, n_perm=200, seed=0)
    ge = stats["gate_entropy"]
    assert ge["incorrect"] < ge["correct"]
    assert ge["perm_p"] < 0.01


def test_single_class_is_graceful():
    routings, labels = _planted(n=20)
    labels = np.zeros_like(labels)
    stats = routing_stats(routings, labels, n_experts=8, n_perm=50, seed=0)
    layer0 = stats["per_layer"][0]
    assert np.allclose(layer0["usage_delta"], 0.0, atol=1e-9)
    assert np.all(np.asarray(layer0["perm_p_abs_delta"]) == 1.0)
    assert stats["jaccard"]["between"] is None


def test_deterministic_for_fixed_seed():
    routings, labels = _planted()
    a = routing_stats(routings, labels, n_experts=8, n_perm=100, seed=7)
    b = routing_stats(routings, labels, n_experts=8, n_perm=100, seed=7)
    assert a["per_layer"][0]["perm_p_abs_delta"] == b["per_layer"][0]["perm_p_abs_delta"]


def test_shape_mismatch_raises():
    a = torch.zeros((2, 2), dtype=torch.long)
    b = torch.zeros((3, 2), dtype=torch.long)
    with pytest.raises(ValueError, match="share one shape"):
        routing_stats([a, b], np.array([0, 1]), n_experts=4)


def test_labels_length_mismatch_raises():
    routings, labels = _planted(n=10)
    with pytest.raises(ValueError, match="labels length"):
        routing_stats(routings, labels[:-1], n_experts=8)


class TestRoutingOverride:
    """#42: forced counterfactual routing at chosen positions."""

    def setup_method(self):
        torch.manual_seed(0)
        from test_moe import _toy_experts

        self.experts = _toy_experts(num_experts=4, d_in=8, d_inter=6)
        self.hidden = torch.randn(6, 8)
        self.top_k_index = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3], [0, 3], [2, 1]])
        w = torch.rand(6, 2)
        self.top_k_weights = w / w.sum(dim=-1, keepdim=True)

    def _capture(self):
        from hprobes.moe import MoECapture

        cap = MoECapture(num_experts=4, layer=0)
        handle = cap.attach(self.experts)
        return cap, handle

    def test_self_override_is_bit_identical(self):
        base = self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        cap, handle = self._capture()
        cap.set_routing_override(
            positions=[1, 3],
            experts=self.top_k_index[[1, 3]],
            weights=self.top_k_weights[[1, 3]],
        )
        again = self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        handle.remove()
        assert torch.equal(base, again)

    def test_forced_override_changes_effective_routing_and_output(self):
        base = self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        cap, handle = self._capture()
        cap.set_routing_override(
            positions=[0, 5],
            experts=torch.tensor([[3, 3], [3, 3]]),
            weights=torch.full((2, 2), 0.5),
        )
        out = self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        handle.remove()
        assert not torch.equal(out, base)
        rt = cap.routing_table()
        assert rt[0].tolist() == [3, 3]
        assert rt[5].tolist() == [3, 3]
        # records (expert -> tokens actually computed) reflect the override
        used = {}
        for rec in cap.records:
            used.setdefault(rec["expert"], set()).update(rec["tokens"].tolist())
        assert 0 in used[3] and 5 in used[3]
        assert 0 not in used[0]  # token 0 was re-routed away from expert 0

    def test_override_scoped_to_positions(self):
        cap, handle = self._capture()
        cap.set_routing_override(
            positions=[2], experts=torch.tensor([[3, 3]]), weights=torch.full((1, 2), 0.5)
        )
        self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        handle.remove()
        used = {}
        for rec in cap.records:
            used.setdefault(rec["expert"], set()).update(int(t) for t in rec["tokens"])
        # token 2 was re-routed to expert 3 and away from experts 0/2;
        # other tokens keep the experts their own routing selected
        # (tokens 3, 4 also use expert 3 via their original routing)
        assert 2 in used[3]
        assert 2 not in used[0]
        assert 2 not in used[2]
        assert 0 in used[0] and 4 in used[0]

    def test_clear_override_restores_original_behavior(self):
        base = self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        cap, handle = self._capture()
        cap.set_routing_override(
            positions=[0], experts=torch.tensor([[3, 3]]), weights=torch.full((1, 2), 0.5)
        )
        self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        cap.clear_routing_override()
        restored = self.experts(self.hidden, self.top_k_index, self.top_k_weights)
        handle.remove()
        assert torch.equal(restored, base)

    def test_override_validation(self):
        cap, handle = self._capture()
        with pytest.raises(ValueError, match="n_positions"):
            cap.set_routing_override(positions=[0, 1], experts=torch.tensor([[1, 2]]))
        with pytest.raises(ValueError, match="shape"):
            cap.set_routing_override(
                positions=[0],
                experts=torch.tensor([[1, 2]]),
                weights=torch.ones(1, 3),
            )
        handle.remove()
