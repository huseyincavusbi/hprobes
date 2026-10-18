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


class TestCounterfactualRouting:
    """#42 chunk 3: donor matching, doses, controls (five approved defaults)."""

    def _probe(self):
        import _fake_moe
        from test_probe import TOK

        from hprobes.probe import HProbes

        model = _fake_moe._FakeMoEModel()
        return HProbes(model, TOK, readout="logits")

    def _samples(self, n=8):
        # Answers alternate so both correct and incorrect samples exist under
        # the fake model's deterministic prediction (first letter id).
        return [{"question": f"Q{i}?", "options": ["aa", "bb"], "answer": i % 2} for i in range(n)]

    def test_helper_match_donors_opposite_labels(self):
        from hprobes.routing import match_donors

        labels = [1, 0, 1, 0]
        lengths = [10, 11, 12, 10]
        entropies = [0.5, 0.5, 0.6, 0.5]
        donors = match_donors(labels, lengths, entropies)
        assert all(labels[i] != labels[d] for i, d in donors.items())
        # For label-1 sample 0, nearest opposite is index 3 (length diff 0)
        assert donors[0] == 3

    def test_helper_dose_rows(self):
        from hprobes.routing import dose_rows

        own_e = [1, 2]
        own_w = [0.6, 0.4]
        donor_e = [3, 0]
        donor_w = [0.7, 0.3]
        e, w = dose_rows(own_e, own_w, donor_e, donor_w, dose=1)
        assert e == [3, 2] and w == [0.7, 0.4]
        e, w = dose_rows(own_e, own_w, donor_e, donor_w, dose=5)  # capped at k
        assert e == [3, 0] and w == [0.7, 0.3]

    def test_helper_frequency_sampled_experts_respects_usage(self):
        from hprobes.routing import frequency_sampled_experts

        usage = {0: 10.0, 1: 0.0, 2: 1.0, 3: 0.0}
        seen = set()
        for seed in range(20):
            picks = frequency_sampled_experts(usage, dose=1, k=2, seed=seed)
            seen.update(picks)
            # only experts with positive usage may ever be drawn
            assert all(p in (0, 2) for p in picks), picks
        # expert 0 has 10x the usage of expert 2: it must dominate but both appear
        assert 0 in seen and 2 in seen

    def test_self_swap_is_exact_noop(self):
        probe = self._probe()
        result = probe.counterfactual_routing(self._samples(), layers=[0], doses=[1])
        self_c = result["self_swap"]
        # Default 4: self-swap must be a bit-exact no-op on prediction + entropy.
        assert self_c["n"] > 0
        assert self_c["flips"] == 0
        assert self_c["max_abs_entropy_delta"] == 0.0

    def test_dose_result_shape_and_controls(self):
        probe = self._probe()
        result = probe.counterfactual_routing(self._samples(), layers=[0], doses=[1, 2, 5])
        assert result["donor_matches"]["n_targets"] > 0
        doses_seen = {c["dose"] for c in result["rescue"] + result["corruption"]}
        assert doses_seen == {1, 2}  # 5 capped to k=2, deduped
        for c in result["rescue"] + result["corruption"] + result["random"]:
            assert 0 <= c["flip_rate"] <= 1
            assert 0 <= c["accuracy_after"] <= 1
            assert c["n_targets"] > 0
        # random controls exist for both directions
        assert {"rescue", "corruption"} == {c["direction"] for c in result["random"]}

    def test_deterministic_given_seed(self):
        probe = self._probe()
        a = probe.counterfactual_routing(self._samples(), layers=[0], doses=[1], seed=0)
        b = probe.counterfactual_routing(self._samples(), layers=[0], doses=[1], seed=0)
        assert a["rescue"] == b["rescue"]
        assert a["random"] == b["random"]


class TestCettRoutingCrossCheck:
    """#42 requirement 5: do CETT-selected experts and routing agree?"""

    def _names(self):
        # CETT feature names from fit_moe: L{layer}:E{expert}:N{neuron}
        return ["L0:E0:N3", "L0:E1:N5", "L1:E2:N1"]

    def _stats(self, flagged_by_layer):
        return {
            "n_layers": 2,
            "per_layer": [
                {
                    "layer": li,
                    "top_flagged": [{"expert": e, "delta": 0.5, "p": 0.001} for e in experts],
                }
                for li, experts in flagged_by_layer.items()
            ],
        }

    def test_overlap_and_jaccard(self):
        from hprobes.routing import cett_routing_overlap

        stats = self._stats({0: [1, 7], 1: [2]})
        out = cett_routing_overlap(self._names(), stats)
        # CETT experts: (0,0),(0,1),(1,2); flagged: (0,1),(0,7),(1,2)
        assert out["n_cett_experts"] == 3
        assert out["n_routing_flagged"] == 3
        assert out["overlap"] == [[0, 1], [1, 2]]
        assert abs(out["jaccard"] - 2 / 4) < 1e-9
        assert out["per_layer"][0]["layer"] == 0

    def test_no_overlap(self):
        from hprobes.routing import cett_routing_overlap

        out = cett_routing_overlap(self._names(), self._stats({0: [9], 1: [8]}))
        assert out["overlap"] == []
        assert out["jaccard"] == 0.0

    def test_parses_only_wellformed_names(self):
        from hprobes.routing import cett_routing_overlap

        names = ["L0:E1:N0", "bogus", "L2:E0:N4"]
        out = cett_routing_overlap(names, self._stats({0: [1]}))
        assert out["n_cett_experts"] == 2
        assert out["overlap"] == [[0, 1]]

    def test_counts_shared_features_separately(self):
        from hprobes.routing import cett_routing_overlap

        names = ["L0:E1:N0", "L0:S:N2", "L1:S:N7"]
        out = cett_routing_overlap(names, self._stats({0: [1]}))
        assert out["n_cett_experts"] == 1
        assert out["n_shared_features"] == 2
        assert out["overlap"] == [[0, 1]]


class TestMoeRoutingAnalysisEndToEnd:
    """End-to-end on the fake MoE LM: stats + cross-check wiring."""

    def test_analysis_runs_and_cross_checks_after_fit(self):
        import _fake_moe
        from test_probe import TOK

        from hprobes.probe import HProbes

        model = _fake_moe._FakeMoEModel()
        probe = HProbes(model, TOK, readout="logits", l1_C=10)
        samples = [
            {"question": f"Q{i}?", "options": ["aa", f"x{i % 2}"], "answer": i % 2}
            for i in range(8)
        ]
        probe.fit_moe(samples, layers=[0], per_expert_top_n=4, top_k=16, min_active=1)
        stats = probe.moe_routing_analysis(samples, layers=[0], n_perm=50, seed=0)
        assert stats["n_samples"] == 8
        assert stats["n_layers"] == 1
        assert "cett_cross_check" in stats
        cc = stats["cett_cross_check"]
        # Wiring check: counts are ints and jaccard is a probability. Whether
        # the selected space is neurons or routing columns (both legal for L1)
        # is data-dependent; overlap correctness is unit-tested separately.
        assert isinstance(cc["n_cett_experts"], int)
        assert isinstance(cc["n_routing_flagged"], int)
        assert 0.0 <= cc["jaccard"] <= 1.0

    def test_counterfactual_method_deterministic_and_noop(self):
        import _fake_moe
        from test_probe import TOK

        from hprobes.probe import HProbes

        model = _fake_moe._FakeMoEModel()
        probe = HProbes(model, TOK, readout="logits")
        samples = [
            {"question": f"Q{i}?", "options": ["aa", "bb"], "answer": i % 2} for i in range(8)
        ]
        out = probe.counterfactual_routing(samples, layers=[0], doses=[1, 2], seed=0)
        assert out["self_swap"]["flips"] == 0
        assert out["self_swap"]["max_abs_entropy_delta"] == 0.0
        assert out["doses"] == [1, 2]
        assert len(out["rescue"]) == 2 and len(out["corruption"]) == 2
        assert len(out["random"]) == 4
