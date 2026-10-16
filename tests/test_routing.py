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
