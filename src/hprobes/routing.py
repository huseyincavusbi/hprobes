"""Routing-pattern analysis for MoE probes (issue #42).

Pure analysis over decision-token routing tables captured by
:class:`hprobes.moe.MoECapture` -- no additional forwards. Compares routing
of correct vs incorrect samples on selection frequencies, selected-set
overlap (Jaccard), and gate-mass entropy, each guarded by a permutation
null over labels. The counterfactual-routing machinery builds on the same
captured tables.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

__all__ = ["routing_stats"]


def _stack_tables(tables: Sequence[torch.Tensor], what: str) -> np.ndarray:
    arrs = [np.asarray(t.cpu()) if torch.is_tensor(t) else np.asarray(t) for t in tables]
    if not arrs:
        raise ValueError(f"{what} must be a non-empty sequence")
    shapes = {a.shape for a in arrs}
    if len(shapes) != 1:
        raise ValueError(f"{what} must share one shape, got {sorted(shapes)}")
    return np.stack(arrs)


def _indicators(routing_np: np.ndarray, layer: int, n_experts: int) -> np.ndarray:
    """(N, n_experts) 0/1 matrix: expert selected at ``layer`` for each sample."""
    idx = routing_np[:, layer, :]
    X = np.zeros((idx.shape[0], n_experts), dtype=np.float64)
    rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
    X[rows, idx.reshape(-1).astype(np.int64)] = 1.0
    return X


def _permutation_counts(
    X: np.ndarray, n_incorrect: int, rng: np.random.Generator, n_perm: int
) -> np.ndarray:
    """Null distribution of per-expert counts in a shuffled 'incorrect' group."""
    n = X.shape[0]
    order = np.stack([rng.permutation(n)[:n_incorrect] for _ in range(n_perm)])
    return X[order].sum(axis=1)  # (n_perm, n_experts)


def _delta_and_perm_p(
    X: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_perm: int
) -> tuple[np.ndarray, np.ndarray]:
    """Usage-frequency delta (incorrect - correct) with a two-sided permutation p."""
    n1 = int(y.sum())
    n0 = int(y.size - n1)
    if n1 == 0 or n0 == 0:
        return np.zeros(X.shape[1]), np.ones(X.shape[1])
    counts = X[y].sum(axis=0)
    totals = X.sum(axis=0)
    delta = counts / n1 - (totals - counts) / n0
    null_counts = _permutation_counts(X, n1, rng, n_perm)
    delta_null = null_counts / n1 - (totals[None, :] - null_counts) / n0
    p = (1 + (np.abs(delta_null) >= np.abs(delta)[None, :]).sum(axis=0)) / (1 + n_perm)
    return delta, p


def _mean_gate_entropy(gates_np: np.ndarray) -> np.ndarray:
    """Per-sample mean gate-mass entropy over layers: (N,)."""
    w = np.clip(gates_np.astype(np.float64), 1e-12, None)
    w = w / w.sum(axis=2, keepdims=True)
    h = -(w * np.log(w)).sum(axis=2)  # (N, n_layers)
    return h.mean(axis=1)


def _jaccard(a: set, b: set) -> float:
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _mean_pairwise_jaccard(sets: List[set]) -> Optional[float]:
    if len(sets) < 2:
        return None
    vals = [_jaccard(sets[i], sets[j]) for i in range(len(sets)) for j in range(i + 1, len(sets))]
    return float(np.mean(vals))


def routing_stats(
    routings: Sequence[torch.Tensor],
    labels: np.ndarray,
    gates: Optional[Sequence[torch.Tensor]] = None,
    n_experts: Optional[int] = None,
    n_perm: int = 200,
    seed: int = 0,
    top_flagged: int = 10,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Compare decision-token routing between correct and incorrect samples.

    Parameters
    ----------
    routings : per-sample ``(n_layers, k)`` int tensors of selected expert
        indices (``MoECapture.routing_table()`` at the decision position).
    labels : 1 = incorrect (hallucination), 0 = correct.
    gates : optional per-sample ``(n_layers, k)`` routing weights.
    n_experts : population size for the indicator space (default: max index + 1).
    n_perm : permutation draws for the nulls.

    Returns per-layer frequencies, usage deltas with two-sided permutation
    p-values and flagged experts, within/between-group Jaccard over
    ``(layer, expert)`` pairs, and gate-mass entropy comparison if gates
    were passed.
    """
    rng = np.random.default_rng(seed)
    routing_np = _stack_tables(routings, "routing tables")
    y = np.asarray(labels).astype(bool)
    if y.shape[0] != routing_np.shape[0]:
        raise ValueError("labels length must match the number of samples")
    n_samples, n_layers, k = routing_np.shape
    if n_experts is None:
        n_experts = int(routing_np.max()) + 1

    n1 = int(y.sum())
    n0 = n_samples - n1

    per_layer: List[Dict[str, Any]] = []
    for layer in range(n_layers):
        X = _indicators(routing_np, layer, n_experts)
        delta, p = _delta_and_perm_p(X, y, rng, n_perm)
        freq_incorrect = X[y].sum(axis=0) / n1 if n1 else np.zeros(n_experts)
        freq_correct = X[~y].sum(axis=0) / n0 if n0 else np.zeros(n_experts)
        order = np.argsort(-np.abs(delta))
        flagged = [
            {"expert": int(e), "delta": float(delta[e]), "p": float(p[e])}
            for e in order[:top_flagged]
            if p[e] < alpha
        ]
        per_layer.append(
            {
                "layer": layer,
                "freq_correct": freq_correct.tolist(),
                "freq_incorrect": freq_incorrect.tolist(),
                "usage_delta": delta.tolist(),
                "perm_p_abs_delta": p.tolist(),
                "top_flagged": flagged,
            }
        )

    sets = [
        {(layer, int(e)) for layer in range(n_layers) for e in routing_np[i, layer].tolist()}
        for i in range(n_samples)
    ]
    correct_sets = [s for s, is_inc in zip(sets, y) if not is_inc]
    incorrect_sets = [s for s, is_inc in zip(sets, y) if is_inc]
    jaccard: Dict[str, Optional[float]] = {
        "within_correct": _mean_pairwise_jaccard(correct_sets),
        "within_incorrect": _mean_pairwise_jaccard(incorrect_sets),
        "between": (
            float(np.mean([_jaccard(a, b) for a in correct_sets for b in incorrect_sets]))
            if correct_sets and incorrect_sets
            else None
        ),
    }

    gate_entropy: Optional[Dict[str, Any]] = None
    if gates is not None:
        gates_np = _stack_tables([g.float() for g in gates], "gate tables")
        if gates_np.shape[:2] != routing_np.shape[:2]:
            raise ValueError("gate tables must match routing table shapes")
        h = _mean_gate_entropy(gates_np)
        if n1 and n0:
            delta_h = float(h[y].mean() - h[~y].mean())
            null = np.empty(n_perm)
            for b in range(n_perm):
                perm = rng.permutation(n_samples)
                null[b] = h[perm[:n1]].mean() - h[perm[n1:]].mean()
            p_val = (1 + int((np.abs(null) >= abs(delta_h)).sum())) / (1 + n_perm)
        else:
            delta_h, p_val = 0.0, 1.0
        gate_entropy = {
            "correct": float(h[~y].mean()) if n0 else 0.0,
            "incorrect": float(h[y].mean()) if n1 else 0.0,
            "delta": delta_h,
            "perm_p": p_val,
        }

    return {
        "n_samples": int(n_samples),
        "n_correct": n0,
        "n_incorrect": n1,
        "n_layers": int(n_layers),
        "k": int(k),
        "n_experts": int(n_experts),
        "per_layer": per_layer,
        "jaccard": jaccard,
        "gate_entropy": gate_entropy,
    }
