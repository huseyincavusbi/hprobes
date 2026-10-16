"""Active-only per-expert CETT capture for MoE models (issue #41, phase 0).

Design
------
Routed experts contribute to a token's residual stream only when the router
selects them; unselected experts contribute exactly zero. So we capture
*active-only*: for every token and selected (expert, slot), the pre-down_proj
activation ``a`` (the same tensor dense CETT hooks), keyed by
``(layer, token, expert)``, plus the router gate weight ``g``.

Per-expert sketches accumulate only over tokens where the expert was active:
mean/variance of the raw activation (order-preserving Welford, so masked
passes reproduce full passes bit-for-bit) and mean |gate|. Contribution
magnitudes are ``|a * w_down|`` scaled by the gate as appropriate per
architecture; the sketch records the raw activation so downstream code can
form any of these without re-running forwards.

This module is deliberately model-family-agnostic: it works through the
"experts module receives (hidden_states, top_k_index, top_k_weights)" contract
shared by Olmoe/Qwen2Moe/Qwen3.5MoE/DeepSeek-style blocks in transformers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

# Router gate weights are applied inside the experts module by some families
# (Olmoe) and outside it by others (Qwen2Moe) -- the capture is identical
# either way; the distinction only matters when reconstructing contributions.
GATE_APPLIED_INSIDE = {"olmoe"}


@dataclass
class MoESketch:
    """Incremental per-(layer, expert) activation sketch.

    Pure-Python Welford with an ordered pass: seeded bootstrap slices of the
    captured row list reproduce a full pass bit-for-bit.
    """

    num_experts: int
    per_layer: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)

    def ensure_layer(self, layer: int) -> List[Dict[str, Any]]:
        if layer not in self.per_layer:
            self.per_layer[layer] = [
                {"n": 0, "mean": None, "M2": None, "gate_sum": 0.0} for _ in range(self.num_experts)
            ]
        return self.per_layer[layer]

    def update(
        self, layer: int, expert: int, activations: torch.Tensor, gates: torch.Tensor
    ) -> None:
        """Incorporate one (layer, expert) batch: activations (T, d), gates (T,)."""
        rows = self.ensure_layer(layer)
        slot = rows[expert]
        a = activations.detach().float().cpu()
        g = gates.detach().float().cpu()
        for t in range(a.shape[0]):
            vec = a[t]
            if slot["mean"] is None:
                slot["mean"] = torch.zeros_like(vec)
                slot["M2"] = torch.zeros_like(vec)
            slot["n"] += 1
            delta = vec - slot["mean"]
            slot["mean"] += delta / slot["n"]
            slot["M2"] += delta * (vec - slot["mean"])
            slot["gate_sum"] += float(g[t])

    def expert_stats(self, layer: int, expert: int) -> Dict[str, float]:
        slot = self.ensure_layer(layer)[expert]
        n = slot["n"]
        if n == 0:
            return {"n_active": 0, "mean_abs": 0.0, "var_mean": 0.0, "gate_mean": 0.0}
        var = slot["M2"] / max(n - 1, 1)
        return {
            "n_active": n,
            "mean_abs": float(slot["mean"].abs().mean()),
            "var_mean": float(var.mean()),
            "gate_mean": slot["gate_sum"] / n,
        }

    def summary(self) -> List[Dict[str, Any]]:
        out = []
        for layer in sorted(self.per_layer):
            for expert, _ in enumerate(self.per_layer[layer]):
                stats = self.expert_stats(layer, expert)
                if stats["n_active"] > 0:
                    out.append({"layer": layer, "expert": expert, **stats})
        return out


class MoECapture:
    """Capture active-only per-expert activations + router decisions.

    Install on the experts module of each MoE layer:

        cap = MoECapture(num_experts)
        handle = cap.attach(experts_module, layer_idx)
        ...
        handle.remove()

    ``col_norms`` (optional) maps expert -> ||w_down^e|| column norms for
    contribution magnitudes; pass the down_proj parameter rows.
    """

    def __init__(
        self,
        num_experts: int,
        layer: int,
        col_norms: Optional[Dict[int, torch.Tensor]] = None,
        top_k: Optional[int] = None,
    ):
        self.num_experts = num_experts
        self.layer = layer
        self.col_norms = col_norms
        self.top_k = top_k
        self.records: List[Dict[str, Any]] = []
        self.sketch = MoESketch(num_experts=num_experts)
        self._ctx: Dict[str, torch.Tensor] = {}

    def attach(self, experts_module: torch.nn.Module) -> torch.utils.hooks.RemovableHandle:
        """Wrap the experts module forward to capture per-expert activations.

        The family loop calls ``F.linear(act, down_proj[e])`` on *parameter
        slices*, so module hooks cannot see it; the supported contract is the
        loop's one-hot mapping from ``(top_k_index, top_k_weights)``. The
        wrapper replays that mapping: every expert receives exactly the rows
        the family's own one-hot would select, and we reuse the model's
        weights (gate_up_proj / down_proj parameters) to recompute them.
        Outputs are bit-identical to the family implementation.
        """
        import types

        cap = self
        original_forward = experts_module.forward

        def wrapped_forward(self_mod, hidden_states, top_k_index, top_k_weights):
            # Seed routing context: record_expert reads it per expert below.
            cap._ctx = {
                "top_k_index": top_k_index.detach(),
                "top_k_weights": top_k_weights.detach(),
                "token_norm": hidden_states.detach().float().norm(dim=-1).cpu(),
            }
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=cap.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            final_hidden_states = torch.zeros_like(hidden_states)
            for expert_idx in expert_hit:
                expert_idx = int(expert_idx[0])
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate, up = torch.nn.functional.linear(
                    current_state, self_mod.gate_up_proj[expert_idx]
                ).chunk(2, dim=-1)
                act = self_mod.act_fn(gate) * up
                cap.record_expert(expert_idx, act)
                out = torch.nn.functional.linear(act, self_mod.down_proj[expert_idx])
                out = out * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, out.to(final_hidden_states.dtype))
            return final_hidden_states

        experts_module.forward = types.MethodType(wrapped_forward, experts_module)
        self._original_forward = original_forward

        class _Handle:
            def remove(_self):
                experts_module.forward = original_forward

        return _Handle()

    def record_expert(self, expert: int, activations: torch.Tensor) -> None:
        """Called once per expert per layer-forward with its pre-down_proj activations."""
        ctx = self._ctx
        if not ctx:
            raise RuntimeError("record_expert called before routing context was captured")
        top_k_index = ctx["top_k_index"]
        top_k_weights = ctx["top_k_weights"]
        # token positions x slots where this expert is selected
        mask = top_k_index == expert
        token_idx, slot_idx = torch.where(mask)
        gates = top_k_weights[token_idx, slot_idx]
        if activations.shape[0] != token_idx.shape[0]:
            raise ValueError(
                f"activation rows {activations.shape[0]} != selected tokens {token_idx.shape[0]} "
                f"for expert {expert}"
            )
        self.sketch.update(self.layer, expert, activations, gates)
        norm = ctx["token_norm"].to(activations.device)[token_idx]
        self.records.append(
            {
                "layer": self.layer,
                "expert": expert,
                "tokens": token_idx.detach().cpu(),
                "slots": slot_idx.detach().cpu(),
                "gates": gates.detach().float().cpu(),
                "token_norm": norm.detach().float().cpu(),
                "acts": activations.detach().float().cpu(),
            }
        )

    def routing_table(self) -> torch.Tensor:
        """Per-token selected experts from the last forward: (T, top_k)."""
        return self._ctx["top_k_index"].detach().cpu()

    def gate_table(self) -> torch.Tensor:
        return self._ctx["top_k_weights"].detach().float().cpu()

    def summary(self) -> List[Dict[str, Any]]:
        """Per-(layer, expert) sketch statistics."""
        return self.sketch.summary()


class MoESparseDataset:
    """Two-stage feature selection and sparse row assembly for MoE probes.

    Stage 1: per-expert neuron candidates, ranked within each expert by the
    chosen score (variance or gate-weighted cumulative |activation|), floored
    by ``n_active >= min_active`` to drop statistically-starved experts.
    Stage 2: global top-k across all (layer, expert, neuron) candidates.
    Rows are then assembled sparse: a token contributes values only for its
    active experts' selected neurons; everything else is a structural zero.
    """

    def __init__(
        self,
        captures: List["MoECapture"],
        score: str = "mean_abs",
        per_expert_top_n: int = 64,
        top_k: int = 2000,
        min_active: int = 0,
        gate_weighted: bool = True,
        col_norms: Optional[Dict[Tuple[int, int], torch.Tensor]] = None,
    ):
        if score not in ("variance", "mean_abs"):
            raise ValueError(f"score must be 'variance' or 'mean_abs', got {score!r}")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if per_expert_top_n <= 0:
            raise ValueError("per_expert_top_n must be positive")
        self.captures = captures
        self.score = score
        self.per_expert_top_n = per_expert_top_n
        self.top_k = top_k
        self.min_active = min_active
        self.gate_weighted = gate_weighted
        self.col_norms = col_norms or {}
        self.features_: List[Dict[str, Any]] = []

    def _expert_score(self, mean_abs: float, variance: float) -> float:
        return variance if self.score == "variance" else mean_abs

    def resolve_features(self) -> List[Dict[str, Any]]:
        """Two-stage selection over all captures; returns the chosen features."""
        candidates: List[Dict[str, Any]] = []
        for cap in self.captures:
            layer = cap.layer
            for expert, slot in enumerate(cap.sketch.ensure_layer(layer)):
                n = slot["n"]
                if n == 0 or n < self.min_active or slot["mean"] is None:
                    continue
                mean = slot["mean"]
                var = slot["M2"] / max(n - 1, 1)
                mean_abs = mean.abs()
                if self.score == "variance":
                    ranking = var
                else:
                    ranking = mean_abs
                k = min(self.per_expert_top_n, ranking.numel())
                top_idx = torch.topk(ranking, k).indices.tolist()
                gate_mean = slot["gate_sum"] / n
                for neuron in top_idx:
                    candidates.append(
                        {
                            "layer": layer,
                            "expert": expert,
                            "neuron": neuron,
                            "expert_n_active": n,
                            "expert_mean_abs": float(mean_abs.mean()),
                            "expert_var_mean": float(var.mean()),
                            "gate_mean": gate_mean,
                            "score": float(ranking[neuron]),
                        }
                    )
        candidates.sort(key=lambda c: (-c["score"], c["layer"], c["expert"], c["neuron"]))
        self.features_ = candidates[: self.top_k]
        return self.features_

    def assemble_rows(self) -> Dict[str, Any]:
        """Sparse per-token rows over the resolved features (single forward pass).

        For every token, a feature fires only where its expert was selected;
        the value is ``gate x activation`` (gate_weighted) or the raw
        activation. Returns a CSR-style bundle: dense ``rows`` for tests,
        plus CSR arrays for scalable consumers.
        """
        if not self.features_:
            self.resolve_features()
        by_layer_expert: Dict[Tuple[int, int], List[int]] = {}
        for fi, feat in enumerate(self.features_):
            by_layer_expert.setdefault((feat["layer"], feat["expert"]), []).append(fi)

        # routing tables are per-capture (per layer); token count from the first.
        first = self.captures[0]
        n_tokens = int(first.routing_table().shape[0])
        rows = torch.zeros(n_tokens, len(self.features_))
        for cap in self.captures:
            for rec in cap.records:
                key = (cap.layer, rec["expert"])
                fids = by_layer_expert.get(key)
                if not fids:
                    continue
                for row_pos in range(rec["tokens"].shape[0]):
                    tok = int(rec["tokens"][row_pos])
                    if tok >= n_tokens:
                        continue
                    val_base = rec["acts"][row_pos].float()
                    if self.gate_weighted:
                        val_base = val_base * float(rec["gates"][row_pos])
                    for fi in fids:
                        neuron = self.features_[fi]["neuron"]
                        rows[tok, fi] = float(val_base[neuron])
        # CSR arrays for scalable consumers
        nz = (rows != 0).nonzero(as_tuple=False)
        csr = {
            "indptr": torch.zeros(n_tokens + 1, dtype=torch.long),
            "indices": nz[:, 1] if nz.numel() else torch.zeros(0, dtype=torch.long),
            "data": rows[nz[:, 0], nz[:, 1]] if nz.numel() else torch.zeros(0),
        }
        counts = torch.zeros(n_tokens + 1, dtype=torch.long)
        counts[1:] = (rows != 0).sum(dim=1)
        csr["indptr"] = torch.cumsum(counts, dim=0)
        token_norms = (
            torch.stack([self._token_norms_for(cap) for cap in self.captures], dim=1).sum(dim=1)
            if self.captures
            else torch.zeros(n_tokens)
        )
        return {"rows": rows, "csr": csr, "token_norms": token_norms}

    @staticmethod
    def _token_norms_for(cap: "MoECapture") -> torch.Tensor:
        norms = cap.__dict__.get("_token_norm")
        if norms is None:
            ctx = cap.__dict__.get("_ctx")
            norms = ctx.get("token_norm") if isinstance(ctx, dict) else None
        if norms is None:
            r = cap.routing_table()
            return torch.zeros(r.shape[0])
        return norms.float()
