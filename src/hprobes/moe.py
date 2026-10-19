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

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch

# Router gate weights are applied inside the experts module by some families
# (Olmoe) and outside it by others (Qwen2Moe) -- the capture is identical
# either way; the distinction only matters when reconstructing contributions.
GATE_APPLIED_INSIDE = {"olmoe"}

# Neuron ranking modes for MoE feature selection (#41).
#   variance     -- per-neuron variance over active rows
#   mean_abs     -- mean|x| over active rows (historical default; == conditional)
#   conditional  -- alias of mean_abs, kept for issue #41.7 vocabulary
#   expected     -- sum|x| / n_tokens: expected contribution with structural zeros
_MOE_SCORES = ("variance", "mean_abs", "conditional", "expected")


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
        # Routing-aware intervention state (#41 causal): when set, the
        # wrapped forward rescales the target expert's activation rows --
        # only where the expert was actually routed -- before down_proj.
        self._intervention: Optional[Dict[str, Any]] = None
        # Counterfactual routing state (#42): when set, the wrapped forward
        # replaces the router's decisions (experts + gates) at the given
        # token positions before anything consumes them.
        self._routing_override: Optional[Dict[str, Any]] = None

    def attach(
        self,
        experts_module: torch.nn.Module,
        allow_eager_switch: bool = False,
    ) -> torch.utils.hooks.RemovableHandle:
        """Wrap the experts module forward to capture per-expert activations.

        The family loop calls ``F.linear(act, down_proj[e])`` on *parameter
        slices*, so module hooks cannot see it; the supported contract is the
        loop's one-hot mapping from ``(top_k_index, top_k_weights)``. The
        wrapper replays that mapping: every expert receives exactly the rows
        the family's own one-hot would select, and we reuse the model's
        weights (gate_up_proj / down_proj parameters) to recompute them.
        Outputs are bit-identical to the family implementation.

        Requires the model to run the ``eager`` experts implementation:
        transformers dispatches ``batched_mm``/``grouped_mm`` fast experts
        paths for loaded models, and the eager replay would then diverge from
        the model's actual forward. Load with
        ``experts_implementation="eager"``, or pass
        ``allow_eager_switch=True`` to switch a loaded model's config in
        place (a warning is emitted; every subsequent forward of the model
        then runs the eager loop).

        CPU/disk-offloaded experts (accelerate ``device_map`` offload) are
        supported when the module carries accelerate's device hook: the
        wrapper materializes the weights through ``pre_forward`` for the
        replay and offloads them again via ``post_forward``, mirroring
        accelerate's own forward wrapper. Meta-device parameters without a
        usable offload hook are rejected loudly instead of silently producing
        NaN features.
        """
        config = getattr(experts_module, "config", None)
        impls = {
            getattr(config, "_experts_implementation", None),
            getattr(config, "_experts_implementation_internal", None),
        }
        dispatched = sorted(i for i in impls if i not in (None, "eager"))
        if dispatched:
            if not allow_eager_switch:
                raise ValueError(
                    f"experts module dispatches {dispatched[0]!r} from transformers' experts "
                    "interface, but MoECapture replays the 'eager' implementation and would "
                    "not be bit-identical to the model. Load the model with "
                    "`experts_implementation='eager'`, or pass `allow_eager_switch=True` to "
                    "switch this model in place (affects all subsequent forwards in this "
                    "process)."
                )
            warnings.warn(
                f"switching experts implementation from {dispatched[0]!r} to 'eager' in "
                "place; all subsequent forwards of this model now run the eager experts "
                "loop.",
                stacklevel=2,
            )
            for field_name in ("_experts_implementation", "_experts_implementation_internal"):
                if hasattr(config, field_name):
                    setattr(config, field_name, "eager")

        # CPU/disk offload (#68): accelerate keeps offloaded weights in a map
        # and leaves meta tensors on the module, materializing them only inside
        # its own forward wrapper. We replace that wrapper, so detect the hook
        # up front and compose with it in the replay; refuse loudly when the
        # weights cannot be materialized (silent NaNs otherwise).
        offload_hook = getattr(experts_module, "_hf_hook", None)
        hook_is_usable = bool(
            offload_hook is not None
            and callable(getattr(offload_hook, "pre_forward", None))
            and callable(getattr(offload_hook, "post_forward", None))
        )
        meta_params = [
            name for name, param in experts_module.named_parameters() if param.device.type == "meta"
        ]
        if meta_params and not (
            hook_is_usable
            and getattr(offload_hook, "offload", False)
            and getattr(offload_hook, "weights_map", None) is not None
        ):
            raise ValueError(
                "MoECapture cannot replay offloaded experts: parameters "
                f"{meta_params} are on the meta device and the module has no usable "
                "accelerate offload hook. Keep the MoE layers resident on GPU(s) "
                "(a device_map without CPU/disk offload) or load them through "
                "accelerate dispatch_model so offloading has a weights map."
            )

        import types

        cap = self
        original_forward = experts_module.forward

        def wrapped_forward(self_mod, hidden_states, top_k_index, top_k_weights):
            # accelerate offload (#68): its forward wrapper (which we replaced)
            # would have materialized the offloaded weights here; do the same
            # so the replay sees real tensors, and offload again afterwards.
            if hook_is_usable:
                args, _ = offload_hook.pre_forward(
                    self_mod, hidden_states, top_k_index, top_k_weights
                )
                hidden_states, top_k_index, top_k_weights = args
            try:
                # Counterfactual routing (#42): swap the router's decisions at
                # the requested positions before any consumer sees them.
                ov = cap._routing_override
                if ov is not None:
                    top_k_index = top_k_index.clone()
                    top_k_weights = top_k_weights.clone()
                    pos = ov["positions"].to(top_k_index.device)
                    top_k_index[pos] = ov["experts"].to(
                        device=top_k_index.device, dtype=top_k_index.dtype
                    )
                    if ov["weights"] is not None:
                        top_k_weights[pos] = ov["weights"].to(
                            device=top_k_weights.device, dtype=top_k_weights.dtype
                        )
                # Seed routing context: record_expert reads it per expert below.
                cap._ctx = {
                    "top_k_index": top_k_index.detach(),
                    "top_k_weights": top_k_weights.detach(),
                    "token_norm": hidden_states.detach().float().norm(dim=-1).cpu(),
                }
                with torch.no_grad():
                    # num_experts + 1 mirrors the family one-hot: the extra class
                    # is a sentinel for dropped slots, skipped in the loop below.
                    expert_mask = torch.nn.functional.one_hot(
                        top_k_index, num_classes=cap.num_experts + 1
                    )
                    expert_mask = expert_mask.permute(2, 1, 0)
                    expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
                final_hidden_states = torch.zeros_like(hidden_states)
                for expert_idx in expert_hit:
                    expert_idx = int(expert_idx[0])
                    if expert_idx == cap.num_experts:
                        continue
                    top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                    current_state = hidden_states[token_idx]
                    gate, up = torch.nn.functional.linear(
                        current_state, self_mod.gate_up_proj[expert_idx]
                    ).chunk(2, dim=-1)
                    act = self_mod.act_fn(gate) * up
                    iv = cap._intervention
                    if iv is not None and expert_idx == iv.get("expert"):
                        names = iv["features"]
                        if iv["ablate"]:
                            act = act.clone()
                            for j in names:
                                act[:, j] = 0.0
                        else:
                            alpha = float(iv["alpha"])
                            act = act.clone()
                            for j in names:
                                act[:, j] = act[:, j] * alpha
                    cap.record_expert(expert_idx, act, top_k_pos=top_k_pos, token_idx=token_idx)
                    out = torch.nn.functional.linear(act, self_mod.down_proj[expert_idx])
                    out = out * top_k_weights[token_idx, top_k_pos, None]
                    final_hidden_states.index_add_(0, token_idx, out.to(final_hidden_states.dtype))
            except Exception:
                if hook_is_usable:
                    try:
                        offload_hook.post_forward(self_mod, None)
                    except Exception:
                        pass
                raise
            if hook_is_usable:
                return offload_hook.post_forward(self_mod, final_hidden_states)
            return final_hidden_states

        experts_module.forward = types.MethodType(wrapped_forward, experts_module)
        self._original_forward = original_forward

        class _Handle:
            def remove(_self):
                experts_module.forward = original_forward

        return _Handle()

    def record_expert(
        self,
        expert: int,
        activations: torch.Tensor,
        top_k_pos: Optional[torch.Tensor] = None,
        token_idx: Optional[torch.Tensor] = None,
    ) -> None:
        """Called once per expert per layer-forward with its pre-down_proj activations.

        ``top_k_pos`` and ``token_idx`` are the (slot, token) routing rows for
        this expert in the family loop's row order; the wrapping loop passes
        its own indices so recorded rows pair 1:1 with ``activations``.
        External callers may omit them: they are then re-derived in the same
        slot-major order of the loop's ``(k, T)`` expert mask.
        """
        ctx = self._ctx
        if not ctx:
            raise RuntimeError("record_expert called before routing context was captured")
        top_k_index = ctx["top_k_index"]
        top_k_weights = ctx["top_k_weights"]
        if top_k_pos is None or token_idx is None:
            # row-major over the (k, T) expert mask: same order as the family
            # loop's torch.where(expert_mask[expert_idx]) row replay.
            top_k_pos, token_idx = torch.where((top_k_index == expert).t())
        gates = top_k_weights[token_idx, top_k_pos]
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
                "slots": top_k_pos.detach().cpu(),
                "gates": gates.detach().float().cpu(),
                "token_norm": norm.detach().float().cpu(),
                "acts": activations.detach().float().cpu(),
            }
        )

    def set_intervention(
        self,
        expert: int,
        features: List[int],
        alpha: Optional[float] = None,
        ablate: bool = False,
    ) -> None:
        """Routing-aware intervention on one expert's neurons (#41 causal).

        With ``ablate=True`` the listed neurons are zeroed; with ``alpha``
        given they are scaled by it. Applies only inside this expert's routed
        rows, i.e. exclusively at tokens where the router selected the expert
        -- the structural-zeros requirement of the issue.
        """
        if not ablate and alpha is None:
            raise ValueError("pass ablate=True or an alpha scale")
        if not features:
            raise ValueError("features must be a non-empty list of neuron indices")
        self._intervention = {
            "expert": int(expert),
            "features": [int(j) for j in features],
            "alpha": alpha,
            "ablate": bool(ablate),
        }

    def clear_intervention(self) -> None:
        self._intervention = None

    def set_routing_override(
        self,
        positions: List[int],
        experts: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Force router decisions at chosen token positions (#42).

        ``experts`` is ``(n_positions, k)`` forced expert indices (same ``k``
        as the model's top-k); ``weights`` optionally overrides the gates for
        those slots (default: keep the sample's own gates). Positions index
        the flattened token dimension the experts module receives. Effective
        routing (overridden values) is what ``routing_table()`` reports.
        """
        pos = torch.as_tensor(positions, dtype=torch.long)
        exp = torch.as_tensor(experts, dtype=torch.long)
        if exp.dim() != 2 or exp.shape[0] != pos.numel():
            raise ValueError(
                "experts must be (n_positions, k); got "
                f"{tuple(exp.shape)} for {pos.numel()} positions"
            )
        w = None
        if weights is not None:
            w = torch.as_tensor(weights, dtype=torch.float)
            if w.shape != exp.shape:
                raise ValueError(
                    f"weights shape {tuple(w.shape)} must match experts {tuple(exp.shape)}"
                )
        self._routing_override = {"positions": pos, "experts": exp, "weights": w}

    def clear_routing_override(self) -> None:
        self._routing_override = None

    def clear_records(self) -> None:
        """Drop recorded per-forward rows.

        Recorded activations are only valid for the forward that produced
        them; callers running multiple forwards (per-sample loops,
        counterfactual conditions) must clear between forwards so features
        never leak across samples.
        """
        self.records.clear()

    def routing_table(self) -> torch.Tensor:
        """Per-token selected experts from the last forward: (T, top_k)."""
        return self._ctx["top_k_index"].detach().cpu()

    def gate_table(self) -> torch.Tensor:
        return self._ctx["top_k_weights"].detach().float().cpu()

    def summary(self) -> List[Dict[str, Any]]:
        """Per-(layer, expert) sketch statistics."""
        return self.sketch.summary()


class SharedExpertCapture:
    """Read-only capture of an always-on (shared) expert's activation.

    MoE blocks with a dense shared expert (Qwen2Moe / Qwen3.5-style) apply it
    to every token, so unlike routed experts there is no routing decision to
    replay: the module's own forward runs untouched and we only observe it.
    Hooks ``mlp.shared_expert.down_proj`` (input = pre-down_proj activation)
    and, when present, ``mlp.shared_expert_gate`` (output -> sigmoid gate).
    """

    def __init__(self, mlp: torch.nn.Module, layer: int):
        self.layer = layer
        self._down = getattr(getattr(mlp, "shared_expert", None), "down_proj", None)
        self._gate_mod = getattr(mlp, "shared_expert_gate", None)
        self.acts: Optional[torch.Tensor] = None
        self.gates: Optional[torch.Tensor] = None
        self._handles: List[Any] = []

    @property
    def available(self) -> bool:
        return self._down is not None

    def attach(self) -> "SharedExpertCapture":
        def hook_down(module, inputs, output):
            x = inputs[0].detach()
            self.acts = x.reshape(-1, x.shape[-1]).float()

        self._handles.append(self._down.register_forward_hook(hook_down))
        if self._gate_mod is not None:

            def hook_gate(module, inputs, output):
                self.gates = torch.sigmoid(output.detach().float().reshape(-1))

            self._handles.append(self._gate_mod.register_forward_hook(hook_gate))
        return self

    def clear(self) -> None:
        self.acts = None
        self.gates = None

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def decision_feature(self, pos: int) -> Optional[Tuple[List[int], torch.Tensor, torch.Tensor]]:
        """(tokens, acts, gates) for one token position, or None."""
        if self.acts is None or pos >= self.acts.shape[0]:
            return None
        acts = self.acts[pos : pos + 1].cpu()
        if self.gates is not None and pos < self.gates.shape[0]:
            gates = self.gates[pos : pos + 1].cpu()
        else:
            gates = torch.ones(1)
        return [pos], acts, gates


def moe_col_norms(model: torch.nn.Module, layers: List[int]) -> Dict[Tuple[int, int], torch.Tensor]:
    """Per-expert column norms ||w_down^(l,e)[:, j]|| for routed experts.

    Mirrors ``cett.precompute_col_norms`` for the stacked-expert layout:
    down_proj has shape (num_experts, d_model, intermediate).
    """
    from .cett import _get_transformer_layers

    blocks = _get_transformer_layers(model)
    out: Dict[Tuple[int, int], torch.Tensor] = {}
    for layer_idx in layers:
        experts = blocks[layer_idx].mlp.experts
        W = experts.down_proj.detach().float()
        if W.device.type == "meta":
            raise ValueError("meta-device experts not supported for col norms")
        for e in range(W.shape[0]):
            out[(layer_idx, e)] = torch.norm(W[e], dim=0).cpu()
    return out


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
        if score not in _MOE_SCORES:
            raise ValueError(f"score must be one of {_MOE_SCORES}, got {score!r}")
        if top_k < 0:
            raise ValueError("top_k must be non-negative (0 = all features)")
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
            n_tokens = max(int(cap.routing_table().shape[0]), 1)
            for expert, slot in enumerate(cap.sketch.ensure_layer(layer)):
                n = slot["n"]
                if n == 0 or n < self.min_active or slot["mean"] is None:
                    continue
                mean = slot["mean"]
                var = slot["M2"] / max(n - 1, 1)
                mean_abs = mean.abs()
                if self.score == "variance":
                    ranking = var
                elif self.score == "expected":
                    # expected contribution over ALL tokens (structural zeros count)
                    ranking = mean_abs * (n / n_tokens)
                else:
                    # mean_abs (historical default) and conditional: mean|x| over
                    # the rows where the expert was active.
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


@dataclass
class _MoEFeatures:
    """Decision-token MoE features for a fitted probe (issue #41)."""

    feats: List[Dict[Tuple[int, Any], Tuple[List[int], torch.Tensor, torch.Tensor]]]
    routings: List[torch.Tensor]
    labels: np.ndarray
    prompts: List[str]
    per_sample: List[Dict[str, Any]]
    capture_meta: Dict[str, Any]
    # Per sample: layer -> ||layer FFN write|| at the decision token (#41.2).
    write_norms: List[Dict[int, float]] = field(default_factory=list)


def _expert_sort_key(expert: Any) -> Tuple[int, int]:
    """Order routed experts (int) before the shared pseudo-expert."""
    return (0, int(expert)) if isinstance(expert, int) else (1, 0)


def _feature_name(layer: int, expert: Any, neuron: int) -> str:
    """Probe feature name: ``L{l}:E{e}:N{n}`` routed, ``L{l}:S:N{n}`` shared."""
    if isinstance(expert, int):
        return f"L{layer}:E{expert}:N{neuron}"
    return f"L{layer}:S:N{neuron}"


def _design_from_features(
    artifact: "_MoEFeatures",
    top_k: int,
    per_expert_top_n: int,
    score: str,
    min_active: int,
    gate_weighted: bool = True,
    normalize_write: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Flatten decision-token features into a (n_samples, n_sel) dense matrix.

    Stage 1 ranks neurons within each (layer, expert) by the chosen score over
    the samples where that expert was active; stage 2 keeps the global top-k
    (``top_k=0`` keeps all candidates).

    ``normalize_write`` divides each sample's feature values by the layer's
    total FFN write norm at the decision token, mirroring dense CETT
    (``|z| * col_norms / ||h||``) and making features comparable across
    layers, tokens and models (#41.2).
    """

    if score not in _MOE_SCORES:
        raise ValueError(f"score must be one of {_MOE_SCORES}, got {score!r}")
    if top_k < 0:
        raise ValueError("top_k must be non-negative (0 = all features)")

    # Per (layer, expert): stacked activations + gates, sample index per row.
    grouped: Dict[Tuple[int, Any], Dict[str, Any]] = {}
    for si, feat in enumerate(artifact.feats):
        for key, (tokens, acts, gates) in feat.items():
            g = grouped.setdefault(key, {"acts": [], "gates": [], "samples": []})
            for r in range(acts.shape[0]):
                g["acts"].append(acts[r])
                g["gates"].append(float(gates[r]))
                g["samples"].append(si)

    per_expert_pick: Dict[Tuple[int, Any], List[int]] = {}
    candidates: List[Tuple[float, Tuple[int, Any], int]] = []
    n_samples = max(len(artifact.feats), 1)
    for key, g in sorted(grouped.items(), key=lambda kv: (kv[0][0], _expert_sort_key(kv[0][1]))):
        A = torch.stack(g["acts"]).float()
        n = A.shape[0]
        if n < max(1, min_active):
            continue
        if score == "variance":
            ranking = A.var(dim=0)
        elif score == "mean_abs":
            # issue #41: expected contribution over ALL samples -- structural
            # zeros (expert not routed) count in the denominator.
            ranking = A.abs().sum(dim=0) / n_samples
        else:
            # conditional: mean|x| over the rows where the expert was active.
            ranking = A.abs().mean(dim=0)
        k = min(per_expert_top_n, ranking.numel())
        idx = torch.topk(ranking, k).indices.tolist()
        per_expert_pick[key] = idx
        for j in idx:
            candidates.append((float(ranking[j]), key, j))

    candidates.sort(key=lambda c: (-c[0], c[1][0], _expert_sort_key(c[1][1]), c[2]))
    chosen = candidates if top_k == 0 else candidates[:top_k]
    feature_index = [(key, j) for _, key, j in chosen]
    col_of = {fk: c for c, fk in enumerate(feature_index)}

    X = np.zeros((len(artifact.feats), len(feature_index)), dtype=np.float32)
    if normalize_write:
        missing = [
            (si, li)
            for si, wn in enumerate(artifact.write_norms)
            for li in {k[0] for k in artifact.feats[si]}
            if not wn.get(li)
        ]
        if missing:
            si, li = missing[0]
            raise RuntimeError(
                f"normalize_write=True but sample {si} has no FFN write norm for layer {li}; "
                "the block's mlp forward was not invoked. Disable normalize_write for this model."
            )
    for si, feat in enumerate(artifact.feats):
        wnorms = artifact.write_norms[si] if si < len(artifact.write_norms) else {}
        for key, (tokens, acts, gates) in feat.items():
            picks = per_expert_pick.get(key, [])
            layer = key[0]
            denom = float(wnorms.get(layer, 0.0)) if normalize_write else 0.0
            for j in picks:
                col = col_of.get((key, j))
                if col is None:
                    continue
                val = float(acts[0, j])
                if gate_weighted:
                    val *= float(gates[0])
                if normalize_write and denom > 0.0:
                    val /= denom
                X[si, col] = val

    return X, {
        "feature_index": feature_index,
        "n_candidates": len(candidates),
        "n_selected": len(feature_index),
    }


def _routing_matrix(artifact: "_MoEFeatures") -> Tuple[np.ndarray, List[str]]:
    """Routing-only design: binary indicators of routed experts per layer.

    Columns are ``(layer, expert)``; value 1 if the expert was among the
    token's top-k at the decision position, 0 otherwise. This is the #41
    routing control: any probe signal capturable from routing alone is
    attributed to it, not to neuron-level CETT.
    """
    num_experts = int(artifact.capture_meta["num_experts"])
    cols = [(lyr, e) for lyr in artifact.capture_meta["layers"] for e in range(num_experts)]
    col_of = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(artifact.feats), len(cols)), dtype=np.float32)
    for si, routing in enumerate(artifact.routings):
        for li, row in enumerate(routing):
            layer = artifact.capture_meta["layers"][li]
            for e in row.tolist():
                X[si, col_of[(layer, e)]] = 1.0
    return X, [f"L{lyr}:E{e}" for lyr, e in cols]


def _probe_metrics(X: np.ndarray, y: np.ndarray, l1_C: float, seed: int) -> Dict[str, Any]:
    """L1 probe metrics with a permutation null (no positives => null result)."""
    from sklearn.linear_model import LogisticRegression

    out: Dict[str, Any] = {"n_samples": int(X.shape[0]), "n_features": int(X.shape[1])}
    if X.shape[1] == 0 or len(np.unique(y)) < 2:
        return {**out, "n_selected": 0, "selected": [], "llr": 0.0, "perm_p": 1.0}
    clf = LogisticRegression(
        solver="liblinear", l1_ratio=1, C=l1_C, max_iter=1000, random_state=seed
    )
    clf.fit(X, y)
    coef = clf.coef_[0]
    selected = np.where(coef > 0)[0].tolist()
    llr = float(clf.decision_function(X) @ (2 * y - 1)).__float__()
    # permutation null over labels
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(200):
        yp = rng.permutation(y)
        c = LogisticRegression(
            solver="liblinear", l1_ratio=1, C=l1_C, max_iter=1000, random_state=seed
        )
        c.fit(X, yp)
        null.append(float(c.decision_function(X) @ (2 * yp - 1)))
    null_arr = np.sort(np.asarray(null))
    perm_p = float((1 + (null_arr >= llr).sum()) / (1 + len(null_arr)))
    return {
        **out,
        "n_selected": len(selected),
        "selected": selected,
        "llr": llr,
        "llr_null_mean": float(null_arr.mean()),
        "llr_null_p95": float(null_arr[int(0.95 * len(null_arr)) - 1]) if len(null_arr) else 0.0,
        "perm_p": perm_p,
    }


def fit_moe_probe(
    artifact: "_MoEFeatures",
    l1_C: float = 1.0,
    validation_split: float = 0.2,
    seed: int = 42,
    fusion: str = "concat",
    routing_only: bool = False,
) -> Dict[str, Any]:
    """Fit a probe on decision-token MoE features (#41).

    ``routing_only`` fits the routing control instead (features = per-layer
    routed-expert indicators). ``fusion="concat"`` in the default neuron-only
    mode appends the routing columns so the probe can use both; the separate
    routing-only result is returned alongside by the caller.

    ``top_k=0`` in ``capture_meta`` keeps **all** candidates (no global cap),
    matching the dense path's "0 = all features" convention.
    """

    meta = artifact.capture_meta
    y = np.asarray(artifact.labels)

    if routing_only:
        X, names = _routing_matrix(artifact)
        metrics = _probe_metrics(X, y, l1_C, seed)
        metrics["kind"] = "routing_only"
        metrics["feature_names"] = names
        return metrics

    X, design = _design_from_features(
        artifact,
        top_k=int(meta["top_k"]),
        per_expert_top_n=int(meta["per_expert_top_n"]),
        score=str(meta["score"]),
        min_active=int(meta["min_active"]),
        normalize_write=bool(meta.get("normalize_write", False)),
    )
    names = [_feature_name(lyr, e, j) for (lyr, e), j in design["feature_index"]]
    if fusion == "concat":
        Xr, rnames = _routing_matrix(artifact)
        X = np.concatenate([X, Xr], axis=1)
        names = names + rnames
    metrics = _probe_metrics(X, y, l1_C, seed)
    metrics["kind"] = "neuron"
    metrics["feature_names"] = names
    metrics["n_candidates"] = design["n_candidates"]
    metrics["layers"] = meta["layers"]
    return metrics
