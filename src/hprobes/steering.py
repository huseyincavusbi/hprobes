"""Inference-time steering primitives for hprobes.

The module provides additive neuron write-vector steering, residual direction
steering, activation scaling, and ablation. Interventions are installed as
forward hooks and are active only inside a :class:`SteeringController` context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import torch

from .cett import _get_transformer_layers, get_mlp_down_proj

SteeringKind = Literal["neuron_write", "direction", "alpha_scale", "ablation"]
SteeringPosition = Literal["all", "last"]
SteeringNormalize = Literal["unit", "none"]


def _as_tensor(output: Any) -> torch.Tensor:
    """Return the hidden-state tensor from a layer or block output."""
    if isinstance(output, tuple):
        return output[0]
    return output


def write_vector(
    model: torch.nn.Module,
    layer: int,
    neuron: int,
    expert: Optional[Union[int, str]] = None,
) -> torch.Tensor:
    """Return one MLP neuron's raw write column as a CPU float32 vector.

    ``expert=None`` targets a dense MLP or a shared-expert fallback.
    ``expert="shared"`` targets ``mlp.shared_expert.down_proj``.
    An integer targets a routed expert in a stacked ``experts.down_proj``
    parameter of shape ``(n_experts, hidden, intermediate)``.
    """
    block = _get_transformer_layers(model)[layer]
    mlp = getattr(block, "mlp", None)
    down_proj = None
    weight = None

    if expert == "shared":
        shared = getattr(mlp, "shared_expert", None)
        down_proj = getattr(shared, "down_proj", None)
    elif isinstance(expert, int):
        experts = getattr(mlp, "experts", None)
        stacked = getattr(experts, "down_proj", None)
        if stacked is None or stacked.dim() != 3:
            raise NotImplementedError(
                "routed expert steering requires stacked experts.down_proj "
                "with shape (n_experts, hidden, intermediate)"
            )
        if expert < 0 or expert >= stacked.shape[0]:
            raise IndexError(
                f"expert {expert} out of range for layer {layer} (n_experts {stacked.shape[0]})"
            )
        weight = stacked[expert]
    else:
        down_proj = get_mlp_down_proj(model, layer)

    if weight is None:
        weight = getattr(down_proj, "weight", None)
    if weight is None or weight.dim() != 2:
        raise ValueError(
            f"down projection at layer {layer} has no 2-D weight; got {type(down_proj).__name__}"
        )
    if neuron < 0 or neuron >= weight.shape[1]:
        raise IndexError(
            f"neuron {neuron} out of range for layer {layer} (width {weight.shape[1]})"
        )
    return weight[:, neuron].detach().float().cpu()


def _down_proj_for(
    model: torch.nn.Module, layer: int, expert: Optional[Union[int, str]] = None
) -> torch.nn.Module:
    if expert is None:
        return get_mlp_down_proj(model, layer)
    if expert == "shared":
        block = _get_transformer_layers(model)[layer]
        shared = getattr(getattr(block, "mlp", None), "shared_expert", None)
        down_proj = getattr(shared, "down_proj", None)
        if down_proj is None:
            raise NotImplementedError(f"layer {layer} has no shared_expert.down_proj")
        return down_proj
    raise ValueError("routed experts use the expert forward hook, not down_proj")


def _model_type_name(model: torch.nn.Module) -> str:
    config = getattr(model, "config", None)
    name = str(getattr(config, "model_type", "") or "")
    text_config = getattr(config, "text_config", None)
    text_name = str(getattr(text_config, "model_type", "") or "")
    return f"{name} {text_name}".lower()


def _post_ffn_scale(model: torch.nn.Module, block: torch.nn.Module) -> Optional[torch.Tensor]:
    """Return the post-feedforward norm scale for Gemma-style layers."""
    norm = getattr(block, "post_feedforward_layernorm", None)
    weight = getattr(norm, "weight", None)
    if weight is None:
        return None
    scale = weight.detach().float().cpu()
    class_name = type(block).__name__.lower()
    name = _model_type_name(model)
    if "gemma4" in name or "gemma4" in class_name:
        return scale
    if "gemma" in name or "gemma" in class_name:
        return 1.0 + scale
    return scale


def _layer_scalar(block: torch.nn.Module) -> float:
    scalar = getattr(block, "layer_scalar", None)
    if scalar is None or getattr(scalar, "numel", lambda: 0)() != 1:
        return 1.0
    value = float(scalar.detach().float().cpu())
    return value if value != 0 else 1.0


def residual_write_vector(
    model: torch.nn.Module,
    layer: int,
    neuron: int,
    expert: Optional[Union[int, str]] = None,
) -> torch.Tensor:
    """Return the write vector that reaches the residual stream.

    For standard pre-norm MLPs this is the raw ``W_down[:, n]`` column. For
    Gemma-style layers it applies the post-feedforward norm scale. The learned
    layer scalar is applied separately at injection time.
    """
    block = _get_transformer_layers(model)[layer]
    if expert is not None and _post_ffn_scale(model, block) is not None:
        raise NotImplementedError(
            "expert write steering is not implemented for post-FFN-norm layers"
        )
    vector = write_vector(model, layer, neuron, expert=expert)
    scale = _post_ffn_scale(model, block)
    if scale is None:
        return vector
    if scale.numel() != vector.numel():
        raise ValueError(
            f"post-FFN scale width {scale.numel()} does not match write vector "
            f"width {vector.numel()}"
        )
    return scale * vector


def contrastive_direction(
    high: torch.Tensor,
    low: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """Build a direction from ``mean(high) - mean(low)``.

    ``high`` and ``low`` must be 2-D tensors ``(n_samples, hidden)``. The
    result is returned on CPU as float32.
    """
    if high.shape != low.shape:
        raise ValueError(f"shape mismatch: high={tuple(high.shape)} low={tuple(low.shape)}")
    if high.dim() != 2:
        raise ValueError(f"expected 2-D tensors, got {high.dim()}-D")
    direction = high.float().mean(dim=0) - low.float().mean(dim=0)
    if normalize:
        norm = direction.norm()
        if not torch.isfinite(norm) or norm == 0:
            raise ValueError("cannot normalize a zero/non-finite contrastive direction")
        direction = direction / norm
    return direction.detach().cpu()


def random_direction(hidden_size: int, seed: int = 0, normalize: bool = True) -> torch.Tensor:
    """Return a seeded random direction for norm-matched controls."""
    if hidden_size < 1:
        raise ValueError(f"hidden_size must be >= 1, got {hidden_size}")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    direction = torch.randn(hidden_size, generator=generator, dtype=torch.float32)
    if normalize:
        direction = direction / direction.norm()
    return direction


def median_residual_norm(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    layers: Optional[Sequence[int]] = None,
    exclude_first: bool = True,
) -> Dict[int, float]:
    """Measure the median residual-stream norm leaving each layer.

    Calibration is intended for a single prompt (batch size 1). Token 0 is
    excluded by default because it is the attention sink.
    """
    blocks = _get_transformer_layers(model)
    layer_ids = list(range(len(blocks))) if layers is None else list(layers)
    captured: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layer_ids:

        def make_hook(idx: int):
            def hook(module, args, output):
                captured[idx] = _as_tensor(output).detach()

            return hook

        handles.append(blocks[layer_idx].register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()

    out: Dict[int, float] = {}
    for layer_idx, hidden in captured.items():
        norms = hidden.float().norm(dim=-1)
        if norms.dim() == 2:
            norms = norms[0]
        if exclude_first and norms.numel() > 1:
            norms = norms[1:]
        if norms.numel():
            out[layer_idx] = float(norms.median())
    return out


@dataclass
class SteeringSpec:
    """One steering intervention.

    ``neuron_write`` adds ``sign * dose * scale / ||W_down[:, n]||`` to the
    neuron's activation before ``down_proj`` on standard pre-norm MLPs. On
    post-FFN-norm layers (Gemma) it injects the corrected residual write vector
    at the post-FFN norm output. ``direction`` adds
    ``sign * dose * scale * unit(vector)`` to the residual leaving ``layer``.
    ``alpha_scale`` multiplies one neuron's activation by ``dose``;
    ``ablation`` sets it to zero. For MoE, ``expert`` selects a routed expert
    (stacked ``experts.down_proj``) or ``"shared"``.
    """

    kind: SteeringKind
    layer: int
    dose: float = 1.0
    sign: int = 1
    neuron: Optional[int] = None
    expert: Optional[Union[int, str]] = None
    vector: Optional[torch.Tensor] = None
    positions: SteeringPosition = "all"
    normalize: SteeringNormalize = "unit"
    scale: Optional[float] = None

    def __post_init__(self) -> None:
        if self.kind not in ("neuron_write", "direction", "alpha_scale", "ablation"):
            raise ValueError(f"unknown steering kind: {self.kind!r}")
        if self.layer < 0:
            raise ValueError(f"layer must be >= 0, got {self.layer}")
        if self.sign not in (-1, 1):
            raise ValueError(f"sign must be -1 or 1, got {self.sign}")
        if self.dose < 0:
            raise ValueError(f"dose must be >= 0, got {self.dose}")
        if self.positions not in ("all", "last"):
            raise ValueError(f"unknown positions: {self.positions!r}")
        if self.normalize not in ("unit", "none"):
            raise ValueError(f"unknown normalize mode: {self.normalize!r}")
        if self.kind in ("neuron_write", "alpha_scale", "ablation") and self.neuron is None:
            raise ValueError(f"{self.kind} requires neuron")
        if self.kind == "direction" and self.vector is None:
            raise ValueError("direction requires vector")
        if self.expert is not None:
            if self.kind != "neuron_write":
                raise ValueError("expert is only supported for neuron_write")
            if not (isinstance(self.expert, int) or self.expert == "shared"):
                raise ValueError("expert must be an int or 'shared'")


class SteeringController:
    """Install and remove a set of :class:`SteeringSpec` hooks."""

    def __init__(
        self,
        model: torch.nn.Module,
        specs: Sequence[SteeringSpec],
        residual_norms: Optional[Dict[int, float]] = None,
    ) -> None:
        self.model = model
        self.specs = list(specs)
        self.residual_norms = dict(residual_norms or {})
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.calls = 0
        self.injected_norms: List[float] = []
        for spec in self.specs:
            self._register(spec)

    def _scale_for(self, spec: SteeringSpec) -> float:
        if spec.scale is not None:
            return float(spec.scale)
        return float(self.residual_norms.get(spec.layer, 1.0))

    def _register(self, spec: SteeringSpec) -> None:
        if spec.kind == "direction":
            self._register_direction(spec)
            return

        block = _get_transformer_layers(self.model)[spec.layer]
        neuron = int(spec.neuron)  # type: ignore[arg-type]
        if spec.kind == "neuron_write" and isinstance(spec.expert, int):
            self._register_routed_expert_write(spec, block)
            return
        if spec.kind == "neuron_write":
            vector = residual_write_vector(self.model, spec.layer, neuron, expert=spec.expert)
            vector_norm = vector.norm()
            if not torch.isfinite(vector_norm) or vector_norm == 0:
                raise ValueError(f"invalid write vector at L{spec.layer} N{neuron}")
            delta = spec.sign * spec.dose * self._scale_for(spec) / vector_norm * vector

            if _post_ffn_scale(self.model, block) is not None:
                delta = delta / _layer_scalar(block)
                self.injected_norms.append(spec.dose * self._scale_for(spec))
                self._register_residual(block.post_feedforward_layernorm, delta, spec.positions)
                return

            down_proj = _down_proj_for(self.model, spec.layer, spec.expert)
            # Standard pre-norm MLP: adding to the neuron activation before
            # down_proj is equivalent to adding delta * W_down[:, n] to the
            # MLP output.
            activation_delta = spec.sign * spec.dose * self._scale_for(spec) / vector_norm
            self.injected_norms.append(abs(float(activation_delta)) * float(vector_norm))

            def pre_hook(
                module,
                args,
                delta=activation_delta,
                neuron=neuron,
                positions=spec.positions,
            ):
                z = args[0].clone()
                if z.dim() == 3:
                    if positions == "last":
                        z[:, -1, neuron] += delta
                    else:
                        z[..., neuron] += delta
                elif z.dim() == 2:
                    z[:, neuron] += delta
                else:
                    raise RuntimeError(f"unexpected down_proj input shape: {tuple(z.shape)}")
                self.calls += 1
                return (z,) + tuple(args[1:])

            self.handles.append(down_proj.register_forward_pre_hook(pre_hook))
            return

        down_proj = _down_proj_for(self.model, spec.layer, spec.expert)
        alpha = 0.0 if spec.kind == "ablation" else float(spec.dose)
        self.injected_norms.append(0.0)

        def pre_hook(module, args, alpha=alpha, neuron=neuron, positions=spec.positions):
            z = args[0].clone()
            if z.dim() == 3:
                if positions == "last":
                    z[:, -1, neuron] *= alpha
                else:
                    z[..., neuron] *= alpha
            elif z.dim() == 2:
                z[:, neuron] *= alpha
            else:
                raise RuntimeError(f"unexpected down_proj input shape: {tuple(z.shape)}")
            self.calls += 1
            return (z,) + tuple(args[1:])

        self.handles.append(down_proj.register_forward_pre_hook(pre_hook))

    def _register_routed_expert_write(self, spec: SteeringSpec, block: torch.nn.Module) -> None:
        if _post_ffn_scale(self.model, block) is not None:
            raise NotImplementedError(
                "expert write steering is not implemented for post-FFN-norm layers"
            )
        experts = getattr(getattr(block, "mlp", None), "experts", None)
        stacked = getattr(experts, "down_proj", None)
        if stacked is None or stacked.dim() != 3:
            raise NotImplementedError(
                "routed expert steering requires stacked experts.down_proj "
                "with shape (n_experts, hidden, intermediate)"
            )
        expert = int(spec.expert)  # type: ignore[arg-type]
        neuron = int(spec.neuron)  # type: ignore[arg-type]
        vector = write_vector(self.model, spec.layer, neuron, expert=expert)
        vector_norm = vector.norm()
        if not torch.isfinite(vector_norm) or vector_norm == 0:
            raise ValueError(f"invalid expert write vector at L{spec.layer} E{expert} N{neuron}")
        delta = spec.sign * spec.dose * self._scale_for(spec) / vector_norm * vector
        self.injected_norms.append(float(delta.norm()))

        def hook(module, args, output, delta=delta, expert=expert):
            hidden = _as_tensor(output).clone()
            if hidden.dim() != 2:
                raise RuntimeError(
                    "routed expert steering expects a 2-D experts output "
                    f"(n_tokens, hidden); got {tuple(hidden.shape)}"
                )
            if len(args) < 2 or not torch.is_tensor(args[1]):
                raise RuntimeError("experts forward must expose top_k_index as its second argument")
            top_k = args[1]
            mask = top_k == expert
            if mask.dim() > 1:
                mask = mask.any(dim=-1)
            if spec.positions == "last":
                last = torch.zeros_like(mask)
                last[-1] = True
                mask = mask & last
            if mask.any():
                key = delta.to(device=hidden.device, dtype=hidden.dtype)
                hidden[mask] += key
            self.calls += 1
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        self.handles.append(experts.register_forward_hook(hook))

    def _register_direction(self, spec: SteeringSpec) -> None:
        vector = spec.vector
        if vector is None:
            raise ValueError("direction requires vector")
        vector = vector.detach().float().cpu()
        if spec.normalize == "unit":
            norm = vector.norm()
            if not torch.isfinite(norm) or norm == 0:
                raise ValueError("cannot normalize a zero/non-finite direction")
            vector = vector / norm
        delta = spec.sign * spec.dose * self._scale_for(spec) * vector
        self.injected_norms.append(float(delta.norm()))
        blocks = _get_transformer_layers(self.model)
        self._register_residual(blocks[spec.layer], delta, spec.positions)

    def _register_residual(
        self, block: torch.nn.Module, delta: torch.Tensor, positions: SteeringPosition
    ) -> None:
        def hook(module, args, output, delta=delta, positions=positions):
            hidden = _as_tensor(output).clone()
            if hidden.dim() == 3:
                if positions == "last":
                    hidden[:, -1, :] += delta.to(device=hidden.device, dtype=hidden.dtype)
                else:
                    hidden = hidden + delta.to(device=hidden.device, dtype=hidden.dtype)
            elif hidden.dim() == 2:
                hidden = hidden + delta.to(device=hidden.device, dtype=hidden.dtype)
            else:
                raise RuntimeError(f"unexpected layer output shape: {tuple(hidden.shape)}")
            self.calls += 1
            if isinstance(output, tuple):
                return (hidden,) + output[1:]
            return hidden

        self.handles.append(block.register_forward_hook(hook))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __enter__(self) -> "SteeringController":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False
