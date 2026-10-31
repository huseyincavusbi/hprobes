# Steering

`hprobes.steering` provides inference-time interventions on MLP neurons and
residual-stream directions. It is the additive counterpart to the neuron-level
α-scaling used by `HProbes.causal_validate`.

## Quick start

```python
import torch
from hprobes import (
    SteeringController,
    SteeringSpec,
    median_residual_norm,
)

# Calibrate the median residual norm leaving the target layer(s).
norms = median_residual_norm(model, tokenizer(prompt, return_tensors="pt").input_ids, layers=[14])

# Add the normalized L14 N8949 write vector at every token.
spec = SteeringSpec(
    kind="neuron_write",
    layer=14,
    neuron=8949,
    sign=+1,
    dose=0.5,          # fraction of one median residual norm
    scale=norms[14],   # residual-norm calibration
)

with SteeringController(model, [spec]) as steer:
    out = model.generate(**inputs, max_new_tokens=64)
```

The hook is removed when the context exits. `steer.calls` counts hook
invocations and `steer.injected_norms` records the nominal injected norm per
spec.

## Specs

| kind | fields | effect |
|---|---|---|
| `neuron_write` | `layer`, `neuron`, `expert`, `sign`, `dose`, `scale` | add `sign × dose × scale / ‖W_down[:,n]‖` to the neuron activation before `down_proj` (dense/shared) or the corrected residual write (Gemma) |
| `direction` | `layer`, `vector`, `sign`, `dose`, `normalize`, `scale` | add `sign × dose × scale × unit(vector)` to the residual leaving `layer` |
| `alpha_scale` | `layer`, `neuron`, `dose` | multiply one neuron’s activation by `dose` |
| `ablation` | `layer`, `neuron` | set one neuron’s activation to zero |

`expert` is optional: an integer targets a routed expert in a stacked
`experts.down_proj`; `"shared"` targets `mlp.shared_expert.down_proj`.

`positions` is `all` (default) or `last`. `last` edits only the final position
of each forward pass, which is the decision token during prefill and the new
token during cached generation.

`normalize` applies to `direction`: `unit` normalizes the vector before dosing;
`none` uses the vector as given.

## Dose normalization

For `neuron_write`, the write vector is

```
W_down[:, n] / ‖W_down[:, n]‖
```

and `scale` is normally the median residual-stream norm leaving the layer
(first token excluded). `dose=1` therefore adds one typical residual norm’s
worth of the neuron direction.

`median_residual_norm` implements the calibration pass. It is intended for a
single prompt (batch size 1).

## CLI

Apply a normalized write-vector intervention:

```bash
hprobes steer \
  --model gpt2 \
  --prompt "The capital of France is" \
  --kind neuron_write \
  --layer 6 \
  --neuron 123 \
  --dose 0.5 \
  --max-new-tokens 16
```

Apply a residual direction from a `.npy` or JSON file:

```bash
hprobes steer \
  --model gpt2 \
  --prompt "..." \
  --kind direction \
  --layer 6 \
  --vector direction.npy \
  --dose 1.0
```

For `neuron_write`, `--scale` is auto-calibrated from the prompt’s median
residual norm when omitted. Use `--chat-template` for instruction-tuned models.
`--positions last` edits only the decision/final token. Add `--output path` to
save the baseline, steered text, hook count, and injected norms as JSON.

## Controls

Use the same dose and norm with:

- `dose=0`, or a zero vector with `normalize="none"`, for the bit-exact no-op gate,
- the sign flipped,
- `random_direction(hidden_size, seed=...)`,
- a norm-matched random direction,
- a held-out vector-construction split.

`contrastive_direction` builds a direction from `mean(high) - mean(low)`.

## Architecture notes

- `neuron_write` uses `get_mlp_down_proj`, which covers `down_proj`,
  `c_proj`, `fc2`, and shared experts.
- Routed experts: set `expert=<int>` for models with stacked
  `experts.down_proj` of shape `(n_experts, hidden, intermediate)`. The write
  vector is added after the gate, only on tokens where that expert is active.
  ModuleList-style experts and post-FFN-norm MoE blocks are not implemented.
- Shared experts: set `expert="shared"` to target
  `mlp.shared_expert.down_proj`.
- Gemma applies a post-feedforward norm after the MLP. `neuron_write` uses
  `residual_write_vector`, which applies the post-FFN norm scale, then injects
  at the post-FFN norm output and divides by the learned layer scalar. The raw
  `write_vector` remains available for inspection.
- Gemma 3 uses the `1 + weight` norm scale; Gemma 4 uses `weight`. This follows
  the public neuron-explorer write-column convention.
- The hook is inference-time only; no weights are modified.

## No-op gate

A zero-dose or zero-vector intervention must produce token-identical output to
the unsteered model. This is covered by `tests/test_steering.py`.
