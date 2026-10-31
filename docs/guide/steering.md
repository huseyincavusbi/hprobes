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
    write_vector,
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
| `neuron_write` | `layer`, `neuron`, `sign`, `dose`, `scale` | add `sign × dose × scale / ‖W_down[:,n]‖` to the neuron activation before `down_proj` |
| `direction` | `layer`, `vector`, `sign`, `dose`, `normalize`, `scale` | add `sign × dose × scale × unit(vector)` to the residual leaving `layer` |
| `alpha_scale` | `layer`, `neuron`, `dose` | multiply one neuron’s activation by `dose` |
| `ablation` | `layer`, `neuron` | set one neuron’s activation to zero |

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

## Controls

Use the same dose and norm with:

- a zero vector / `dose=0` for the bit-exact no-op gate,
- the sign flipped,
- a random direction,
- a norm-matched random direction,
- a held-out vector-construction split.

`contrastive_direction` builds a direction from `mean(high) - mean(low)`.

## Architecture notes

- `neuron_write` uses `get_mlp_down_proj`, which covers `down_proj`,
  `c_proj`, `fc2`, and shared experts.
- Routed-expert write-vector steering is not implemented in this slice. For a
  routed-only MoE, use a residual `direction` or the MoE capture/routing tools.
- Gemma applies a post-feedforward norm after the MLP. The true residual write
  direction is the normed column, not the raw `down_proj` column. Use
  `direction` for Gemma residual steering until the Gemma write-column
  correction is implemented.
- The hook is inference-time only; no weights are modified.

## No-op gate

A zero-dose or zero-vector intervention must produce token-identical output to
the unsteered model. This is covered by `tests/test_steering.py`.
