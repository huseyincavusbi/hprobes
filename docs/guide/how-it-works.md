# How It Works

hprobes implements the H-Neuron discovery pipeline from [arXiv:2512.01797](https://arxiv.org/abs/2512.01797). This page explains the core concepts.

## The CETT Metric

**CETT** (Contribution to rEsidual sTream norm of Token t) measures how much each individual FFN neuron contributes to the model's hidden state at a given token position.

For neuron `j` at token position `t`:

```
CETT(j, t) = |z_{j,t}| * ||W_down[:, j]||_2 / ||h_t||_2
```

Where:

- `z_{j,t}` — SwiGLU activation of neuron `j` at token `t` (input to `W_down`)
- `W_down[:, j]` — the `j`-th column of the down-projection weight matrix
- `h_t = W_down * z_t` — FFN output vector at token `t`

The column norms `||W_down[:, j]||_2` are precomputed once and reused across all samples.

## H-Neuron Discovery

The discovery pipeline has four steps:

### 1. Feature Extraction

For each MCQ sample, the model predicts an answer letter by reading logits at the last prompt token. CETT is extracted and the prediction is compared against the ground truth.

### 2. Labeling

**Contrastive mode** (default, `contrastive=True`):

- Appends the predicted answer token to the prompt and captures CETT at that position
- If the model is **wrong**: label = 1 (hallucination)
- If the model is **correct**: label = 0
- Also captures CETT at the last **prompt** token and always labels it 0 (negative example)
- This 3-vs-1 scheme gives the classifier both "where the model decides" and "where it hallucinates"

**Non-contrastive mode** (`contrastive=False`):

- Captures CETT at the last prompt token only
- Incorrect answers → label 1, correct → label 0

### 3. Sparse Selection

1. **Variance pre-selection**: Top 5,000 features by variance (Welford's online algorithm)
2. **Standardization**: Zero-mean, unit-variance normalization
3. **L1 logistic regression**: `LogisticRegression(solver="liblinear", l1_ratio=1, C=l1_C, class_weight="balanced")`
4. **H-Neuron extraction**: Neurons with positive coefficients in the trained classifier

The `l1_C` parameter controls sparsity — lower values yield fewer, more confident H-Neurons.

### 4. Causal Validation

To confirm that H-Neurons causally influence hallucination (not just correlate with it), `causal_validate()` scales their activations during inference:

| Alpha | Effect |
|-------|--------|
| 0.0 | Full suppression (zero out H-Neurons) |
| 0.5 | Partial suppression |
| 1.0 | Baseline (no intervention) |
| 1.5 | Mild amplification |
| 2.0 | Strong amplification |

If H-Neurons are truly causal, suppressing them should change accuracy on the validation set.

## Contrastive vs Non-Contrastive

| | Contrastive (default) | Non-Contrastive |
|---|---|---|
| **CETT position** | Predicted answer token | Last prompt token |
| **Forward passes** | 2 per sample | 1 per sample |
| **Labeling** | 3-vs-1 (answer + prompt rows) | Binary (correct/incorrect) |
| **Captures** | Where hallucination manifests | Where the model makes the decision |

Contrastive mode is recommended for most use cases. It produces more discriminative features because it captures the model's representation specifically at the point of answer generation.

## Consistency Filtering

When `n_consistency > 1`, the probe draws `n` samples from the restricted letter distribution (softmax over A–J logits) before committing to a prediction. If the draws disagree, the sample is skipped entirely.

This matches the paper's methodology of filtering out ambiguous samples where the model is uncertain between multiple letters.

## Threshold Calibration

After `score()`, the probe computes an optimal decision threshold using **Youden's J statistic**:

```
J = TPR - FPR
threshold = argmax(J)
```

This is stored as `probe.threshold_` and persisted through `save()`/`load()`. The threshold is used in production when you need a binary hallucination/not-hallucination decision rather than a continuous score.

## Performance Optimizations and Numerical Equivalence

hprobes batches the deterministic CETT extraction loops (`fit_from_responses`, `score_on_responses`, `causal_validate`) when `batch_size > 1`, and uses `torch.inference_mode()` (numerically identical to `no_grad`, just faster). The batched paths are verified equivalent to the single-sample paths by `tests/test_equivalence.py` and `tests/test_attn_equivalence.py`.

### Attention implementations are approximately-equal, not bit-identical

`eager` attention is the reference used for published results. `sdpa` and `flash_attention_2` (opt-in via `--attn-implementation`) are faster but only approximately numerically equal — fused kernels reorder floating-point accumulation. On MPS we measured eager-vs-sdpa logits differing by up to ~0.5 (argmax stable on test samples); activations at `down_proj` shift by ~1e-2–1e-1 relative. This can flip borderline predictions. **Do not use sdpa/flash to produce or reproduce published paper numbers unless you run and record an equivalence check** (the harness in `tests/test_attn_equivalence.py`).

### Why the hooked probing paths do not use `torch.compile`

CETT extraction and α-scaling read/modify activations through forward hooks that write to Python dicts. `torch.compile` with `fullgraph=True` is incompatible with this pattern: the hook logic gets baked into the compiled graph at trace time and never executes during the forward pass, so no activations are captured (see pytorch/pytorch#173452, open as of 2026). The `fullgraph=False` workaround loses ~30–50% of the CUDA-graph speedup. Consequently the probing paths intentionally stay eager. `torch.compile` is only safe for hook-free, fixed-shape forward passes (e.g. plain generation).
