# hprobes

Discover and causally validate hallucination-associated FFN neurons (**H-Neurons**) in transformer LLMs, including dense and mixture-of-experts models.

Builds on [arXiv:2512.01797](https://arxiv.org/abs/2512.01797).

Our paper: **Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons**, [arXiv:2609.29781](https://arxiv.org/abs/2609.29781), AIMLAI @ ECML-PKDD 2026.

## What It Does

- Identifies a sparse set of FFN neurons whose CETT activations predict hallucination
- Supports dense and mixture-of-experts models (routed + shared experts)
- Validates them causally via activation scaling, ablation, and routing counterfactuals
- Provides inference-time steering (write-vectors, contrastive directions, ablation)
- Reports bootstrap stability for dense and MoE probes
- Provides production-ready hallucination risk scoring (`detect()` / `detect_batch()`)
- Calibrates decision thresholds automatically using Youden's J statistic

## Install

```bash
pip install hprobes
# or
uv add hprobes
```

## Basic Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from hprobes import HProbes

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

probe = HProbes(model, tokenizer)
probe.fit(samples, options_key="choices", answer_key="answer")

results = probe.score()
print(f"AUROC {results['auroc']:.3f}  threshold {results['threshold']:.3f}")

# Production scoring — no ground truth needed
risk = probe.detect("Which organ is most affected? A) Heart B) Lung C) Liver D) Kidney\n\nAnswer:")
```

## Pipeline Overview

| Step | Method | What It Does |
|------|--------|--------------|
| Discover | `fit()` | Extract CETT features, train L1 logistic regression, select H-Neurons |
| Discover (MoE) | `fit_moe()` | Active-only per-expert CETT + shared experts; routing control |
| Evaluate | `score()` | AUROC on held-out validation split + random neuron baseline |
| Validate | `causal_validate()` | Scale H-Neuron activations to confirm causal role |
| Validate (MoE) | `counterfactual_routing()` | Donor/random/self-swap routing interventions |
| Steer | `SteeringController` | Write-vector, contrastive direction, α-scale, ablation |
| Detect | `detect()` / `detect_batch()` | Production hallucination risk scoring (no labels needed) |
| Transfer | `score_on()` | Score a saved probe on a different model or dataset |

## CLI

```bash
hprobes run --model google/gemma-3-4b-it --data dataset.jsonl --samples 500
hprobes moe --model allenai/OLMoE-1B-7B-0924 --data medqa.parquet --stability
hprobes steer --model gpt2 --prompt "The capital of France is" \
  --kind neuron_write --layer 6 --neuron 123 --dose 0.5
hprobes transfer --probe results/probe --model google/gemma-3-4b --data dataset.jsonl
hprobes responses --model google/gemma-3-4b-it --data responses.jsonl
```

## Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `l1_C` | `0.01` | Inverse L1 strength — lower = fewer neurons |
| `contrastive` | `True` | 3-vs-1 labeling at the generated answer token |
| `layer_stride` | `1` | Sample every Nth layer (2 = faster) |
| `batch_size` | `1` | GPU batch size for CETT extraction |
| `n_consistency` | `1` | Consistency filter draws (1 = disabled) |

## License

MIT
