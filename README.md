# hprobes

[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://huseyincavusbi.github.io/hprobes)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/huseyincavusbi/hprobes)

Discover and causally validate hallucination-associated FFN neurons (H-Neurons) in transformer LLMs, including dense and mixture-of-experts models.

Builds on [arXiv:2512.01797](https://arxiv.org/abs/2512.01797).

Our paper: **Hallucination Neurons and Where to Find Them: An Investigation into the existence of Hallucination Neurons**, [arXiv:2609.29781](https://arxiv.org/abs/2609.29781), AIMLAI @ ECML-PKDD 2026.

## Features

- Dense and MoE H-Neuron discovery
- Causal validation: α-scaling, ablation, routing counterfactuals
- Inference-time steering: neuron write-vectors, contrastive directions, ablation
- Bootstrap stability for dense and MoE probes
- Production hallucination-risk scoring (`detect()` / `detect_batch()`)

## Install

```bash
pip install hprobes
# or
uv add hprobes
```

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from hprobes import HProbes

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

# samples: list of dicts with question, options, answer
probe = HProbes(model, tokenizer)
probe.fit(samples, options_key="choices", answer_key="answer")

print(probe.n_neurons_, probe.layer_distribution_)

results = probe.score()
print(f"AUROC {results['auroc']:.3f}  gap {results['auroc_gap']:+.3f}")

probe.causal_validate()
```

## CLI

```bash
# Fit and score on an MCQ dataset
hprobes run --model google/gemma-3-4b-it --data dataset.jsonl --samples 500

# Transfer: score a saved probe on a different model
hprobes transfer --probe results/probe --model google/gemma-3-4b --data dataset.jsonl

# Fit from pre-generated responses with judge labels
hprobes responses --model google/gemma-3-4b-it --data responses.jsonl

# MoE probing (routed + shared experts)
hprobes moe --model allenai/OLMoE-1B-7B-0924 --data medqa.parquet --score mean_abs --stability

# Inference-time steering
hprobes steer --model gpt2 --prompt "The capital of France is" \
  --kind neuron_write --layer 6 --neuron 123 --dose 0.5
```

## Supported formats

Input files: `.jsonl`, `.json`, `.parquet`

Auto-detected dataset formats: `mmlu`, `medqa`, `medmcqa`. Any other format works by passing `options_key` and `answer_key` directly.

## Key options

| Parameter | Default | Description |
|---|---|---|
| `l1_C` | `0.01` | Inverse L1 strength — lower = fewer neurons |
| `contrastive` | `True` | 3-vs-1 labeling at the generated answer token |
| `layer_stride` | `1` | Sample every Nth layer (2 = faster) |
| `validation_split` | `0.2` | Holdout fraction for scoring |
| `max_tokens` | `1024` | Truncation length |

## Save & load

```python
probe.save("results/gemma_medqa")          # writes .json + .pkl
probe = HProbes.load("results/gemma_medqa", model, tokenizer)
probe.score_on(new_samples, options_key="choices", answer_key="answer")
```


## Acknowledgements

This research is conducted in collaboration with the
[Great Ormond Street Hospital DRIVE Unit](https://www.gosh.nhs.uk/our-research/drive-unit-for-digital-innovation/).

## Contributors

- **Huseyin Cavus** — Core Contributor
- **Pavithra Rajendran, PhD** — Machine Learning Lead, GOSH DRIVE
- **Sebin Sabu** — Senior AI Scientist, GOSH DRIVE
- **Jaskaran Singh Kawatra** — ML Engineer, GOSH DRIVE
- **Josh Spear, PhD** — Postdoctoral Researcher, GOSH DRIVE
