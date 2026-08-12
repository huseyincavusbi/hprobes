# hprobes

[![Docs](https://img.shields.io/badge/docs-github.io-blue)](https://huseyincavusbi.github.io/hprobes)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/huseyincavusbi/hprobes)

Discover and causally validate hallucination-associated FFN neurons (H-Neurons) in transformer LLMs.

Based on [arXiv:2512.01797](https://arxiv.org/abs/2512.01797).

## Install

```bash
pip install hprobes
# or
uv add hprobes
```

## Quickstart

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from hprobes import HProbe

model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")

# samples: list of dicts with question, options, answer
probe = HProbe(model, tokenizer)
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
probe = HProbe.load("results/gemma_medqa", model, tokenizer)
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
