# CLI Reference

hprobes provides three CLI subcommands: `run`, `responses`, and `transfer`.

```bash
hprobes --help
hprobes --version
```

## `hprobes run`

Fit, score, and causally validate a probe on an MCQ dataset.

```bash
hprobes run \
  --model google/gemma-3-4b-it \
  --data dataset.jsonl \
  --samples 500 \
  --l1-c 0.01 \
  --batch-size 8
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model` | HuggingFace model ID (e.g. `google/gemma-3-4b-it`) |
| `--data` | Path to `.jsonl`, `.json`, or `.parquet` dataset file |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--format` | `auto` | Dataset format: `auto`, `mmlu`, `medqa`, `medmcqa` |
| `--samples` | `-1` (all) | Number of samples to use |
| `--no-contrastive` | — | Disable contrastive labeling (use binary mode) |
| `--output` | auto-named | Base path for output files (`.json` + `.pkl`) |
| `--device` | `auto` | Device: `auto`, `cpu`, `mps`, `cuda` |
| `--dtype` | `auto` | Precision: `auto`, `float16`, `bfloat16`, `float32` |
| `--attn-implementation` | `auto` | Attention implementation: `auto`, `eager`, `sdpa`, `flash_attention_2` |
| `--l1-c` | `0.5` | Inverse L1 regularisation strength (Python API default is `0.01`) |
| `--seed` | `42` | Random seed |
| `--layer-stride` | `1` | Sample every Nth layer |
| `--validation-split` | `0.2` | Fraction held out for validation |
| `--max-tokens` | `1024` | Max input tokens before truncation |
| `--alphas` | `0.0,0.5,1.0,1.5,2.0` | Comma-separated alpha values for causal validation |
| `--batch-size` | `1` | Batch size for CETT extraction |

> **Numerical-equivalence warning (research integrity).** `eager` attention is the
> reference used for published results. `sdpa` and `flash_attention_2` are faster
> but only *approximately* numerically equal to eager — they can shift activations
> and flip borderline predictions. They must **not** be used to produce or
> reproduce published paper numbers unless you run and record an equivalence
> check (see `tests/test_attn_equivalence.py`). The default `auto` keeps the
> model's native path unchanged.

### Output

Produces two files:

- `<output>.json` — Human-readable results (H-Neurons, AUROC, accuracy, causal validation)
- `<output>.pkl` — Serialized classifier for later loading

## `hprobes responses`

Fit from pre-generated responses with judge labels (open-ended / free-text mode).

```bash
hprobes responses \
  --model google/gemma-3-4b-it \
  --data responses.jsonl \
  --question-key question \
  --response-key response \
  --answer-tokens-key answer_tokens \
  --label-key judge
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model` | HuggingFace model ID |
| `--data` | Path to dataset file |

### Additional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--question-key` | `question` | Key for question text |
| `--response-key` | `response` | Key for generated response |
| `--answer-tokens-key` | `answer_tokens` | Key for answer token list |
| `--label-key` | `judge` | Key for correctness label (`true`/`false` or `1`/`0`) |
| `--aggregation` | `mean` | How to aggregate CETT over the answer span: `mean` or `max` |

All shared arguments (`--device`, `--dtype`, `--l1-c`, etc.) are the same as `hprobes run`.

## `hprobes transfer`

Score a saved probe on a different model (transfer experiment).

```bash
hprobes transfer \
  --probe results/gemma_medqa \
  --model google/medgemma-27b-text-it \
  --data dataset.jsonl
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--probe` | Base path of saved probe (e.g. `results/gemma_medqa`) |
| `--model` | HuggingFace model ID for the target model |
| `--data` | Path to dataset file |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--format` | `auto` | Dataset format |
| `--samples` | `-1` (all) | Number of samples |
| `--output` | auto-named | Output path |
| `--device` | `auto` | Device |
| `--dtype` | `auto` | Precision |
| `--max-tokens` | `1024` | Max input tokens |

### What Transfer Does

1. Loads the saved classifier (`.pkl`) and attaches it to the target model
2. Extracts CETT from the **new** model using the **original** layer/neuron mapping
3. Normalizes features using the **original** training statistics
4. Scores with the **original** classifier
5. Computes AUROC and random baseline on the new data
