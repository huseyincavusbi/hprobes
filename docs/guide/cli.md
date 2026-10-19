# CLI Reference

hprobes provides six CLI subcommands: `run`, `moe`, `responses`, `transfer`, `compare`, and `edit`.

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
| `--top-k` | `0` (all features) | Variance pre-selection: keep top-K features. `0` = all features (published pipeline); `auto` = largest K that safely fits available RAM; a positive int is an explicit cap |
| `--auto-safety` | `0.8` | RAM safety factor for `--top-k auto` |
| `--no-predictions` | off (predictions are written) | Skip the per-sample predictions sidecar |
| `--readout` | `both` | MCQ answer readout: `logits` (top letter token; legacy), `generate` (greedy generation + parse), `both` (default; generation is the primary label, both recorded). Causal validation uses the same readout |
| `--alphas` | `0.0,0.5,1.0,1.5,2.0` | Comma-separated alpha values for causal validation |
| `--batch-size` | `1` | Batch size for CETT extraction |

> **Auto top-k (`--top-k auto`).** Resolves the largest feature count that safely fits the
> machine's available RAM at fit time (Linux `/proc/meminfo`, macOS `vm_stat`, `psutil` when
> installed, `os.sysconf` fallback). The fit peak is linear in `top_k` — `8·n_rows·k·(1+n_fits)`
> bytes per the analytic model, corrected by a measured calibration factor for the solver's real
> internals — so it is inverted as
> `k = (safety·available − cett_bytes − overhead) // (calibration · 8·n_rows·(1+n_fits))`,
> capped at all features. If even a minimal `k` cannot fit, the run raises `MemoryError`
> with the maximum sample count instead of an OOM kill. The resolved value and reasoning
> are recorded in the results JSON (`top_k`, `top_k_auto`, `top_k_reason`).

> **Numerical-equivalence warning (research integrity).** `eager` attention is the
> reference used for published results. `sdpa` and `flash_attention_2` are faster
> but only *approximately* numerically equal to eager — they can shift activations
> and flip borderline predictions. They must **not** be used to produce or
> reproduce published paper numbers unless you run and record an equivalence
> check (see `tests/test_attn_equivalence.py`). The default `auto` keeps the
> model's native path unchanged.

### Output

Produces:

- `<output>.json` — Human-readable results (H-Neurons, AUROC, accuracy, causal validation)
- `<output>.pkl` — Serialized classifier for later loading
- `<output>.predictions.jsonl` — Per-sample predictions (written by default)

The predictions sidecar has one JSON object per sample. For MCQ, `--readout` controls the
primary label (`predicted` / `is_correct`): `logits` uses the top letter token (legacy),
`generate` parses the greedy generation, and `both` (default) makes the **generation** the
primary label while recording both readouts:

```json
{"index": 0, "prompt": "...exact model input...", "ground_truth": "B",
 "predicted": "B", "is_correct": true,
 "letter_logits": {"A": 10.42, "B": 17.88, "C": 9.15, "D": 12.03},
 "logits_letter": "B", "logits_is_correct": true,
 "generated": "B. proteins.", "generated_letter": "B", "generated_is_correct": true}
```

Open-ended runs record `{"index", "prompt", "is_correct"}` (raw response text is saved
separately by the response pipeline). Disable with `--no-predictions`.

Causal validation follows the same readout: with `generate`/`both`, each alpha's accuracy is
measured on generated answers, and unparseable generations count as **incorrect** so the
denominator stays fixed across alphas.

## `hprobes moe`

Fit H-Neurons on **Mixture-of-Experts** models (issue #41). Routed experts are replayed
read-only with active-only per-expert CETT, always-on shared experts (Qwen2MoE / Qwen3.5 MoE
style) are captured and fused into the same design, and correctness labels come from the
same forward's decision-token logits.

```bash
hprobes moe \
  --model allenai/OLMoE-1B-7B-0924 \
  --data medqa.parquet \
  --samples 400 \
  --score variance \
  --per-expert-top-n 64 \
  --min-active 2 \
  --top-k 2000 \
  --experts-implementation eager
```

> **Eager experts protocol (research integrity).** `transformers >= 5.20` dispatches MoE
> experts through fast kernels (`grouped_mm` / `batched_mm`) by default. The capture replays
> the layer's `eager` expert loop, so `fit_moe` raises a guard error on models loaded with a
> fast dispatch. Pass `--experts-implementation eager` to load the model exactly like the
> published protocol (recommended). The default `auto` keeps the model's native dispatch.

### Required Arguments

| Argument | Description |
|----------|-------------|
| `--model` | HuggingFace model ID (e.g. `allenai/OLMoE-1B-7B-0924`) |
| `--data` | Path to `.jsonl`, `.json`, or `.parquet` dataset file |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--format` | `auto` | Dataset format: `auto`, `mmlu`, `medqa`, `medmcqa` |
| `--samples` | `-1` (all) | Number of samples to use |
| `--question-key` | `question` | Key for question text |
| `--options-key` | from `--format` | Options field override |
| `--answer-key` | from `--format` | Answer field override |
| `--score` | `mean_abs` | Per-expert neuron ranking: `variance` (zeros counted), `mean_abs` (active-row mean \|z\|, historical default), `expected` (gate-weighted expected value), `conditional` (alias of `mean_abs`) |
| `--per-expert-top-n` | `64` | Neurons kept per `(layer, expert)` before the global cap |
| `--min-active` | `2` | Minimum active rows for a neuron to be scorable |
| `--top-k` | `2000` | Global cap after per-expert ranking. `0` = all candidates. `auto` is dense-only |
| `--layers` | all layers | Comma-separated layer indices (default: every layer selected by `--layer-stride`) |
| `--fusion` | `concat` | `concat` = neuron features + routing indicators; `neuron` = neuron features only |
| `--no-routing-baseline` | off | Skip the routing-only control probe fitted alongside the main probe |
| `--normalize-write` | off | Divide features by the layer's total FFN write norm at the decision token, mirroring dense CETT |
| `--routing-analysis` | off | After fitting: routing-vs-correctness analysis — per-layer usage deltas with permutation nulls, gate entropy, routing churn, and the CETT × routing cross-check (one extra forward per sample) |
| `--n-perm` | `200` | Permutations for the `--routing-analysis` nulls |
| `--counterfactual` | off | After fitting: counterfactual routing — matched-donor rescue/corruption, frequency-matched random control, self-swap no-op gate (many extra forwards; bound the sweep with `--doses`) |
| `--doses` | `1,2,5` (capped at k) | Expert-slot doses for `--counterfactual` |
| `--experts-implementation` | `auto` | `eager` loads the published capture protocol; `auto` keeps the model default |
| `--output` | auto-named | Base path for the results JSON |
| `--device` / `--dtype` / `--attn-implementation` / `--trust-remote-code` | as `run` | Model-loading options |
| `--l1-c` / `--solver` / `--seed` / `--layer-stride` / `--validation-split` / `--max-tokens` / `--batch-size` | as `run` | Probe options |
| `--readout` | `logits` | `fit_moe` labels from the decision-token logits; any other value is rejected |
| `--no-predictions` | off (predictions are written) | Skip the per-sample predictions sidecar |

Dense-only validation options (`--alphas`, `--l2`, `--stability`, `--correlation`,
`--cluster`, `--random-baseline`) are accepted for CLI parity but do not run in MoE mode;
passing one prints a note and the flag is ignored.

### Output

`<output>.json` — MoE probe results:

```json
{
  "mode": "moe",
  "config": {"score": "variance", "top_k": 2000, "min_active": 2, "normalize_write": false},
  "moe": {"n_samples": 400, "n_candidates": 11648, "n_selected": 25, "llr": 137.4,
           "perm_p": 0.408, "feature_names": ["L5:E12:N1023"],
           "routing_baseline": {"llr": 91.2, "perm_p": 0.537}}
}
```

Feature names use `L{layer}:E{expert}:N{neuron}` for routed experts and
`L{layer}:S:N{neuron}` for shared experts.

With `--routing-analysis` the JSON additionally carries `routing_analysis` (per-layer usage
deltas with permutation p-values, gate entropy, Jaccard churn, and `cett_cross_check` against
the fitted probe); with `--counterfactual` it carries `counterfactual` (`rescue`,
`corruption`, `random` rows per dose, `self_swap` gate, donor matching).

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

On macOS, MPS loads go through CPU first and are then moved to the device (`model.to("mps")`). Loading directly with `device_map="mps"` can hang when the load forces a dtype conversion (concurrent async materialization onto MPS, [transformers#48029](https://github.com/huggingface/transformers/issues/48029)); `HF_DEACTIVATE_ASYNC_LOAD=1` is the equivalent escape hatch.

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
