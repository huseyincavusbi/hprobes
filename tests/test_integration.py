import subprocess
import json
import tempfile
import os
import pytest
from transformers import AutoModelForCausalLM, AutoTokenizer
from hprobes import HProbes


@pytest.fixture(scope="module")
def gpt2_setup():
    """Download official GPT2 once for all integration tests."""
    model_id = "openai-community/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return model, tokenizer


def test_full_pipeline_with_gpt2(gpt2_setup):
    """
    Integration test using official GPT2 to verify:
    1. HProbes initialization
    2. fit() workflow (partial)
    3. Save/Load round-trip
    """
    model, tokenizer = gpt2_setup

    # Initialize probe
    probe = HProbes(model, tokenizer, l1_C=10, layer_stride=2)

    # Tiny dataset in MedQA format (dict options + letter answer)
    samples = [
        {
            "question": "What is the capital of France?",
            "options": {"A": "Paris", "B": "London"},
            "answer_idx": "A",
        },
        {"question": "What is 2+2?", "options": {"A": "3", "B": "4"}, "answer_idx": "B"},
        {
            "question": "What is the largest ocean?",
            "options": {"A": "Atlantic", "B": "Pacific"},
            "answer_idx": "B",
        },
        {
            "question": "What color is the sky?",
            "options": {"A": "Blue", "B": "Green"},
            "answer_idx": "A",
        },
        {"question": "What is H2O?", "options": {"A": "Water", "B": "Gold"}, "answer_idx": "A"},
        {
            "question": "How many legs does a cat have?",
            "options": {"A": "3", "B": "4"},
            "answer_idx": "B",
        },
    ]

    # Run fit
    # Note: HProbes.fit() expects options_key and answer_key for medqa
    probe.fit(samples, options_key="options", answer_key="answer_idx")

    assert probe.is_fitted_

    # 2. Save and Load
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        probe.save(tmp_path)
        # Load back
        new_probe = HProbes.load(tmp_path, model, tokenizer)

        # Verify fitted state and parameters match
        assert new_probe.is_fitted_
        assert new_probe.layer_stride == probe.layer_stride

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_fit_from_responses_batched_equals_single(gpt2_setup):
    """
    Real-model equivalence: fit_from_responses with batch_size>1 must produce
    the same feature statistics as the single-sample path, on real GPT-2.
    """
    import numpy as np

    model, tokenizer = gpt2_setup

    samples = []
    for i in range(12):
        q = f"What is the answer to question {i}?"
        r = f"The answer is number {i}."
        resp_ids = tokenizer(r, add_special_tokens=False)["input_ids"][:3]
        ans_toks = [tokenizer.decode([tid]) for tid in resp_ids]
        samples.append(
            {
                "question": q,
                "response": r,
                "answer_tokens": ans_toks,
                "judge": i % 2 == 0,
            }
        )

    p1 = HProbes(model, tokenizer, l1_C=1.0, layer_stride=2, batch_size=1)
    p1.fit_from_responses(samples)

    p2 = HProbes(model, tokenizer, l1_C=1.0, layer_stride=2, batch_size=4)
    p2.fit_from_responses(samples)

    assert p1.accuracy_ == p2.accuracy_
    assert p1._welford_n == p2._welford_n
    # Feature statistics must match within float32 tolerance
    assert np.allclose(p1._welford_mean, p2._welford_mean, atol=1e-4, rtol=1e-3)
    assert np.allclose(p1._welford_M2, p2._welford_M2, atol=1e-4, rtol=1e-3)


def test_pad_token_resizes_embeddings(gpt2_setup):
    """_ensure_pad_token must resize embeddings when adding a brand-new token."""
    model, tokenizer = gpt2_setup
    tokenizer.pad_token_id = None
    tokenizer.eos_token_id = None
    before = model.get_input_embeddings().num_embeddings
    probe = HProbes(model, tokenizer, l1_C=10, layer_stride=2, batch_size=2)
    assert probe.tokenizer.pad_token_id is not None, "pad token must be added"
    after = model.get_input_embeddings().num_embeddings
    assert after == len(tokenizer), "embeddings must be resized to the new vocab size"
    assert after > before, "new [PAD] token requires a new embedding row"


def test_score_on_batched_equals_single(gpt2_setup):
    """
    Real-model equivalence: score_on must produce identical results whether the
    samples are processed with batch_size=1 or batch_size=4.
    """

    model, tokenizer = gpt2_setup

    samples = []
    for i in range(12):
        q = f"What is the answer to question {i}?"
        r = f"The answer is number {i}."
        resp_ids = tokenizer(r, add_special_tokens=False)["input_ids"][:3]
        ans_toks = [tokenizer.decode([tid]) for tid in resp_ids]
        samples.append(
            {
                "question": q,
                "response": r,
                "answer_tokens": ans_toks,
                "judge": i % 2 == 0,
            }
        )

    p1 = HProbes(model, tokenizer, l1_C=1.0, layer_stride=2, batch_size=1)
    p1.fit_from_responses(samples)

    p2 = HProbes(model, tokenizer, l1_C=1.0, layer_stride=2, batch_size=4)
    p2.fit_from_responses(samples)

    r1 = p1.score_on(samples)
    r2 = p2.score_on(samples)
    assert r1["auroc"] == pytest.approx(r2["auroc"], abs=1e-6)
    assert r1["balanced_accuracy"] == pytest.approx(r2["balanced_accuracy"], abs=1e-6)


def test_cli_run_command(gpt2_setup):
    """
    Verifies the 'hprobes run' CLI command works end-to-end.
    """
    _, _ = gpt2_setup  # Ensure downloaded

    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        # Create a dummy dataset in MMLU format (list choices + int answer)
        # We need at least 2 samples for a minimal fit attempt
        items = [
            {"question": "Is GPT-2 official?", "choices": ["No", "Yes"], "answer": 1},
            {"question": "Is 1+1=2?", "choices": ["No", "Yes"], "answer": 1},
        ]
        for item in items:
            f.write(json.dumps(item) + "\n")
        f.flush()
        data_path = f.name

    # hprobes run creates <output>.json and <output>.pkl (or similar)
    output_base = "integration_test_output"

    import sys

    try:
        # Run CLI via subprocess
        # Note: CLI uses --samples instead of --n
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "hprobes.cli",
                "run",
                "--model",
                "openai-community/gpt2",
                "--data",
                data_path,
                "--output",
                output_base,
                "--samples",
                "2",
                "--mcq",
                "--format",
                "mmlu",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed with stderr: {result.stderr}"
        # The CLI saves .json results
        assert os.path.exists(f"{output_base}.json")

    finally:
        if os.path.exists(data_path):
            os.remove(data_path)
        # Cleanup any generated files
        for ext in [".json", ".pkl", ".safetensors"]:
            p = f"{output_base}{ext}"
            if os.path.exists(p):
                os.remove(p)
