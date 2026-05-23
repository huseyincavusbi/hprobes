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
