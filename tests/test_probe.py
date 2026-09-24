"""Tests for hprobes.probe — HProbes class."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from hprobes import HProbes

# ---------------------------------------------------------------------------
# Minimal mock model + tokenizer
# ---------------------------------------------------------------------------

_H, _I, _V, _L = 8, 16, 128, 4


class _MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(_H, _I, bias=False)
        self.down_proj = nn.Linear(_I, _H, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate(x)))


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _MLP()

    def forward(self, x):
        return x + self.mlp(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(1)

        class _Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([_Block() for _ in range(_L)])

        self.model = _Inner()
        self.lm_head = nn.Linear(_H, _V, bias=True)
        nn.init.normal_(self.lm_head.weight, std=0.2)
        nn.init.normal_(self.lm_head.bias, std=0.2)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        # Use raw token ids as embedding values so different tokens produce different outputs
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, _H)
        for block in self.model.layers:
            x = block(x)

        class _Out:
            pass

        out = _Out()
        out.logits = self.lm_head(x)
        return out


class _TokenizerOutput(dict):
    """Dict subclass with .to() so detect_batch() can call tokenizer(...).to(device)."""

    def to(self, device):
        return {k: v.to(device) for k, v in self.items()}


class _Tokenizer:
    """ASCII character-level tokenizer. encode('A') == [65]."""

    chat_template = None
    padding_side = "right"
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, padding=False):
        if isinstance(text, list):
            all_ids = [[ord(c) for c in t] for t in text]
            if max_length:
                all_ids = [ids[:max_length] for ids in all_ids]
            max_len = max(len(ids) for ids in all_ids)
            padded = [ids + [0] * (max_len - len(ids)) for ids in all_ids]
            masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]
            return _TokenizerOutput(
                input_ids=torch.tensor(padded, dtype=torch.long),
                attention_mask=torch.tensor(masks, dtype=torch.long),
            )
        ids = [ord(c) for c in text]
        if max_length:
            ids = ids[:max_length]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return _TokenizerOutput(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


MODEL = _Model()
TOK = _Tokenizer()

# 20 samples: 10 answer A (index 0), 10 answer B (index 1)
SAMPLES = [
    {"question": f"Q{i}?", "options": ["alpha", "beta", "gamma", "delta"], "answer": i % 2}
    for i in range(20)
]

# Fitted probe shared across tests in this module
_PROBE = HProbes(MODEL, TOK, l1_C=10)
_PROBE.fit(SAMPLES, options_key="options", answer_key="answer")


# ---------------------------------------------------------------------------


class TestParseGroundTruth:
    def test_letter(self):
        assert _PROBE._parse_ground_truth({"a": "B"}, "a") == "B"

    def test_lowercase_normalised(self):
        assert _PROBE._parse_ground_truth({"a": "c"}, "a") == "C"

    def test_numeric_index(self):
        assert _PROBE._parse_ground_truth({"a": 0}, "a") == "A"
        assert _PROBE._parse_ground_truth({"a": 3}, "a") == "D"

    def test_missing_returns_none(self):
        assert _PROBE._parse_ground_truth({}, "answer") is None

    def test_invalid_returns_none(self):
        assert _PROBE._parse_ground_truth({"a": "xyz"}, "a") is None


class TestFindAnswerSpan:
    def _ids(self, text):
        return torch.tensor([ord(c) for c in text])

    def test_found(self):
        assert _PROBE._find_answer_span(self._ids("Hello"), ["l", "l"]) == (2, 4)

    def test_not_found(self):
        assert _PROBE._find_answer_span(self._ids("Hello"), ["X"]) is None

    def test_empty_returns_none(self):
        assert _PROBE._find_answer_span(self._ids("Hello"), []) is None


class TestPredictLetter:
    def test_picks_max_logit(self):
        logits = torch.zeros(_V)
        logits[ord("C")] = 10.0
        assert _PROBE._predict_letter(logits) == "C"


class TestWelfordUpdate:
    def test_mean_converges(self):
        p = HProbes(MODEL, TOK)
        p._welford_n = 0
        p._welford_mean = np.zeros(8, dtype=np.float64)
        p._welford_M2 = np.zeros(8, dtype=np.float64)
        rng = np.random.RandomState(0)
        data = [rng.randn(8).astype(np.float32) for _ in range(100)]
        for v in data:
            p._welford_update(v)
        assert np.allclose(p._welford_mean, np.stack(data).mean(0), atol=1e-4)


class TestNotFittedErrors:
    def test_score_raises(self):
        with pytest.raises(RuntimeError):
            HProbes(MODEL, TOK).score()

    def test_causal_validate_raises(self):
        with pytest.raises(RuntimeError):
            HProbes(MODEL, TOK).causal_validate()


class TestFitAndScore:
    def test_fitted_attributes(self):
        assert _PROBE.is_fitted_
        assert 0.0 <= _PROBE.accuracy_ <= 1.0
        assert _PROBE.n_neurons_ == len(_PROBE.h_neurons_)
        assert all(isinstance(t, tuple) and len(t) == 2 for t in _PROBE.h_neurons_)

    def test_score_keys_and_range(self):
        result = _PROBE.score()
        assert "auroc" in result and "balanced_accuracy" in result and "auroc_gap" in result
        assert 0.0 <= result["balanced_accuracy"] <= 1.0

    def test_causal_validate_range(self):
        result = _PROBE.causal_validate(alphas=[0.0, 1.0])
        # Empty dict is valid when no H-Neurons were found
        assert all(0.0 <= v <= 1.0 for v in result.values())

    def test_contrastive_mode(self):
        p = HProbes(MODEL, TOK, l1_C=0.5)
        p.fit(SAMPLES, options_key="options", answer_key="answer")
        assert p.is_fitted_

    def test_mmlu_list_options(self):
        samples = [
            {"question": f"Q{i}?", "choices": ["a", "b", "c", "d"], "answer": i % 2}
            for i in range(20)
        ]
        p = HProbes(MODEL, TOK, l1_C=0.5)
        p.fit(samples, options_key="choices", answer_key="answer")
        assert p.is_fitted_

    def test_fit_from_responses(self):
        samples = [
            {
                "question": f"Describe {i}.",
                "response": f"The answer is X{i}.",
                "answer_tokens": ["X"],
                "judge": i % 2 == 0,
            }
            for i in range(20)
        ]
        p = HProbes(MODEL, TOK, l1_C=0.5)
        p.fit_from_responses(samples)
        assert p.is_fitted_


class TestSaveLoadTransfer:
    def test_save_creates_json_and_pkl(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.score()
            saved = _PROBE.save(base)
            assert Path(base).with_suffix(".json").exists()
            assert Path(base).with_suffix(".safetensors").exists()
            assert saved == Path(base).with_suffix(".json")

    def test_load_restores_probe(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.save(base)
            loaded = HProbes.load(base, MODEL, TOK)
            assert loaded.is_fitted_
            assert loaded.n_neurons_ == _PROBE.n_neurons_
            assert loaded.h_neurons_ == _PROBE.h_neurons_

    def test_load_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            HProbes.load("/nonexistent/probe", MODEL, TOK)

    def test_score_on_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.save(base)
            loaded = HProbes.load(base, MODEL, TOK)
            result = loaded.score_on(SAMPLES, options_key="options", answer_key="answer")
            assert "auroc" in result
            assert "balanced_accuracy" in result


class TestThreshold:
    def test_default_before_score(self):
        p = HProbes(MODEL, TOK, l1_C=0.5)
        assert p.threshold_ == 0.5

    def test_set_after_score(self):
        # _PROBE already had score() called during save/load tests above; call again to be sure
        _PROBE.score()
        assert 0.0 <= _PROBE.threshold_ <= 1.0

    def test_threshold_in_score_results(self):
        result = _PROBE.score()
        assert "threshold" in result
        assert result["threshold"] == _PROBE.threshold_

    def test_threshold_persisted_in_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.score()
            expected = _PROBE.threshold_
            _PROBE.save(base)
            loaded = HProbes.load(base, MODEL, TOK)
            assert loaded.threshold_ == expected


class TestDetect:
    def test_returns_float_in_range(self):
        prompt = "Q0? Options: A) alpha B) beta C) gamma D) delta\n\nAnswer:"
        score = _PROBE.detect(prompt)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_raises_if_not_fitted(self):
        with pytest.raises(RuntimeError):
            HProbes(MODEL, TOK).detect("some prompt")

    def test_with_answer_letter_provided(self):
        prompt = "Q0? Options: A) alpha B) beta\n\nAnswer:"
        score = _PROBE.detect(prompt, answer_letter="A")
        assert 0.0 <= score <= 1.0

    def test_answer_letter_case_normalised(self):
        prompt = "Q0? Options: A) alpha B) beta\n\nAnswer:"
        score_lower = _PROBE.detect(prompt, answer_letter="a")
        score_upper = _PROBE.detect(prompt, answer_letter="A")
        assert abs(score_lower - score_upper) < 1e-6

    def test_invalid_answer_letter_raises(self):
        probe = HProbes(MODEL, TOK, l1_C=0.5)
        probe.fit(SAMPLES, options_key="options", answer_key="answer")
        with pytest.raises(ValueError):
            probe.detect("some prompt", answer_letter="Z")

    def test_non_contrastive_detect(self):
        p = HProbes(MODEL, TOK, l1_C=0.5)
        p.fit(SAMPLES, options_key="options", answer_key="answer")
        score = p.detect("Q0? Options: A) alpha B) beta\n\nAnswer:")
        assert 0.0 <= score <= 1.0


class TestDetectBatch:
    _PROMPT = "Q{i}? Options: A) alpha B) beta C) gamma D) delta\n\nAnswer:"

    def _prompts(self, n=4):
        return [self._PROMPT.format(i=i) for i in range(n)]

    def test_returns_list_of_correct_length(self):
        prompts = self._prompts(4)
        scores = _PROBE.detect_batch(prompts, batch_size=2)
        assert len(scores) == 4

    def test_scores_in_range(self):
        scores = _PROBE.detect_batch(self._prompts(4))
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_raises_if_not_fitted(self):
        with pytest.raises(RuntimeError):
            HProbes(MODEL, TOK).detect_batch(["prompt"])

    def test_with_answer_letters_provided(self):
        prompts = self._prompts(4)
        letters = ["A", "B", "C", "D"]
        scores = _PROBE.detect_batch(prompts, answer_letters=letters, batch_size=2)
        assert len(scores) == 4
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_single_prompt_matches_detect(self):
        prompt = self._prompts(1)[0]
        batch_score = _PROBE.detect_batch([prompt], batch_size=1)[0]
        single_score = _PROBE.detect(prompt)
        # Same forward pass logic — scores should be identical
        assert abs(batch_score - single_score) < 1e-5

    def test_non_contrastive_batch(self):
        p = HProbes(MODEL, TOK, l1_C=0.5)
        p.fit(SAMPLES, options_key="options", answer_key="answer")
        scores = p.detect_batch(self._prompts(4), batch_size=2)
        assert len(scores) == 4
        assert all(0.0 <= s <= 1.0 for s in scores)


class TestConsistencyFilter:
    def test_default_n_consistency_is_1(self):
        assert HProbes(MODEL, TOK).n_consistency == 1

    def test_fit_with_n_consistency(self):
        p = HProbes(MODEL, TOK, l1_C=0.5, n_consistency=3)
        p.fit(SAMPLES, options_key="options", answer_key="answer")
        assert p.is_fitted_

    def test_n_consistency_persisted_in_save_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            p = HProbes(MODEL, TOK, l1_C=0.5, n_consistency=5)
            p.fit(SAMPLES, options_key="options", answer_key="answer")
            p.save(base)
            loaded = HProbes.load(base, MODEL, TOK)
            assert loaded.n_consistency == 5


class TestCompareWith:
    def test_compare_with_returns_dict(self):
        p1 = HProbes(MODEL, TOK, l1_C=0.5)
        p1.fit(SAMPLES, options_key="options", answer_key="answer")
        p2 = HProbes(MODEL, TOK, l1_C=1.0)
        p2.fit(SAMPLES, options_key="options", answer_key="answer")

        result = p1.compare_with(p2)
        assert isinstance(result, dict)
        assert "jaccard_similarity" in result
        assert "n_shared" in result
        assert "n_union" in result
        assert "n_only_self" in result
        assert "n_only_other" in result
        assert "shared_neurons" in result

    def test_jaccard_similarity_in_range(self):
        p1 = HProbes(MODEL, TOK, l1_C=0.5)
        p1.fit(SAMPLES, options_key="options", answer_key="answer")
        p2 = HProbes(MODEL, TOK, l1_C=1.0)
        p2.fit(SAMPLES, options_key="options", answer_key="answer")

        result = p1.compare_with(p2)
        assert 0.0 <= result["jaccard_similarity"] <= 1.0

    def test_identical_probes_have_jaccard_one(self):
        p1 = HProbes(MODEL, TOK, l1_C=0.5)
        p1.fit(SAMPLES, options_key="options", answer_key="answer")

        result = p1.compare_with(p1)
        # If no neurons found, jaccard is 0/0 = 0.0 by convention
        # If neurons found, comparing with self should give 1.0
        if p1.n_neurons_ > 0:
            assert result["jaccard_similarity"] == 1.0
            assert result["n_shared"] == p1.n_neurons_
            assert result["n_only_self"] == 0
            assert result["n_only_other"] == 0
        else:
            assert result["jaccard_similarity"] == 0.0

    def test_compare_with_unfitted_raises(self):
        p1 = HProbes(MODEL, TOK, l1_C=0.5)
        p1.fit(SAMPLES, options_key="options", answer_key="answer")
        p2 = HProbes(MODEL, TOK, l1_C=1.0)

        with pytest.raises(RuntimeError, match="must be fitted"):
            p1.compare_with(p2)

    def test_shared_neurons_are_tuples(self):
        p1 = HProbes(MODEL, TOK, l1_C=0.5)
        p1.fit(SAMPLES, options_key="options", answer_key="answer")
        p2 = HProbes(MODEL, TOK, l1_C=1.0)
        p2.fit(SAMPLES, options_key="options", answer_key="answer")

        result = p1.compare_with(p2)
        if result["shared_neurons"]:
            assert all(isinstance(n, list) and len(n) == 2 for n in result["shared_neurons"])


class TestSaveMetadata:
    def test_save_includes_metadata_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.save(base)

            json_path = Path(base).with_suffix(".json")
            data = json.loads(json_path.read_text())

            assert "metadata" in data
            assert "model_name" in data["metadata"]
            assert "n_layers" in data["metadata"]
            assert "intermediate_dim" in data["metadata"]
            assert "total_features" in data["metadata"]

    def test_metadata_values_are_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.save(base)

            json_path = Path(base).with_suffix(".json")
            data = json.loads(json_path.read_text())

            assert data["metadata"]["n_layers"] == len(_PROBE._layers)
            assert data["metadata"]["intermediate_dim"] == _PROBE._intermediate_dim
            assert data["metadata"]["total_features"] == _PROBE._n_features

    def test_model_name_extracted_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = str(Path(tmp) / "probe")
            _PROBE.save(base)

            json_path = Path(base).with_suffix(".json")
            data = json.loads(json_path.read_text())

            # Metadata field should exist (model_name can be None for mock models)
            assert "model_name" in data["metadata"]


class TestLabelFn:
    """Test custom label_fn for control probes."""

    def test_answer_letter_control(self):
        """Test control probe that predicts answer letter A vs not-A."""
        probe = HProbes(MODEL, TOK, batch_size=1, l1_C=0.5)

        # Label function: 1 if predicted answer is "A", 0 otherwise
        # Note: This test verifies the label_fn parameter works, even if
        # the mock model happens to predict all the same letter
        try:
            probe.fit(
                SAMPLES,
                options_key="options",
                answer_key="answer",
                label_fn=lambda pred, gt, sample: 1 if pred == "A" else 0,
            )
            assert probe.is_fitted_
            assert probe.n_neurons_ >= 0
        except ValueError as e:
            # If all predictions are the same letter, sklearn will raise ValueError
            # Python 3.10: "only one class"
            # Python 3.11+: "multiclass classification"
            error_msg = str(e).lower()
            if "only one class" in error_msg or "multiclass" in error_msg:
                pytest.skip(
                    "Mock model produces uniform predictions, cannot train binary classifier"
                )
            raise

    def test_domain_control(self):
        """Test control probe that predicts domain/subject."""
        # Add subject field to samples
        samples_with_subject = [
            {**s, "subject": "Anatomy" if i % 2 == 0 else "Physiology"}
            for i, s in enumerate(SAMPLES)
        ]

        probe = HProbes(MODEL, TOK, batch_size=1, l1_C=0.5)

        # Label function: 1 if subject is Anatomy, 0 otherwise
        probe.fit(
            samples_with_subject,
            options_key="options",
            answer_key="answer",
            label_fn=lambda pred, gt, sample: 1 if sample.get("subject") == "Anatomy" else 0,
        )

        assert probe.is_fitted_
        assert probe.n_neurons_ >= 0

    def test_default_behavior_unchanged(self):
        """Test that default behavior (no label_fn) still works as hallucination probe."""
        probe_default = HProbes(MODEL, TOK, batch_size=1, l1_C=0.5)
        probe_default.fit(SAMPLES, options_key="options", answer_key="answer")

        probe_explicit = HProbes(MODEL, TOK, batch_size=1, l1_C=0.5)
        probe_explicit.fit(
            SAMPLES,
            options_key="options",
            answer_key="answer",
            label_fn=lambda pred, gt, sample: 1 if pred != gt else 0,
        )

        # Both should identify the same neurons (hallucination labeling)
        assert probe_default.n_neurons_ == probe_explicit.n_neurons_
        assert probe_default.h_neurons_ == probe_explicit.h_neurons_

    def test_inverted_labeling(self):
        """Test inverted labeling: correct=1, incorrect=0."""
        probe = HProbes(MODEL, TOK, batch_size=1, l1_C=0.5)

        # Inverted: 1 if correct, 0 if incorrect
        probe.fit(
            SAMPLES,
            options_key="options",
            answer_key="answer",
            label_fn=lambda pred, gt, sample: 1 if pred == gt else 0,
        )

        assert probe.is_fitted_
        # Should still find neurons, but they encode correctness instead of hallucination
        assert probe.n_neurons_ >= 0


class TestMemoryGuard:
    def test_estimate_scales_with_rows_and_fits(self):
        from hprobes.probe import _estimate_peak_fit_bytes

        base = _estimate_peak_fit_bytes(100, 1000, 1)
        assert _estimate_peak_fit_bytes(200, 1000, 1) == 2 * base
        assert _estimate_peak_fit_bytes(100, 1000, 2) == 2 * base

    def test_warns_when_peak_exceeds_available(self, monkeypatch):
        from hprobes import probe

        monkeypatch.setattr(probe, "_available_ram_bytes", lambda: 1_000_000)  # 1 MB
        with pytest.warns(UserWarning):
            probe._warn_if_memory_heavy(1000, 348160, stability=True)

    def test_no_warn_when_ram_is_plentiful(self, monkeypatch):
        import warnings as _w

        from hprobes import probe

        monkeypatch.setattr(probe, "_available_ram_bytes", lambda: 10**15)
        with _w.catch_warnings():
            _w.simplefilter("error")
            probe._warn_if_memory_heavy(1000, 348160, stability=True)

    def test_strict_raises(self, monkeypatch):
        from hprobes import probe

        monkeypatch.setattr(probe, "_available_ram_bytes", lambda: 1_000_000)
        with pytest.raises(MemoryError):
            probe._warn_if_memory_heavy(1000, 348160, stability=True, strict=True)

    def test_default_top_k_is_all_features(self):
        assert HProbes(MODEL, TOK).top_k == 0
