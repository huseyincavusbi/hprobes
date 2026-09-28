"""Unit tests for hprobes CLI helpers"""

import json
import tempfile

import pytest

from hprobes.cli import detect_format, format_keys, load_samples


_MMLU_SAMPLE = {
    "question": "Which organ produces insulin?",
    "choices": ["Liver", "Pancreas", "Kidney", "Spleen"],
    "answer": 1,
    "subject": "anatomy",
}

_MEDQA_SAMPLE = {
    "question": "A patient presents with chest pain. What is the first test?",
    "options": {"A": "ECG", "B": "CXR", "C": "Echo", "D": "CT"},
    "answer_idx": "A",
    "meta_info": "step2",
}

_MEDMCQA_SAMPLE = {
    "question": "Drug of choice for TB meningitis?",
    "options": {"A": "Rifampicin", "B": "INH", "C": "Pyrazinamide", "D": "Ethambutol"},
    "cop": 0,
    "exp": "Rifampicin crosses BBB",
}

_UNKNOWN_SAMPLE = {
    "question": "What is 2+2?",
    "answer": "4",
}


class TestDetectFormat:
    def test_mmlu(self):
        assert detect_format(_MMLU_SAMPLE) == "mmlu"

    def test_medqa(self):
        assert detect_format(_MEDQA_SAMPLE) == "medqa"

    def test_medmcqa(self):
        assert detect_format(_MEDMCQA_SAMPLE) == "medmcqa"

    def test_unknown_returns_none(self):
        assert detect_format(_UNKNOWN_SAMPLE) is None

    def test_mmlu_requires_int_answer(self):
        # choices + string answer should NOT be detected as mmlu
        s = {**_MMLU_SAMPLE, "answer": "B"}
        assert detect_format(s) != "mmlu"


class TestFormatKeys:
    def test_mmlu_keys(self):
        assert format_keys("mmlu") == ("choices", "answer")

    def test_medqa_keys(self):
        assert format_keys("medqa") == ("options", "answer_idx")

    def test_medmcqa_keys(self):
        assert format_keys("medmcqa") == ("options", "cop")

    def test_unknown_raises(self):
        import pytest

        with pytest.raises(ValueError, match="Unknown format"):
            format_keys("foobar")


class TestLoadSamples:
    def test_jsonl(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            for s in [_MMLU_SAMPLE, _MEDQA_SAMPLE, _MEDMCQA_SAMPLE]:
                f.write(json.dumps(s) + "\n")
            path = f.name

        samples = load_samples(path, n=10)
        assert len(samples) == 3
        assert samples[0]["question"] == _MMLU_SAMPLE["question"]

    def test_jsonl_respects_n(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
            for _ in range(20):
                f.write(json.dumps(_MMLU_SAMPLE) + "\n")
            path = f.name

        samples = load_samples(path, n=5)
        assert len(samples) == 5

    def test_json_list(self):
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            json.dump([_MMLU_SAMPLE, _MEDQA_SAMPLE], f)
            path = f.name

        samples = load_samples(path, n=10)
        assert len(samples) == 2

    def test_missing_file_exits(self):
        import pytest

        with pytest.raises(SystemExit):
            load_samples("/nonexistent/path.jsonl", n=10)

    def test_unsupported_extension_exits(self):
        with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
            f.write("a,b,c\n")
            path = f.name

        with pytest.raises(SystemExit):
            load_samples(path, n=10)


class TestCLIArgParsing:
    """Smoke-test that all subcommands accept their new flags without error."""

    def _parse(self, argv):
        import argparse

        # Patch sys.argv and capture parse result via argparse internals
        import sys

        old = sys.argv
        sys.argv = ["hprobes"] + argv
        try:
            from hprobes.cli import _add_common_model_args, _add_common_probe_args

            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers(dest="command")

            run_p = subparsers.add_parser("run")
            run_p.add_argument("--model", default="m")
            run_p.add_argument("--data", default="d")
            run_p.add_argument("--format", default="auto")
            run_p.add_argument("--samples", type=int, default=100)
            run_p.add_argument("--no-contrastive", action="store_true", dest="no_contrastive")
            run_p.add_argument("--output", default=None)
            _add_common_model_args(run_p)
            _add_common_probe_args(run_p)

            resp_p = subparsers.add_parser("responses")
            resp_p.add_argument("--model", default="m")
            resp_p.add_argument("--data", default="d")
            resp_p.add_argument("--samples", type=int, default=100)
            resp_p.add_argument("--question-key", default="question", dest="question_key")
            resp_p.add_argument("--response-key", default="response", dest="response_key")
            resp_p.add_argument(
                "--answer-tokens-key", default="answer_tokens", dest="answer_tokens_key"
            )
            resp_p.add_argument("--label-key", default="judge", dest="label_key")
            resp_p.add_argument("--aggregation", default="mean")
            resp_p.add_argument("--output", default=None)
            _add_common_model_args(resp_p)
            _add_common_probe_args(resp_p)

            transfer_p = subparsers.add_parser("transfer")
            transfer_p.add_argument("--probe", default="p")
            transfer_p.add_argument("--model", default="m")
            transfer_p.add_argument("--data", default="d")
            transfer_p.add_argument("--format", default="auto")
            transfer_p.add_argument("--samples", type=int, default=100)
            transfer_p.add_argument("--output", default=None)
            transfer_p.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
            _add_common_model_args(transfer_p)

            return parser.parse_args(argv)
        finally:
            sys.argv = old

    def test_run_no_contrastive_flag(self):
        args = self._parse(["run", "--model", "m", "--data", "d", "--no-contrastive"])
        assert args.no_contrastive is True

    def test_run_contrastive_default(self):
        args = self._parse(["run", "--model", "m", "--data", "d"])
        assert args.no_contrastive is False

    def test_run_layer_stride(self):
        args = self._parse(["run", "--model", "m", "--data", "d", "--layer-stride", "2"])
        assert args.layer_stride == 2

    def test_run_validation_split(self):
        args = self._parse(["run", "--model", "m", "--data", "d", "--validation-split", "0.3"])
        assert args.validation_split == pytest.approx(0.3)

    def test_run_max_tokens(self):
        args = self._parse(["run", "--model", "m", "--data", "d", "--max-tokens", "512"])
        assert args.max_tokens == 512

    def test_run_alphas(self):
        args = self._parse(["run", "--model", "m", "--data", "d", "--alphas", "0.0,1.0,2.0"])
        assert args.alphas == "0.0,1.0,2.0"

    def test_responses_keys(self):
        args = self._parse(
            [
                "responses",
                "--model",
                "m",
                "--data",
                "d",
                "--question-key",
                "q",
                "--response-key",
                "r",
                "--answer-tokens-key",
                "toks",
                "--label-key",
                "correct",
                "--aggregation",
                "max",
            ]
        )
        assert args.question_key == "q"
        assert args.response_key == "r"
        assert args.answer_tokens_key == "toks"
        assert args.label_key == "correct"
        assert args.aggregation == "max"

    def test_transfer_output(self):
        args = self._parse(
            ["transfer", "--probe", "p", "--model", "m", "--data", "d", "--output", "out"]
        )
        assert args.output == "out"

    def test_transfer_max_tokens(self):
        args = self._parse(
            ["transfer", "--probe", "p", "--model", "m", "--data", "d", "--max-tokens", "512"]
        )
        assert args.max_tokens == 512

    def test_run_parser_accepts_auto_top_k_and_safety(self):
        args = self._parse(
            [
                "run",
                "--model",
                "m",
                "--data",
                "d",
                "--top-k",
                "auto",
                "--auto-safety",
                "0.5",
            ]
        )
        assert args.top_k == "auto"
        assert args.auto_safety == 0.5


class TestParseTopK:
    def test_auto_aliases(self):
        from hprobes.cli import _parse_top_k

        assert _parse_top_k("auto") == "auto"
        assert _parse_top_k("AUTO") == "auto"
        assert _parse_top_k("-1") == "auto"

    def test_numeric_values(self):
        from hprobes.cli import _parse_top_k

        assert _parse_top_k("0") == 0
        assert _parse_top_k("5000") == 5000

    def test_rejects_negative_and_junk(self):
        import argparse

        from hprobes.cli import _parse_top_k

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_top_k("-5")
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_top_k("abc")
