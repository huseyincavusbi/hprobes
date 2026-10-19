"""Unit tests for hprobes CLI helpers"""

import json
import tempfile

import pytest

from hprobes.cli import _uses_mps, detect_format, format_keys, load_samples


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


class TestUsesMps:
    def test_explicit_mps(self):
        assert _uses_mps("mps") is True

    def test_cpu(self):
        assert _uses_mps("cpu") is False

    def test_cuda(self):
        assert _uses_mps("cuda") is False

    def test_auto_with_mps(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert _uses_mps("auto") is True

    def test_auto_without_mps(self, monkeypatch):
        import torch

        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert _uses_mps("auto") is False


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
            resp_p.add_argument(
                "--max-new-tokens",
                type=int,
                default=None,
                dest="max_new_tokens_consistency",
            )
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

    def test_responses_max_new_tokens(self):
        args = self._parse(["responses", "--model", "m", "--data", "d", "--max-new-tokens", "77"])
        assert args.max_new_tokens_consistency == 77

    def test_responses_max_new_tokens_default(self):
        args = self._parse(["responses", "--model", "m", "--data", "d"])
        assert args.max_new_tokens_consistency is None

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

    def test_run_parser_accepts_readout(self):
        args = self._parse(["run", "--model", "m", "--data", "d", "--readout", "logits"])
        assert args.readout == "logits"
        args = self._parse(["run", "--model", "m", "--data", "d"])
        assert args.readout == "both"

    def test_run_parser_accepts_no_predictions(self):
        args = self._parse(["run", "--model", "m", "--data", "d", "--no-predictions"])
        assert args.no_predictions is True
        args = self._parse(["run", "--model", "m", "--data", "d"])
        assert args.no_predictions is False

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


class TestParseLayers:
    def test_comma_separated_sorted_unique(self):
        from hprobes.cli import _parse_layers

        assert _parse_layers("8,0,8,4") == [0, 4, 8]

    def test_all_and_empty_return_none(self):
        from hprobes.cli import _parse_layers

        assert _parse_layers("all") is None
        assert _parse_layers("ALL") is None

    def test_rejects_junk_and_negative(self):
        import argparse

        from hprobes.cli import _parse_layers

        with pytest.raises(argparse.ArgumentTypeError):
            _parse_layers("0,x")
        with pytest.raises(argparse.ArgumentTypeError):
            _parse_layers("-1")


class TestMoeCommand:
    @staticmethod
    def _parse(monkeypatch, argv):
        import sys

        from hprobes import cli

        captured = {}
        monkeypatch.setattr(cli, "cmd_moe", lambda args: captured.update(vars(args)))
        monkeypatch.setattr(sys, "argv", ["hprobes"] + argv)
        cli.main()
        return captured

    def test_defaults(self, monkeypatch):
        args = self._parse(monkeypatch, ["moe", "--model", "m", "--data", "d.jsonl"])
        assert args["top_k"] == 2000
        assert args["readout"] == "logits"
        assert args["score"] == "mean_abs"
        assert args["routing_baseline"] is True
        assert args["normalize_write"] is False
        assert args["fusion"] == "concat"
        assert args["layers"] is None
        assert args["min_active"] == 2
        assert args["per_expert_top_n"] == 64

    def test_flags_wire_through(self, monkeypatch):
        args = self._parse(
            monkeypatch,
            [
                "moe",
                "--model",
                "m",
                "--data",
                "d.jsonl",
                "--score",
                "expected",
                "--min-active",
                "5",
                "--per-expert-top-n",
                "16",
                "--top-k",
                "0",
                "--layers",
                "8,0,4",
                "--normalize-write",
                "--no-routing-baseline",
                "--fusion",
                "neuron",
            ],
        )
        assert args["score"] == "expected"
        assert args["min_active"] == 5
        assert args["per_expert_top_n"] == 16
        assert args["top_k"] == 0
        assert args["layers"] == [0, 4, 8]
        assert args["normalize_write"] is True
        assert args["routing_baseline"] is False
        assert args["fusion"] == "neuron"

    def test_top_k_auto_is_rejected(self, monkeypatch):
        import sys

        from hprobes import cli

        monkeypatch.setattr(
            sys,
            "argv",
            ["hprobes", "moe", "--model", "m", "--data", "d.jsonl", "--top-k", "auto"],
        )
        with pytest.raises(SystemExit, match="dense-only"):
            cli.main()

    def test_readout_generate_is_rejected(self, monkeypatch):
        import sys

        from hprobes import cli

        monkeypatch.setattr(
            sys,
            "argv",
            ["hprobes", "moe", "--model", "m", "--data", "d.jsonl", "--readout", "generate"],
        )
        with pytest.raises(SystemExit, match="logits"):
            cli.main()

    def test_cmd_moe_smoke_on_fake_model(self, monkeypatch, tmp_path, capsys):
        import sys

        import _fake_moe
        from hprobes import cli
        from test_probe import TOK

        model = _fake_moe._FakeMoEModel()
        samples = [
            {"question": "Q0?", "options": ["aa", "bb"], "answer": 0},
            {"question": "Q1?", "options": ["cc", "dd"], "answer": 1},
        ]
        monkeypatch.setattr(cli, "load_samples", lambda data, n: samples)
        monkeypatch.setattr(cli, "_load_model", lambda args: (TOK, model))

        out = tmp_path / "moe_run"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "hprobes",
                "moe",
                "--model",
                "fake/moe",
                "--data",
                "fake.jsonl",
                "--samples",
                "2",
                "--layers",
                "0",
                "--per-expert-top-n",
                "4",
                "--min-active",
                "1",
                "--top-k",
                "16",
                "--no-routing-baseline",
                "--fusion",
                "neuron",
                "--stability",
                "--output",
                str(out),
            ],
        )
        cli.main()

        out_text = capsys.readouterr().out
        assert "--stability (dense-only; ignored in MoE mode)" in out_text

        data = json.loads(out.with_suffix(".json").read_text())
        assert data["mode"] == "moe"
        assert data["n_samples"] == 2
        assert data["config"]["score"] == "mean_abs"
        assert data["config"]["layers"] == [0]
        assert data["config"]["routing_baseline"] is False
        assert data["config"]["fusion"] == "neuron"
        assert data["moe"]["n_samples"] == 2


class TestMoeExpertsImplementation:
    def test_parser_default_and_override(self, monkeypatch):
        args = TestMoeCommand._parse(monkeypatch, ["moe", "--model", "m", "--data", "d.jsonl"])
        assert args["experts_implementation"] == "auto"
        args = TestMoeCommand._parse(
            monkeypatch,
            [
                "moe",
                "--model",
                "m",
                "--data",
                "d.jsonl",
                "--experts-implementation",
                "eager",
            ],
        )
        assert args["experts_implementation"] == "eager"

    @staticmethod
    def _fake_transformers(captured):
        import types

        class _AutoModel:
            @staticmethod
            def from_pretrained(name, **kw):
                captured.update(kw)
                return types.SimpleNamespace(eval=lambda: None)

        class _AutoTokenizer:
            @staticmethod
            def from_pretrained(name, **kw):
                return object()

        return types.SimpleNamespace(AutoModelForCausalLM=_AutoModel, AutoTokenizer=_AutoTokenizer)

    @staticmethod
    def _tiny_args(experts):
        import argparse

        return argparse.Namespace(
            model="fake/model",
            dtype="auto",
            trust_remote_code=False,
            device="cpu",
            attn_implementation="auto",
            experts_implementation=experts,
        )

    def test_load_model_passes_eager(self, monkeypatch):
        import sys

        from hprobes import cli

        captured = {}
        monkeypatch.setitem(sys.modules, "transformers", self._fake_transformers(captured))
        cli._load_model(self._tiny_args("eager"))
        assert captured["experts_implementation"] == "eager"

    def test_load_model_omits_flag_for_auto(self, monkeypatch):
        import sys

        from hprobes import cli

        captured = {}
        monkeypatch.setitem(sys.modules, "transformers", self._fake_transformers(captured))
        cli._load_model(self._tiny_args("auto"))
        assert "experts_implementation" not in captured
