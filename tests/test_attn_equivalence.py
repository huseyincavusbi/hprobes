"""Layer-0 / Layer-1 / Layer-3 equivalence harness (Phase 2).

Verifies that the opt-in attention implementations and inference-mode/batching
paths preserve scientific value, using a real sdpa-capable model (SmolLM2-135M).

- Layer 0 (static): only the attention implementation changes; weights/dtype/
  device are bit-identical; model stays in eval mode.
- Layer 1 (tensor): eager vs sdpa logits + down_proj activations within
  empirically-measured tolerances, and prediction (argmax) stability.
- Layer 3 (RNG): greedy generation is deterministic; seeded sampling is
  run-to-run reproducible.
"""

import pytest
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None

from hprobes.cett import get_mlp_down_proj

# SmolLM2 is Llama-arch → supports sdpa. Small enough for CPU CI.
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

# Empirically measured (see NEXT.md Phase 2): fp32 CPU eager-vs-sdpa single-seq
# logits max_abs ~2.5e-1, down_proj z rel ~3.5e-2. Tolerance below is generous
# but still catches real regressions (mask bugs, wrong dtype, etc.).
LOGIT_ATOL, LOGIT_RTOL = 5e-1, 1e-1
ACT_ATOL, ACT_RTOL = 2e-1, 1e-1

pytestmark = pytest.mark.skipif(AutoModelForCausalLM is None, reason="transformers not installed")


@pytest.fixture(scope="module")
def smol_setup():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load(attn, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, attn_implementation=attn)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Layer 0 — static checks
# ---------------------------------------------------------------------------


class TestLayer0Static:
    def test_only_attn_config_changes(self, smol_setup):
        eager = _load("eager", smol_setup)
        sdpa = _load("sdpa", smol_setup)

        cfg_e = eager.config.to_dict()
        cfg_s = sdpa.config.to_dict()
        keys = set(cfg_e) | set(cfg_s)
        diff = {k: (cfg_e.get(k), cfg_s.get(k)) for k in keys if cfg_e.get(k) != cfg_s.get(k)}
        allowed = {
            "_attn_implementation",
            "_attn_implementation_internal",
            "attn_implementation",
            "torch_dtype",
            "_name_or_path",
        }
        assert set(diff) - allowed == set(), f"unexpected config diff: {diff}"

    def test_weights_bit_identical(self, smol_setup):
        eager = _load("eager", smol_setup)
        sdpa = _load("sdpa", smol_setup)
        for (n1, p1), (n2, p2) in zip(eager.named_parameters(), sdpa.named_parameters()):
            assert n1 == n2, f"parameter order differs: {n1} vs {n2}"
            assert p1.dtype == p2.dtype
            assert torch.equal(p1, p2), f"weight drift: {n1}"

    def test_eval_mode_and_no_dropout(self, smol_setup):
        eager = _load("eager", smol_setup)
        sdpa = _load("sdpa", smol_setup)
        assert not eager.training
        assert not sdpa.training


# ---------------------------------------------------------------------------
# Layer 1 — tensor-level equivalence (eager vs sdpa)
# ---------------------------------------------------------------------------


class TestLayer1Tensor:
    def _collect(self, model, tokenizer, prompt):
        enc = tokenizer(prompt, return_tensors="pt")
        zs, hs = {}, {}
        handles = []
        for li in [0, 5, 10]:
            dp = get_mlp_down_proj(model, li)
            handles.append(
                dp.register_forward_hook(
                    (
                        lambda idx: lambda mod, inp, out: (
                            zs.__setitem__(idx, inp[0][0, -1].detach().float()),
                            hs.__setitem__(idx, out[0, -1].detach().float()),
                            out,
                        )[2]
                    )(li)
                )
            )
        with torch.inference_mode():
            out = model(**enc)
        for h in handles:
            h.remove()
        return out.logits[0, -1].detach().float(), zs, hs

    def test_logits_and_activations_close(self, smol_setup):
        prompt = "The capital of France is"
        le, ze, he = self._collect(_load("eager", smol_setup), smol_setup, prompt)
        ls, zs, hs = self._collect(_load("sdpa", smol_setup), smol_setup, prompt)

        torch.testing.assert_close(ls, le, atol=LOGIT_ATOL, rtol=LOGIT_RTOL, check_stride=False)
        for li in [0, 5, 10]:
            torch.testing.assert_close(zs[li], ze[li], atol=ACT_ATOL, rtol=ACT_RTOL)
            torch.testing.assert_close(hs[li], he[li], atol=ACT_ATOL, rtol=ACT_RTOL)

    def test_prediction_stable(self, smol_setup):
        prompt = "The capital of France is"
        le, _, _ = self._collect(_load("eager", smol_setup), smol_setup, prompt)
        ls, _, _ = self._collect(_load("sdpa", smol_setup), smol_setup, prompt)
        assert le.argmax().item() == ls.argmax().item()


# ---------------------------------------------------------------------------
# Layer 3 — determinism / RNG
# ---------------------------------------------------------------------------


class TestLayer3Determinism:
    def test_greedy_generation_deterministic(self, smol_setup):
        model = _load("eager", smol_setup)
        prompt = "The capital of France is"
        enc = smol_setup(prompt, return_tensors="pt")
        with torch.inference_mode():
            o1 = model.generate(**enc, max_new_tokens=10, do_sample=False)
            o2 = model.generate(**enc, max_new_tokens=10, do_sample=False)
        assert torch.equal(o1, o2)

    def test_seeded_sampling_reproducible(self, smol_setup):
        model = _load("eager", smol_setup)
        prompt = "The capital of France is"
        enc = smol_setup(prompt, return_tensors="pt")

        outs = []
        for _ in range(2):
            torch.manual_seed(1234)
            with torch.inference_mode():
                out = model.generate(
                    **enc, max_new_tokens=10, do_sample=True, temperature=1.0, top_k=50
                )
            outs.append(out)
        assert torch.equal(outs[0], outs[1])
