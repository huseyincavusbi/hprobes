"""Numerical-equivalence tests for the transformers-optimizations branch.

Verifies that the batched forward paths introduced for performance
(forward_cett_dual_span_batch, scale_h_neurons_batch) produce results
equivalent to the original single-sample paths, so scientific value is
preserved. Uses the deterministic CPU mock models from test_cett.
"""

import torch

from hprobes.cett import (
    forward_cett_dual_span,
    forward_cett_dual_span_batch,
    scale_h_neurons,
    scale_h_neurons_batch,
)

from test_cett import _CausalLM, _tok, _I, _L


def _pad(tokens_list, device="cpu"):
    """Right-pad a list of single-sample token dicts into a batch dict."""
    ids = [t["input_ids"][0] for t in tokens_list]
    masks = [t["attention_mask"][0] for t in tokens_list]
    max_len = max(x.shape[0] for x in ids)
    batch_ids = torch.zeros(len(ids), max_len, dtype=torch.long)
    batch_masks = torch.zeros(len(ids), max_len, dtype=torch.long)
    for i, (x, m) in enumerate(zip(ids, masks)):
        batch_ids[i, : x.shape[0]] = x
        batch_masks[i, : m.shape[0]] = m
    return {"input_ids": batch_ids, "attention_mask": batch_masks}


class TestDualSpanBatchEquivalence:
    def setup_method(self):
        self.model = _CausalLM()
        self.layers = list(range(_L))
        from hprobes.cett import precompute_col_norms

        self.norms = precompute_col_norms(self.model, self.layers)

    def _assert_close(self, batch_tensor, single_list, atol=1e-5, rtol=1e-4):
        assert batch_tensor.shape[0] == len(single_list)
        for i, single in enumerate(single_list):
            torch.testing.assert_close(
                batch_tensor[i], single, atol=atol, rtol=rtol, check_stride=False
            )

    def test_answer_span_equivalent(self):
        texts = ["ABCDE", "ABC", "ABCDEFG"]
        toks = [_tok(t) for t in texts]
        spans = [(1, 3), (0, 2), (2, 5)]

        single_ans = [
            forward_cett_dual_span(self.model, t, s[0], s[1], self.layers, self.norms)[0]
            for t, s in zip(toks, spans)
        ]
        batch = _pad(toks)
        batch_ans, _ = forward_cett_dual_span_batch(
            self.model, batch, spans, self.layers, self.norms
        )
        self._assert_close(batch_ans, single_ans)

    def test_non_answer_span_equivalent(self):
        texts = ["ABCDE", "ABC", "ABCDEFG"]
        toks = [_tok(t) for t in texts]
        spans = [(1, 3), (0, 2), (2, 5)]

        single_oth = [
            forward_cett_dual_span(self.model, t, s[0], s[1], self.layers, self.norms)[1]
            for t, s in zip(toks, spans)
        ]
        batch = _pad(toks)
        _, batch_oth = forward_cett_dual_span_batch(
            self.model, batch, spans, self.layers, self.norms
        )
        self._assert_close(batch_oth, single_oth)

    def test_max_aggregation_equivalent(self):
        texts = ["ABCDE", "ABC", "ABCDEFG"]
        toks = [_tok(t) for t in texts]
        spans = [(1, 3), (0, 2), (2, 5)]

        single_ans = [
            forward_cett_dual_span(
                self.model, t, s[0], s[1], self.layers, self.norms, aggregation="max"
            )[0]
            for t, s in zip(toks, spans)
        ]
        batch = _pad(toks)
        batch_ans, _ = forward_cett_dual_span_batch(
            self.model, batch, spans, self.layers, self.norms, aggregation="max"
        )
        self._assert_close(batch_ans, single_ans)

    def test_full_span_other_empty(self):
        """Sample whose answer span covers the whole sequence → other is zeros."""
        toks = [_tok("ABCDE")]
        spans = [(0, 5)]
        single_oth = forward_cett_dual_span(self.model, toks[0], 0, 5, self.layers, self.norms)[1]
        batch = _pad(toks)
        _, batch_oth = forward_cett_dual_span_batch(
            self.model, batch, spans, self.layers, self.norms
        )
        torch.testing.assert_close(batch_oth[0], single_oth, atol=1e-5, rtol=1e-4)

    def test_max_aggregation_empty_answer_zeroed(self):
        """Empty answer span + aggregation='max' → zeros, never finfo.min sentinel."""
        toks = [_tok("ABCDE"), _tok("ABCDEFG")]
        spans = [(3, 3), (0, 0)]  # both answer spans empty
        batch = _pad(toks)
        ans, _ = forward_cett_dual_span_batch(
            self.model, batch, spans, self.layers, self.norms, aggregation="max"
        )
        assert torch.all(ans == 0), "empty answer spans must zero out, not propagate finfo.min"
        assert not torch.isinf(ans).any()


class TestScaleNeuronsBatchEquivalence:
    def setup_method(self):
        self.model = _CausalLM()
        self.layers = list(range(_L))
        self.neurons = [(0, 1), (1, 2), (2, 3), (3, 4)]

    def test_matches_single(self):
        texts = ["ABCDE", "ABC"]
        toks = [_tok(t) for t in texts]
        single = [scale_h_neurons(self.model, t, self.neurons, 0.5, self.layers) for t in toks]
        batch = _pad(toks)
        batch_out = scale_h_neurons_batch(self.model, batch, self.neurons, 0.5, self.layers)
        for i, s in enumerate(single):
            torch.testing.assert_close(batch_out[i], s, atol=1e-5, rtol=1e-4, check_stride=False)

    def test_alpha_one_is_identity(self):
        texts = ["ABCDE", "ABC"]
        toks = [_tok(t) for t in texts]
        batch = _pad(toks)
        out = scale_h_neurons_batch(self.model, batch, self.neurons, 1.0, self.layers)
        # alpha=1.0 must equal unmodified forward logits
        from hprobes.cett import forward_cett

        norms = {i: torch.ones(_I) for i in range(_L)}
        for i, t in enumerate(toks):
            _, logits = forward_cett(self.model, t, self.layers, norms)
            torch.testing.assert_close(out[i], logits, atol=1e-5, rtol=1e-4)
