"""Fake MoE LM for tests: deterministic token-id routing, zeroed logits head.

Mirrors the family-faithful experts contract (Olmoe/Qwen2Moe/Qwen3.5MoE) so
MoECapture's wrapper applies unchanged. Routing is a pure function of token
id (expert[t % E], expert[(t+1) % E]) making position coincidences and
cross-sample leaks fully controllable in tests. The zeroed head yields
deterministic predictions (first letter id wins), so correctness labels are
trivial to plant via the sample answers.
"""

from types import SimpleNamespace

import torch
from torch import nn


class _FakeMoEExperts(nn.Module):
    def __init__(self, num_experts=4, d_model=8, d_ff=6):
        super().__init__()
        self.num_experts = num_experts
        self.gate_up_proj = nn.Parameter(torch.randn(num_experts, 2 * d_ff, d_model) * 0.1)
        self.down_proj = nn.Parameter(torch.randn(num_experts, d_model, d_ff) * 0.1)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        expert_mask = torch.nn.functional.one_hot(
            top_k_index, num_classes=self.num_experts
        ).permute(2, 1, 0)
        for e in range(self.num_experts):
            if expert_mask[e].sum() == 0:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[e])
            x = hidden_states[token_idx]
            gate, up = torch.nn.functional.linear(x, self.gate_up_proj[e]).chunk(2, dim=-1)
            act = self.act_fn(gate) * up
            out = torch.nn.functional.linear(act, self.down_proj[e])
            out = out * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, out)
        return final


class _FakeMoELayer(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.experts = _FakeMoEExperts(**kw)


class _FakeMoEModel(nn.Module):
    def __init__(self, vocab=128, d_model=8, num_experts=4, k=2, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeMoELayer(num_experts=num_experts, d_model=d_model)])
        self.embed = nn.Embedding(vocab, d_model)
        self.head = nn.Linear(d_model, vocab)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()
        self.k = k
        self.num_experts = num_experts

    def forward(self, input_ids=None, attention_mask=None, **kw):
        ids = input_ids[0]
        hidden = self.embed(ids)
        e0 = ids % self.num_experts
        e1 = (ids + 1) % self.num_experts
        top_k_index = torch.stack([e0, e1], dim=-1)
        top_k_weights = torch.full(top_k_index.shape, 1.0 / self.k, dtype=torch.float)
        out = self.model.layers[0].mlp.experts(hidden, top_k_index, top_k_weights)
        logits = self.head(hidden + out)
        return SimpleNamespace(logits=logits.unsqueeze(0))
