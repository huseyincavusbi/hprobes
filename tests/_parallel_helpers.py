"""Tiny picklable model/tokenizer helpers for parallel CETT tests."""

from __future__ import annotations

import torch
import torch.nn as nn

H, D_FF, V, L = 8, 16, 128, 4


class TinyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(H, D_FF, bias=False)
        self.down_proj = nn.Linear(D_FF, H, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.gate(x)))


class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = TinyMLP()

    def forward(self, x):
        return x + self.mlp(x)


class TinyModel(nn.Module):
    def __init__(self, n_layers: int = L):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([TinyBlock() for _ in range(n_layers)])
        self.lm_head = nn.Linear(H, V, bias=True)

    def forward(self, input_ids=None, attention_mask=None, **kw):
        x = input_ids.float().unsqueeze(-1).expand(-1, -1, H)
        for block in self.model.layers:
            x = block(x)

        class _Out:
            pass

        out = _Out()
        out.logits = self.lm_head(x)
        return out


class TinyTokenizerOutput(dict):
    def to(self, device):
        return {k: v.to(device) for k, v in self.items()}


class TinyTokenizer:
    chat_template = None
    padding_side = "right"
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)

    def __call__(self, text, return_tensors=None, truncation=False, max_length=None, padding=False):
        if isinstance(text, list):
            all_ids = [[ord(c) for c in t] for t in text]
            if max_length:
                all_ids = [ids[:max_length] for ids in all_ids]
            max_len = max(len(ids) for ids in all_ids)
            padded = [ids + [0] * (max_len - len(ids)) for ids in all_ids]
            masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in all_ids]
            return TinyTokenizerOutput(
                input_ids=torch.tensor(padded, dtype=torch.long),
                attention_mask=torch.tensor(masks, dtype=torch.long),
            )
        ids = [ord(c) for c in text]
        if max_length:
            ids = ids[:max_length]
        input_ids = torch.tensor([ids], dtype=torch.long)
        return TinyTokenizerOutput(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))


def make_tiny_model(state_path: str, device: str = "cpu"):
    """Load one deterministic replica for a worker process."""
    model = TinyModel()
    model.load_state_dict(torch.load(state_path, map_location="cpu"))
    model.to(device)
    model.eval()
    return model, TinyTokenizer()
