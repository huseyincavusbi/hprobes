"""CETT metric computation.

CETT (Contribution to rEsidual sTream norm of Token t) measures how much
a single FFN neuron contributes to the hidden state at a given token position.

Formula
-------
    CETT(j, t) = |z_{j,t}| · ‖W_down[:, j]‖₂ / ‖h_t‖₂

where:
    z_{j,t}  = SwiGLU activation of neuron j at token t (input to W_down)
    h_t      = W_down · z_t  (FFN output vector at token t)
"""

from typing import Dict, List, Tuple

import torch


def _get_transformer_layers(model: torch.nn.Module):
    """Return the transformer layer list for any supported architecture.

    Handles:
      - Standard causal LMs:  model.model.layers (Gemma, Llama, Mistral)
      - Multimodal wrappers:  model.model.language_model.layers (MedGemma)
      - GPT-2:                model.transformer.h
      - OPT:                  model.model.decoder.layers
    """
    # Multimodal: model.model.language_model.layers (MedGemma-4B-IT)
    if hasattr(model, "model") and hasattr(model.model, "language_model"):
        lm = model.model.language_model
        if hasattr(lm, "layers"):
            return lm.layers
    # Standard causal LM: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    # GPT-2: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    # OPT: model.model.decoder.layers
    if (
        hasattr(model, "model")
        and hasattr(model.model, "decoder")
        and hasattr(model.model.decoder, "layers")
    ):
        return model.model.decoder.layers

    raise ValueError(
        f"Unsupported architecture: {type(model).__name__}. "
        "Expected model.model.layers or model.transformer.h."
    )


def get_mlp_down_proj(model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
    """Return the down-projection linear layer for a given transformer layer index.

    Supports:
      - down_proj (Llama, Gemma, Mistral)
      - c_proj (GPT2)
      - fc2 (OPT)
    """
    layers = _get_transformer_layers(model)
    if layer_idx >= len(layers):
        raise IndexError(f"Layer {layer_idx} out of range (model has {len(layers)} layers)")
    block = layers[layer_idx]

    # Handle various MLP layer names
    if hasattr(block, "mlp"):
        mlp = block.mlp
        for name in ["down_proj", "c_proj", "fc2"]:
            if hasattr(mlp, name):
                return getattr(mlp, name)

    # Some architectures might have it at the block level directly
    for name in ["down_proj", "c_proj", "fc2"]:
        if hasattr(block, name):
            return getattr(block, name)

    raise AttributeError(
        f"Could not find MLP down-projection layer in {type(block).__name__}. "
        "Checked: .mlp.down_proj, .mlp.c_proj, .mlp.fc2"
    )


def available_layers(model: torch.nn.Module) -> List[int]:
    """Return list of all available layer indices for the model."""
    return list(range(len(_get_transformer_layers(model))))


def _find_safetensors_files(model: torch.nn.Module):
    """Locate safetensors files for a model loaded from HF Hub."""
    import glob as _glob
    import json as _json
    from pathlib import Path as _Path

    config = getattr(model, "config", None)
    if config is None:
        return {}
    model_dir = getattr(config, "_name_or_path", None)
    if model_dir is None:
        return {}

    snapshot_dirs = []
    if _Path(model_dir).is_dir():
        snapshot_dirs.append(_Path(model_dir))

    cache_root = _Path.home() / ".cache" / "huggingface" / "hub"
    model_cache = cache_root / f"models--{model_dir.replace('/', '--')}"
    if model_cache.is_dir():
        snapshots_root = model_cache / "snapshots"
        if snapshots_root.is_dir():
            for snapshot in snapshots_root.iterdir():
                if snapshot.is_dir():
                    snapshot_dirs.append(snapshot)

    for base in snapshot_dirs:
        index_path = base / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as fh:
                index = _json.load(fh)
            weight_map = index.get("weight_map", {})
            return {base / fn for fn in set(weight_map.values())}
        shards = list(_glob.glob(str(base / "model*.safetensors")))
        if shards:
            return {_Path(s) for s in shards}
    return {}


def _materialize_meta_weight(model: torch.nn.Module, weight: torch.Tensor, param_name: str):
    """Materialize a meta-device weight by loading it from safetensors on disk."""
    import safetensors.torch as _sft

    if weight.device.type != "meta":
        return weight

    shard_paths = _find_safetensors_files(model)
    for shard_path in sorted(shard_paths):
        try:
            with _sft.safe_open(str(shard_path), framework="pt") as f:
                if param_name in f.keys():
                    return f.get_tensor(param_name).float()
        except Exception:
            continue
    return weight


def _get_weight_name(model: torch.nn.Module, layer_idx: int) -> str:
    """Find the full parameter name for a layer's down_proj.weight in the model."""
    down_proj = get_mlp_down_proj(model, layer_idx)
    for name, param in model.named_parameters():
        if param is down_proj.weight:
            return name
    return f"model.layers.{layer_idx}.mlp.down_proj.weight"


def precompute_col_norms(
    model: torch.nn.Module,
    layers: List[int],
) -> Dict[int, torch.Tensor]:
    """Precompute ‖W_down[:, j]‖₂ for each layer.

    Returns dict mapping layer_idx → (intermediate_dim,) tensor of column norms.
    Computed once and reused across all samples.
    """
    col_norms = {}
    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)
        W = down_proj.weight.detach().float()
        if W.device.type == "meta":
            W = _materialize_meta_weight(model, W, _get_weight_name(model, layer_idx))

        if type(down_proj).__name__ == "Conv1D":
            col_norms[layer_idx] = torch.norm(W, dim=1).cpu()
        else:
            col_norms[layer_idx] = torch.norm(W, dim=0).cpu()
    return col_norms


def forward_cett(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    token_position: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Single forward pass — extract CETT at a given token position.

    Parameters
    ----------
    model : causal LM
    tokens : tokenizer output (input_ids, attention_mask on correct device)
    layers : list of layer indices to hook
    col_norms : precomputed column norms from precompute_col_norms()
    token_position : which token to extract CETT at (-1 = last token)

    Returns
    -------
    cett_vec : (n_layers * intermediate_dim,) float32 — concatenated CETT values
    logits   : (vocab_size,) float32 — output logits at the last token
    """
    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                z = input[0]
                h = output
                z_cache[idx] = z[0, token_position, :].detach().float().cpu()
                h_cache[idx] = h[0, token_position, :].detach().float().cpu()
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            out = model(**tokens)
    finally:
        for h in handles:
            h.remove()

    logits = out.logits[0, -1, :].detach().float().cpu()

    cett_parts = []
    for layer_idx in layers:
        z = z_cache[layer_idx]
        h = h_cache[layer_idx]
        h_norm = torch.norm(h).item() + 1e-8
        cett = (torch.abs(z) * col_norms[layer_idx]) / h_norm
        cett_parts.append(cett)

    return torch.cat(cett_parts, dim=0), logits


def forward_cett_at_token(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    extra_token_id: int,
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Append one token to the input and capture CETT at that appended position.

    Returns
    -------
    cett_answer : (n_layers * intermediate_dim,) float32
    """
    input_ids = tokens["input_ids"]
    extra_t = torch.tensor([[extra_token_id]], device=input_ids.device)
    extended_ids = torch.cat([input_ids, extra_t], dim=1)

    extended: Dict[str, torch.Tensor] = {"input_ids": extended_ids}
    if "attention_mask" in tokens:
        m = tokens["attention_mask"]
        extended["attention_mask"] = torch.cat(
            [m, torch.ones((1, 1), device=m.device, dtype=m.dtype)], dim=1
        )

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                z_cache[idx] = input[0][0, -1, :].detach().float().cpu()
                h_cache[idx] = output[0, -1, :].detach().float().cpu()
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            model(**extended)
    finally:
        for h in handles:
            h.remove()

    cett_parts = []
    for layer_idx in layers:
        h_norm = torch.norm(h_cache[layer_idx]).item() + 1e-8
        cett = (torch.abs(z_cache[layer_idx]) * col_norms[layer_idx]) / h_norm
        cett_parts.append(cett)

    return torch.cat(cett_parts, dim=0)


def forward_cett_span(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    span_start: int,
    span_end: int,
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    aggregation: str = "mean",
) -> torch.Tensor:
    """Forward pass over a full sequence — extract CETT aggregated over a token span."""
    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                z_cache[idx] = input[0][0].detach().float().cpu()
                h_cache[idx] = output[0].detach().float().cpu()
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            model(**tokens)
    finally:
        for h in handles:
            h.remove()

    cett_parts = []
    for layer_idx in layers:
        z_span = z_cache[layer_idx][span_start:span_end]
        h_span = h_cache[layer_idx][span_start:span_end]
        h_norms = torch.norm(h_span, dim=-1, keepdim=True) + 1e-8
        cett_span = (torch.abs(z_span) * col_norms[layer_idx].unsqueeze(0)) / h_norms
        if aggregation == "max":
            cett_agg = cett_span.max(dim=0).values
        else:
            cett_agg = cett_span.mean(dim=0)
        cett_parts.append(cett_agg)

    return torch.cat(cett_parts, dim=0)


def forward_cett_dual_span(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    answer_start: int,
    answer_end: int,
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    aggregation: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One forward pass → two CETT vectors: answer span and non-answer span.

      CETT_{j,t} = |z_{j,t}| * ||W_down[:,j]|| / (||h_t|| + 1e-8)
      CETT_answer  = mean_{t in A} CETT_{j,t}
      CETT_other   = mean_{t not in A} CETT_{j,t}

    Returns
    -------
    cett_answer : (n_layers * intermediate_dim,) float32 — aggregated answer span
    cett_other  : (n_layers * intermediate_dim,) float32 — aggregated non-answer span
    """
    seq_len = tokens["input_ids"].shape[1]
    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                z_cache[idx] = input[0][0].detach().float().cpu()
                h_cache[idx] = output[0].detach().float().cpu()
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    try:
        with torch.inference_mode():
            model(**tokens)
    finally:
        for h in handles:
            h.remove()

    cett_answer_parts = []
    cett_other_parts = []

    mask_other = torch.ones(seq_len, dtype=torch.bool)
    mask_other[answer_start:answer_end] = False

    for layer_idx in layers:
        z = z_cache[layer_idx]
        h = h_cache[layer_idx]
        h_norm = torch.norm(h, dim=-1, keepdim=True) + 1e-8

        cett_per_token = (torch.abs(z) * col_norms[layer_idx].unsqueeze(0)) / h_norm

        cett_ans = cett_per_token[answer_start:answer_end]
        if aggregation == "max":
            cett_ans_agg = cett_ans.max(dim=0).values
        else:
            cett_ans_agg = cett_ans.mean(dim=0)

        cett_oth = cett_per_token[mask_other]
        if cett_oth.shape[0] > 0:
            if aggregation == "max":
                cett_oth_agg = cett_oth.max(dim=0).values
            else:
                cett_oth_agg = cett_oth.mean(dim=0)
        else:
            cett_oth_agg = torch.zeros_like(cett_ans_agg)

        cett_answer_parts.append(cett_ans_agg)
        cett_other_parts.append(cett_oth_agg)

    return torch.cat(cett_answer_parts, dim=0), torch.cat(cett_other_parts, dim=0)


def forward_cett_dual_span_batch(
    model: torch.nn.Module,
    batch_tokens: Dict[str, torch.Tensor],
    answer_spans: List[Tuple[int, int]],
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    aggregation: str = "mean",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched dual-span forward pass → CETT for answer and non-answer spans.

    Batched version of :func:`forward_cett_dual_span`. Pads all samples in the
    batch, runs ONE forward pass, and aggregates per-token CETT over each
    sample's answer span and non-answer span. Padded positions are excluded
    from both aggregations (they are not part of the sequence, so single-mode
    equivalence requires them to be masked out here).

    Parameters
    ----------
    model : causal LM
    batch_tokens : tokenizer output for a batch (input_ids, attention_mask padded)
    answer_spans : list of (start, end) exclusive token spans per sample
    layers : list of layer indices to hook
    col_norms : precomputed column norms from precompute_col_norms()
    aggregation : "mean" | "max"

    Returns
    -------
    cett_answer : (B, n_layers * intermediate_dim,) float32 CPU
    cett_other  : (B, n_layers * intermediate_dim,) float32 CPU
    """
    batch_size = batch_tokens["input_ids"].shape[0]
    seq_len = batch_tokens["input_ids"].shape[1]
    device = batch_tokens["input_ids"].device

    if len(answer_spans) != batch_size:
        raise ValueError(f"answer_spans length {len(answer_spans)} != batch_size {batch_size}")

    # Per-sample answer mask over the padded sequence
    answer_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    for i, (start, end) in enumerate(answer_spans):
        answer_mask[i, start:end] = True

    # Non-answer mask: every real (non-padded) token outside the answer span
    if "attention_mask" in batch_tokens:
        real_mask = batch_tokens["attention_mask"].to(torch.bool)
    else:
        real_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    other_mask = real_mask & ~answer_mask

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                z_cache[idx] = input[0]
                h_cache[idx] = output
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    if "attention_mask" in batch_tokens:
        position_ids = (batch_tokens["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)
    else:
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    try:
        with torch.inference_mode():
            model(**batch_tokens, position_ids=position_ids)
    finally:
        for h in handles:
            h.remove()

    cett_answer_parts = []
    cett_other_parts = []

    for layer_idx in layers:
        z = z_cache[layer_idx].float()
        h = h_cache[layer_idx].float()
        col_norm = col_norms[layer_idx].to(device)

        h_norm = torch.norm(h, dim=-1, keepdim=True) + 1e-8
        cett_per_token = (torch.abs(z) * col_norm.unsqueeze(0).unsqueeze(0)) / h_norm

        if aggregation == "max":
            inf = torch.finfo(cett_per_token.dtype).min
            ans_agg = torch.where(
                answer_mask.unsqueeze(-1),
                cett_per_token,
                torch.full_like(cett_per_token, inf),
            ).amax(dim=1)
            oth_agg = torch.where(
                other_mask.unsqueeze(-1),
                cett_per_token,
                torch.full_like(cett_per_token, inf),
            ).amax(dim=1)
            # Samples with an empty non-answer span → zero vector
            oth_agg = torch.where(
                other_mask.any(dim=1, keepdim=True), oth_agg, torch.zeros_like(oth_agg)
            )
        else:
            ans_agg = (cett_per_token * answer_mask.unsqueeze(-1)).sum(dim=1) / answer_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
            oth_agg = (cett_per_token * other_mask.unsqueeze(-1)).sum(dim=1) / other_mask.sum(
                dim=1, keepdim=True
            ).clamp(min=1)
            oth_agg = torch.where(
                other_mask.any(dim=1, keepdim=True), oth_agg, torch.zeros_like(oth_agg)
            )

        cett_answer_parts.append(ans_agg.cpu())
        cett_other_parts.append(oth_agg.cpu())

    cett_answer_matrix = torch.cat(cett_answer_parts, dim=1)
    cett_other_matrix = torch.cat(cett_other_parts, dim=1)

    return cett_answer_matrix, cett_other_matrix


def forward_cett_batch(
    model: torch.nn.Module,
    batch_tokens: Dict[str, torch.Tensor],
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
    token_positions: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Batched forward pass — extract CETT for each sample."""
    batch_size = batch_tokens["input_ids"].shape[0]
    device = batch_tokens["input_ids"].device
    batch_idx = torch.arange(batch_size, device=device)
    token_pos_t = torch.tensor(token_positions, device=device)

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                z_cache[idx] = input[0][batch_idx, token_pos_t].detach().float()
                h_cache[idx] = output[batch_idx, token_pos_t].detach().float()
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    if "attention_mask" in batch_tokens:
        position_ids = (batch_tokens["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)
    else:
        seq_len = batch_tokens["input_ids"].shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    try:
        with torch.inference_mode():
            out = model(**batch_tokens, position_ids=position_ids)
    finally:
        for h in handles:
            h.remove()

    logits_matrix = out.logits[batch_idx, token_pos_t].detach().float().cpu()

    z_all = torch.stack([z_cache[li] for li in layers], dim=0)
    h_all = torch.stack([h_cache[li] for li in layers], dim=0)
    col_norms_gpu = torch.stack([col_norms[li].to(device) for li in layers])

    h_norm = torch.norm(h_all, dim=-1, keepdim=True) + 1e-8
    cett = (torch.abs(z_all) * col_norms_gpu.unsqueeze(1)) / h_norm
    cett_matrix = cett.permute(1, 0, 2).reshape(batch_size, -1).cpu()

    return cett_matrix, logits_matrix


def forward_cett_at_token_batch(
    model: torch.nn.Module,
    batch_tokens: Dict[str, torch.Tensor],
    extra_token_ids: List[int],
    layers: List[int],
    col_norms: Dict[int, torch.Tensor],
) -> torch.Tensor:
    """Batched version of forward_cett_at_token."""
    batch_size = batch_tokens["input_ids"].shape[0]
    device = batch_tokens["input_ids"].device

    extra_t = torch.tensor(extra_token_ids, device=device).unsqueeze(1)
    extended_ids = torch.cat([batch_tokens["input_ids"], extra_t], dim=1)

    extended: Dict[str, torch.Tensor] = {"input_ids": extended_ids}
    if "attention_mask" in batch_tokens:
        m = batch_tokens["attention_mask"]
        extended["attention_mask"] = torch.cat(
            [m, torch.ones((batch_size, 1), device=device, dtype=m.dtype)], dim=1
        )

    z_cache: Dict[int, torch.Tensor] = {}
    h_cache: Dict[int, torch.Tensor] = {}
    handles = []

    for layer_idx in layers:
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_hook(idx: int):
            def hook(module, input, output):
                z_cache[idx] = input[0][:, -1, :].detach().float()
                h_cache[idx] = output[:, -1, :].detach().float()
                return output

            return hook

        handles.append(down_proj.register_forward_hook(make_hook(layer_idx)))

    if "attention_mask" in extended:
        position_ids = (extended["attention_mask"].cumsum(dim=-1) - 1).clamp(min=0)
    else:
        seq_len = extended["input_ids"].shape[1]
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)

    try:
        with torch.inference_mode():
            model(**extended, position_ids=position_ids)
    finally:
        for h in handles:
            h.remove()

    z_all = torch.stack([z_cache[li] for li in layers], dim=0)
    h_all = torch.stack([h_cache[li] for li in layers], dim=0)
    col_norms_gpu = torch.stack([col_norms[li].to(device) for li in layers])

    h_norm = torch.norm(h_all, dim=-1, keepdim=True) + 1e-8
    cett = (torch.abs(z_all) * col_norms_gpu.unsqueeze(1)) / h_norm
    cett_matrix = cett.permute(1, 0, 2).reshape(batch_size, -1).cpu()

    return cett_matrix


def scale_h_neurons_batch(
    model: torch.nn.Module,
    batch_tokens: Dict[str, torch.Tensor],
    h_neurons: List[Tuple[int, int]],
    alpha: float,
    layers: List[int],
) -> torch.Tensor:
    """Batched version of :func:`scale_h_neurons`.

    Runs ONE forward pass over a padded batch, scaling the same H-Neuron set
    by alpha, and returns the logits at each sample's last real token.

    Returns
    -------
    logits_matrix : (B, vocab_size,) float32 CPU
    """
    batch_size = batch_tokens["input_ids"].shape[0]
    device = batch_tokens["input_ids"].device

    neurons_by_layer: Dict[int, List[int]] = {}
    for layer_idx, neuron_idx in h_neurons:
        neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)

    handles = []
    for layer_idx in layers:
        if layer_idx not in neurons_by_layer:
            continue
        indices = torch.tensor(neurons_by_layer[layer_idx], dtype=torch.long)
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_pre_hook(idx: torch.Tensor, a: float):
            def pre_hook(module, input):
                z = input[0].clone()
                z[..., idx.to(z.device)] *= a
                return (z,) + input[1:]

            return pre_hook

        handles.append(down_proj.register_forward_pre_hook(make_pre_hook(indices, alpha)))

    try:
        with torch.inference_mode():
            out = model(**batch_tokens)
    finally:
        for h in handles:
            h.remove()

    last_positions = batch_tokens["attention_mask"].sum(dim=1) - 1
    logits = out.logits[torch.arange(batch_size, device=device), last_positions]
    return logits.detach().float().cpu()


def scale_h_neurons(
    model: torch.nn.Module,
    tokens: Dict[str, torch.Tensor],
    h_neurons: List[Tuple[int, int]],
    alpha: float,
    layers: List[int],
) -> torch.Tensor:
    """Forward pass scaling H-Neuron activations by alpha."""
    neurons_by_layer: Dict[int, List[int]] = {}
    for layer_idx, neuron_idx in h_neurons:
        neurons_by_layer.setdefault(layer_idx, []).append(neuron_idx)

    handles = []
    for layer_idx in layers:
        if layer_idx not in neurons_by_layer:
            continue
        indices = torch.tensor(neurons_by_layer[layer_idx], dtype=torch.long)
        down_proj = get_mlp_down_proj(model, layer_idx)

        def make_pre_hook(idx: torch.Tensor, a: float):
            def pre_hook(module, input):
                z = input[0].clone()
                z[..., idx.to(z.device)] *= a
                return (z,) + input[1:]

            return pre_hook

        handles.append(down_proj.register_forward_pre_hook(make_pre_hook(indices, alpha)))

    try:
        with torch.inference_mode():
            out = model(**tokens)
    finally:
        for h in handles:
            h.remove()

    return out.logits[0, -1, :].detach().float().cpu()
