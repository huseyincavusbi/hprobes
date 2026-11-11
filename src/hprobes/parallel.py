"""Data-parallel CETT extraction (issue #25).

One full model replica per process/device, deterministic sample sharding
(``i % world_size``), ordered merge, and central fitting. The serial path is
unchanged; this module only orchestrates shards and reassembles
:class:`~hprobes.rows.CETTRows` in global sample order.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .rows import CETTRows


@dataclass
class _ShardResult:
    """One worker's extraction result and its global sample positions."""

    shard_id: int
    valid_global_indices: List[int]
    artifact: Optional[CETTRows]


def _device_name(value: Any) -> str:
    """Normalize a device-map value to a stable device string."""
    if isinstance(value, int):
        return f"cuda:{value}"
    return str(value)


def _resident_model_devices(probe: Any) -> List[str]:
    """Return the devices occupied by the parent probe's model, if any."""
    model = getattr(probe, "model", None)
    if model is None:
        return []
    device_map = getattr(model, "hf_device_map", None)
    if device_map:
        return sorted({_device_name(value) for value in device_map.values()})
    return sorted({str(p.device) for p in model.parameters() if p.device.type != "meta"})


def _ensure_config_only_parent(probe: Any) -> None:
    """Refuse a resident non-CPU parent model that would compete with replicas."""
    non_cpu = [device for device in _resident_model_devices(probe) if not device.startswith("cpu")]
    if non_cpu:
        raise RuntimeError(
            "parallel CETT extraction requires a config-only parent probe: "
            f"found resident model device(s) {', '.join(non_cpu)}. "
            "Construct HProbes(None, None, ...) or move the parent model to CPU."
        )


def _shard_indices(n_samples: int, world_size: int, shard_id: int) -> List[int]:
    """Deterministic interleaved shard: sample ``i`` belongs to ``i % K``."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if not 0 <= shard_id < world_size:
        raise ValueError(f"shard_id must be in [0, {world_size}), got {shard_id}")
    return list(range(shard_id, n_samples, world_size))


def _resolve_devices(world_size: int, devices: Optional[Sequence[str]]) -> List[str]:
    if devices is not None:
        resolved = [_device_name(d) for d in devices]
        if len(resolved) < world_size:
            raise ValueError(
                f"need {world_size} devices for world_size={world_size}, got {len(resolved)}"
            )
        resolved = resolved[:world_size]
        if len(set(resolved)) != len(resolved) and not all(
            device.startswith("cpu") for device in resolved
        ):
            raise ValueError("devices must be distinct for parallel extraction")
        return resolved
    if torch.cuda.is_available() and torch.cuda.device_count() >= world_size:
        return [f"cuda:{i}" for i in range(world_size)]
    available = torch.cuda.device_count() if torch.cuda.is_available() else 0
    raise ValueError(
        f"world_size={world_size} requires {world_size} CUDA devices, found {available}. "
        "Pass devices=['cpu', ...] explicitly to run CPU replicas."
    )


def _probe_kwargs(probe: Any) -> Dict[str, Any]:
    """Constructor arguments shared by every worker replica."""
    top_k: Any = "auto" if getattr(probe, "_top_k_auto", False) else probe.top_k
    return {
        "l1_C": probe.l1_C,
        "layer_stride": probe.layer_stride,
        "validation_split": probe.validation_split,
        "seed": probe.seed,
        "max_tokens": probe.max_tokens,
        "batch_size": probe.batch_size,
        "n_consistency": probe.n_consistency,
        "top_k": top_k,
        "auto_safety": probe.auto_safety,
        "save_predictions": probe.save_predictions,
        "readout": probe.readout,
        "readout_max_new_tokens": probe.readout_max_new_tokens,
        "check_l2": probe.check_l2,
        "stability": probe.stability,
        "correlation": probe.correlation,
        "cluster": probe.cluster,
        "strict_memory": probe.strict_memory,
        "solver": probe.solver,
        "stability_n_jobs": probe.stability_n_jobs,
    }


def _extract_shard_worker(
    shard_id: int,
    indexed_samples: List[Tuple[int, Dict[str, Any]]],
    model_factory: Callable[[str], Tuple[Any, Any]],
    device: str,
    probe_kwargs: Dict[str, Any],
    extract_kwargs: Dict[str, Any],
) -> _ShardResult:
    """Worker entry point: build one replica and extract its sample slice."""
    if not indexed_samples:
        return _ShardResult(shard_id, [], None)
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))

    model, tokenizer = model_factory(device)
    from .probe import HProbes  # local import avoids a module cycle

    probe = HProbes(model, tokenizer, **probe_kwargs)

    samples = [sample for _, sample in indexed_samples]
    try:
        artifact = probe.extract_features(samples, **extract_kwargs)
    except ValueError as exc:
        if "no valid samples" in str(exc):
            return _ShardResult(shard_id, [], None)
        raise

    id_to_global = {id(sample): idx for idx, sample in indexed_samples}
    valid_global_indices = []
    for sample in artifact.samples:
        global_idx = id_to_global.get(id(sample))
        if global_idx is None:
            raise ValueError(
                "extracted sample object is not one of the shard inputs; "
                "parallel extraction relies on sample identity"
            )
        valid_global_indices.append(global_idx)
    return _ShardResult(shard_id, valid_global_indices, artifact)


def merge_shards(results: Sequence[_ShardResult], n_features: Optional[int] = None) -> CETTRows:
    """Merge worker artifacts in global sample order and recompute Welford stats."""
    ordered = sorted((r for r in results if r.artifact is not None), key=lambda r: r.shard_id)
    if not ordered:
        raise ValueError("no valid samples — every shard produced zero rows")

    baseline_meta = ordered[0].artifact.meta
    for result in ordered[1:]:
        if result.artifact.meta != baseline_meta:
            raise ValueError(
                "shard metadata differ; workers must load identical models and extraction config"
            )

    width = n_features if n_features is not None else ordered[0].artifact.rows.shape[1]
    entries: List[Tuple[int, int, int, List[int]]] = []
    for result in ordered:
        artifact = result.artifact
        if artifact.rows.shape[1] != width:
            raise ValueError(f"shard {result.shard_id} width {artifact.rows.shape[1]} != {width}")
        if len(result.valid_global_indices) != len(artifact.samples):
            raise ValueError(
                f"shard {result.shard_id} has {len(result.valid_global_indices)} indices "
                f"for {len(artifact.samples)} samples"
            )
        rows_by_sample: Dict[int, List[int]] = {}
        for row_idx, sample_idx in enumerate(artifact.row_to_sample):
            rows_by_sample.setdefault(int(sample_idx), []).append(row_idx)
        for local_idx, global_idx in enumerate(result.valid_global_indices):
            entries.append(
                (int(global_idx), result.shard_id, local_idx, rows_by_sample.get(local_idx, []))
            )

    entries.sort(key=lambda entry: (entry[0], entry[1]))
    global_indices = [entry[0] for entry in entries]
    if len(set(global_indices)) != len(global_indices):
        raise ValueError("duplicate global sample indices across shards")

    by_shard = {result.shard_id: result.artifact for result in ordered}
    rows_list = []
    labels_list = []
    row_to_sample: List[int] = []
    prompts: List[str] = []
    ground_truth: List[str] = []
    per_sample: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []

    for merged_idx, (_, shard_id, local_idx, row_indices) in enumerate(entries):
        artifact = by_shard[shard_id]
        for row_idx in row_indices:
            rows_list.append(artifact.rows[row_idx])
            labels_list.append(artifact.labels[row_idx])
            row_to_sample.append(merged_idx)
        prompts.append(artifact.prompts[local_idx])
        ground_truth.append(artifact.ground_truth[local_idx])
        per_sample.append(artifact.per_sample[local_idx])
        samples.append(artifact.samples[local_idx])

    rows = np.stack(rows_list, axis=0) if rows_list else np.empty((0, width), dtype=np.float32)
    labels = np.array(labels_list)

    from .probe import _welford_update_state

    welford_n = 0
    welford_mean = np.zeros(width, dtype=np.float64)
    welford_M2 = np.zeros(width, dtype=np.float64)
    for row in rows:
        welford_n, welford_mean, welford_M2 = _welford_update_state(
            welford_n, welford_mean, welford_M2, row
        )

    artifact = CETTRows(
        rows=rows,
        labels=labels,
        row_to_sample=row_to_sample,
        prompts=prompts,
        ground_truth=ground_truth,
        per_sample=per_sample,
        samples=samples,
        welford_n=welford_n,
        welford_mean=welford_mean,
        welford_M2=welford_M2,
        meta=dict(baseline_meta),
    )
    artifact.validate()
    return artifact


def run_parallel_extraction(
    probe: Any,
    samples: List[Dict[str, Any]],
    model_factory: Callable[[str], Tuple[Any, Any]],
    world_size: Optional[int],
    devices: Optional[Sequence[str]],
    extract_kwargs: Dict[str, Any],
) -> CETTRows:
    """Run sharded extraction across ``world_size`` spawned processes."""
    if world_size is None:
        if devices is not None:
            world_size = len(devices)
        elif torch.cuda.is_available():
            world_size = max(1, torch.cuda.device_count())
        else:
            world_size = 1
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if world_size == 1:
        if getattr(probe, "model", None) is None:
            raise ValueError(
                "world_size=1 requires a loaded parent model; pass world_size>1 "
                "with explicit devices to use model_factory replicas"
            )
        return probe.extract_features(samples, **extract_kwargs)
    if getattr(probe, "batch_size", 1) != 1:
        raise ValueError(
            "parallel CETT extraction requires batch_size=1 for bit-identical rows; "
            "sharding changes batch composition"
        )
    if model_factory is None:
        raise ValueError("model_factory is required when world_size > 1")
    _ensure_config_only_parent(probe)

    resolved_devices = _resolve_devices(world_size, devices)
    probe_kwargs = _probe_kwargs(probe)
    shards = [
        [(idx, samples[idx]) for idx in _shard_indices(len(samples), world_size, shard_id)]
        for shard_id in range(world_size)
    ]

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=world_size, mp_context=ctx) as pool:
        futures = [
            pool.submit(
                _extract_shard_worker,
                shard_id,
                shards[shard_id],
                model_factory,
                resolved_devices[shard_id],
                probe_kwargs,
                extract_kwargs,
            )
            for shard_id in range(world_size)
        ]
        results = [future.result() for future in futures]

    return merge_shards(results, n_features=getattr(probe, "_n_features", None) or None)
