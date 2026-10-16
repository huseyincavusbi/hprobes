"""Cached CETT rows artifact for issue #26.

Split probe fitting into extraction (expensive forwards, run once) and
selection+fitting (cheap, run per variant). :class:`CETTRows` is the
immutable handoff: full unfiltered per-row features plus the Welford
summary stats, so downstream variance pre-selection reproduces the inline
path exactly. Selection (variance top-k, structural priors, clustering)
always stays downstream of the artifact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class CETTRows:
    """Full unfiltered CETT extraction output for one sample set.

    ``rows`` is (n_rows, n_features); ``labels`` aligns per row.
    ``welford_*`` carry the extraction-time summary stats so
    ``variance()`` matches the inline ``_welford_M2 / (n - 1)`` path bit
    for bit without recomputing from rows.
    """

    rows: np.ndarray
    labels: np.ndarray
    row_to_sample: List[int]
    prompts: List[str]
    ground_truth: List[str]
    per_sample: List[Dict[str, Any]]
    samples: List[Dict[str, Any]]
    welford_n: int
    welford_mean: np.ndarray
    welford_M2: np.ndarray
    meta: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Raise on any shape/content inconsistency."""
        n = self.rows.shape[0]
        if self.rows.ndim != 2:
            raise ValueError(f"rows must be 2-D, got {self.rows.shape}")
        if len(self.labels) != n:
            raise ValueError(f"labels {len(self.labels)} != rows {n}")
        if len(self.row_to_sample) != n:
            raise ValueError(f"row_to_sample {len(self.row_to_sample)} != rows {n}")
        if len(self.prompts) != len(self.samples) or len(self.ground_truth) != len(self.samples):
            raise ValueError("prompts/ground_truth must align with samples")
        if len(self.per_sample) != len(self.samples):
            raise ValueError("per_sample must align with samples")
        if self.welford_mean.shape != (self.rows.shape[1],) or self.welford_M2.shape != (
            self.rows.shape[1],
        ):
            raise ValueError("welford stats must align with n_features")
        if max(self.row_to_sample, default=-1) >= len(self.samples):
            raise ValueError("row_to_sample indexes out of range")

    def variance(self) -> np.ndarray:
        """Sample variance per feature, mirroring the inline path."""
        return self.welford_M2 / max(self.welford_n - 1, 1)


def fingerprint_model(model: Any, layers: List[int]) -> str:
    """Cheap stable fingerprint of model weights for cache keys.

    Hashes element counts plus sums of the first, middle and last
    parameter tensors: catches silent weight swaps without hashing
    gigabytes. Deterministic for a fixed state dict.
    """

    params = [p.detach().cpu().float() for p in model.parameters()]
    if not params:
        raise ValueError("model has no parameters to fingerprint")
    picks = [params[0], params[len(params) // 2], params[-1]]
    parts = [type(model).__name__, str(list(layers))]
    for p in picks:
        parts.append(f"{p.numel()}:{float(p.sum()):.6f}:{float(p.abs().sum()):.6f}")
    try:
        import transformers  # noqa: PLC0415

        parts.append(transformers.__version__)
    except ImportError:
        pass
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def rows_key(model_fp: str, samples: List[Dict[str, Any]], config: Dict[str, Any]) -> str:
    """Cache key binding rows to exact extraction inputs.

    Any change to weights (via ``model_fp``), samples, or extraction
    config (layers, batching, readout, seed, ...) yields a different key,
    so stale rows can never be silently reused.
    """
    payload = json.dumps(
        {"model": model_fp, "samples": samples, "config": config},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def save_rows(artifact: CETTRows, base_path: str) -> List[str]:
    """Persist an artifact as ``<base>.npz`` (arrays) + ``<base>.json`` (rest).

    Returns the written paths. Reload with :func:`load_rows`.
    """
    from pathlib import Path  # noqa: PLC0415

    artifact.validate()
    base = Path(base_path)
    npz_path = base.with_suffix(".npz")
    json_path = base.with_suffix(".json")
    np.savez_compressed(
        npz_path,
        rows=artifact.rows,
        labels=np.asarray(artifact.labels),
        row_to_sample=np.asarray(artifact.row_to_sample),
        welford_mean=artifact.welford_mean,
        welford_M2=artifact.welford_M2,
    )
    json_path.write_text(
        json.dumps(
            {
                "welford_n": artifact.welford_n,
                "prompts": artifact.prompts,
                "ground_truth": artifact.ground_truth,
                "per_sample": artifact.per_sample,
                "samples": artifact.samples,
                "meta": artifact.meta,
            },
            default=str,
        )
    )
    return [str(npz_path), str(json_path)]


def load_rows(base_path: str) -> CETTRows:
    """Reload an artifact written by :func:`save_rows` (validated)."""
    from pathlib import Path  # noqa: PLC0415

    base = Path(base_path)
    data = np.load(base.with_suffix(".npz"))
    doc = json.loads(base.with_suffix(".json").read_text())
    artifact = CETTRows(
        rows=np.asarray(data["rows"]),
        labels=np.asarray(data["labels"]),
        row_to_sample=[int(i) for i in data["row_to_sample"]],
        prompts=list(doc["prompts"]),
        ground_truth=list(doc["ground_truth"]),
        per_sample=list(doc["per_sample"]),
        samples=list(doc["samples"]),
        welford_n=int(doc["welford_n"]),
        welford_mean=np.asarray(data["welford_mean"]),
        welford_M2=np.asarray(data["welford_M2"]),
        meta=dict(doc["meta"]),
    )
    artifact.validate()
    return artifact
