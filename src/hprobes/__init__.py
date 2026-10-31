"""hprobes — Hallucination neuron discovery and causal validation for transformer LLMs."""

from importlib.metadata import version as _metadata_version

from .probe import HProbes
from .rows import CETTRows, load_rows, rows_key, save_rows
from .steering import (
    SteeringController,
    SteeringSpec,
    contrastive_direction,
    median_residual_norm,
    random_direction,
    write_vector,
)

__all__ = [
    "HProbes",
    "CETTRows",
    "save_rows",
    "load_rows",
    "rows_key",
    "SteeringSpec",
    "SteeringController",
    "write_vector",
    "contrastive_direction",
    "random_direction",
    "median_residual_norm",
]
__version__ = _metadata_version("hprobes")
