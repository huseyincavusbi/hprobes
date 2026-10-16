"""hprobes — Hallucination neuron discovery and causal validation for transformer LLMs."""

from importlib.metadata import version as _metadata_version

from .probe import HProbes
from .rows import CETTRows, load_rows, rows_key, save_rows

__all__ = ["HProbes", "CETTRows", "save_rows", "load_rows", "rows_key"]
__version__ = _metadata_version("hprobes")
