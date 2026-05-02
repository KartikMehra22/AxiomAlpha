"""Single source of truth for every pipeline read.

Every notebook imports from this module instead of using pd.read_csv directly.
Every loader validates schema and raises PipelineDataError on missing/malformed
inputs, with a message naming the producer notebook.
"""
from __future__ import annotations
from pathlib import Path


class PipelineDataError(FileNotFoundError):
    """Raised when a pipeline artifact is missing or malformed."""


def _find_project_root() -> Path:
    """Walk up from this file until we find the AxiomAlpha directory."""
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if parent.name == 'AxiomAlpha':
            return parent
    raise RuntimeError(
        f"Could not locate AxiomAlpha project root from {here}"
    )


PROJECT_ROOT = _find_project_root()
RAW = PROJECT_ROOT / 'data' / 'raw'
PROC = PROJECT_ROOT / 'data' / 'processed'
OUTPUTS = PROJECT_ROOT / 'data' / 'outputs'
