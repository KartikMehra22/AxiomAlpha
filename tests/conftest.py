"""Shared test fixtures and collection hooks.

Loader tests that read real artifacts under `data/processed/` are marked
with `@pytest.mark.requires_artifacts` (often via module-level `pytestmark`).
On a fresh clone the artifacts don't exist yet, so those tests would error
with MissingArtifactError. The collection hook below converts that into a
clean SKIP so a fresh-clone `pytest` run reports formula tests + check tests
without spurious failures.
"""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from pathlib import Path


# A small set of canonical artifacts that, if missing, indicates the
# pipeline hasn't been run. We only need to probe a few — if any one is
# present we assume the suite was run; if none are present we skip the
# integration tests cleanly.
_REQUIRED_ARTIFACTS = (
    'data/raw/prices.csv',
    'data/processed/log_returns.csv',
)


def _artifacts_present() -> bool:
    repo_root = Path(__file__).resolve().parent.parent
    return all((repo_root / p).exists() for p in _REQUIRED_ARTIFACTS)


def pytest_collection_modifyitems(config, items):
    """Skip tests marked `requires_artifacts` when the pipeline hasn't been run."""
    if _artifacts_present():
        return
    skip_marker = pytest.mark.skip(
        reason="pipeline artifacts not present; "
               "run `python run_all.py` first to enable integration tests"
    )
    for item in items:
        if 'requires_artifacts' in item.keywords:
            item.add_marker(skip_marker)


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Redirect data paths to a temp directory so unit tests don't touch real data."""
    raw = tmp_path / 'data' / 'raw'
    proc = tmp_path / 'data' / 'processed'
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    return {'root': tmp_path, 'raw': raw, 'proc': proc}


@pytest.fixture
def sample_prices_df():
    """30 trading days of 5 tickers, no NaN. Used by formula/regression tests."""
    dates = pd.date_range('2024-01-01', periods=30, freq='B', name='Date')
    rng = np.random.default_rng(42)
    data = 100 + rng.standard_normal((30, 5)).cumsum(axis=0)
    return pd.DataFrame(data, index=dates,
                        columns=['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'])
