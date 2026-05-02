import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def tmp_data_dir(tmp_path, monkeypatch):
    """Redirect data paths to a temp directory so tests don't touch real data."""
    raw = tmp_path / 'data' / 'raw'
    proc = tmp_path / 'data' / 'processed'
    raw.mkdir(parents=True)
    proc.mkdir(parents=True)
    return {'root': tmp_path, 'raw': raw, 'proc': proc}


@pytest.fixture
def sample_prices_df():
    """30 trading days of 5 tickers, no NaN."""
    dates = pd.date_range('2024-01-01', periods=30, freq='B', name='Date')
    rng = np.random.default_rng(42)
    data = 100 + rng.standard_normal((30, 5)).cumsum(axis=0)
    return pd.DataFrame(data, index=dates,
                        columns=['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'])
