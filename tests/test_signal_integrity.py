import pandas as pd
import pytest
from src.io.loaders import load_master_signals, load_log_returns


@pytest.mark.requires_artifacts
def test_signal_training_provenance_precedes_each_prediction():
    signals = load_master_signals()
    assert 'training_end' in signals.columns
    dates = pd.to_datetime(signals['Date'])
    training_end = pd.to_datetime(signals['training_end'])
    assert dates.notna().all() and training_end.notna().all()
    assert (training_end < dates).all()
    assert not signals.duplicated(['Date', 'ticker']).any()
    calendar = pd.DatetimeIndex(load_log_returns().index)
    gap = calendar.searchsorted(dates) - calendar.searchsorted(training_end)
    assert (gap >= 6).all()
