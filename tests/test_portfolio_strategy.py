import numpy as np
import pandas as pd
import pytest
from src.portfolio import construct_portfolio, trailing_signal


def test_trailing_signal_excludes_future_and_checks_age():
    dates = pd.date_range('2024-01-01', periods=20, freq='B')
    signals = pd.DataFrame({'Date': [dates[10], dates[15], dates[18], dates[10]],
                            'ticker': ['A', 'A', 'A', 'B'],
                            'alpha_signal': [0.4, 0.6, 0.9, 0.3],
                            'training_end': [dates[4], dates[9], dates[12], dates[4]]})
    values, used = trailing_signal(signals, dates[16], dates, ['A', 'B'])
    assert values['A'] == pytest.approx(0.6)
    assert used['A'] == dates[15]
    invalid = signals.copy()
    invalid.loc[1, 'training_end'] = dates[15]
    with pytest.raises(ValueError, match='provenance'):
        trailing_signal(invalid, dates[16], dates, ['A', 'B'])
    with pytest.raises(ValueError, match='provenance'):
        trailing_signal(signals.drop(columns='training_end'), dates[16], dates, ['A', 'B'])
    with pytest.raises(ValueError, match='stale'):
        trailing_signal(signals, dates[19], dates, ['A', 'B'], max_age=2)
    with pytest.raises(ValueError, match='missing'):
        trailing_signal(signals, dates[16], dates, ['A', 'C'])


def test_allocation_obeys_cap_and_cash_overlay():
    rng = np.random.default_rng(55)
    columns = [f'S{i}' for i in range(12)]
    returns = pd.DataFrame(rng.normal(0.0003, 0.01, (100, 12)),
                           index=pd.date_range('2024-01-01', periods=100, freq='B'),
                           columns=columns)
    probabilities = pd.Series(np.linspace(.4, .6, 12), index=columns)
    weights, details = construct_portfolio(returns, probabilities)
    assert weights.sum() == pytest.approx(1 - details['cash_pct'])
    assert weights.min() >= -1e-8
    assert weights.max() <= .1 + 1e-8
    assert np.isfinite(weights).all()
    with pytest.raises(ValueError, match='probabilities'):
        construct_portfolio(returns, probabilities * 3)


def test_allocation_risk_gate():
    from src.portfolio import audit_allocation
    assert not audit_allocation(pd.Series([.05] * 10))
    assert audit_allocation(pd.Series([.11, .05]))
    assert audit_allocation(pd.Series([float('nan'), .05]))
    assert audit_allocation(pd.Series([-.01, .05]))


def test_trailing_signal_rejects_insufficient_training_gap():
    dates = pd.date_range('2024-01-01', periods=20, freq='B')
    signals = pd.DataFrame({'Date': [dates[15]], 'ticker': ['A'], 'alpha_signal': [.5],
                            'training_end': [dates[12]]})
    with pytest.raises(ValueError, match='five-day gap'):
        trailing_signal(signals, dates[15], dates, ['A'])
