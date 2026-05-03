import pytest
import pandas as pd
from src.io.loaders import (
    load_optimal_weights, load_equity_curves, load_performance_metrics,
    load_attribution_results, load_regime_performance,
    load_backtest_results, load_pipeline_output,
)


def test_load_optimal_weights_long_format():
    df = load_optimal_weights()
    assert 'Date' in df.columns and 'ticker' in df.columns
    assert 'weight' in df.columns


def test_load_equity_curves_has_date_index():
    df = load_equity_curves()
    assert df.index.name == 'Date'


def test_load_performance_metrics_returns_dict():
    m = load_performance_metrics()
    assert isinstance(m, dict)
    assert 'sharpe' in m or 'Sharpe' in m


def test_load_pipeline_output_returns_dict():
    out = load_pipeline_output()
    assert isinstance(out, dict)
