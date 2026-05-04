import pytest
import pandas as pd
from src.io.loaders import (
    load_optimal_weights, load_equity_curves, load_performance_metrics,
    load_attribution_results, load_regime_performance,
    load_backtest_results, load_pipeline_output,
)

pytestmark = pytest.mark.requires_artifacts


def test_load_optimal_weights_long_format():
    df = load_optimal_weights()
    assert 'Date' in df.columns and 'ticker' in df.columns
    assert 'weight' in df.columns


def test_load_equity_curves_has_date_index():
    df = load_equity_curves()
    assert df.index.name == 'Date'


def test_load_performance_metrics_returns_dict():
    """The canonical schema is row-per-strategy with a 'sharpe' column.
    Loader returns {strategy_name: {col: value}}; for a metric-keyed file
    it returns {metric_name: value}. Either way 'sharpe' must be reachable.
    """
    m = load_performance_metrics()
    assert isinstance(m, dict)
    if not m:
        pytest.skip("performance_metrics.csv is empty")
    # Strategy-keyed: outer keys are strategy names, inner dicts have 'sharpe'
    first_val = next(iter(m.values()))
    if isinstance(first_val, dict):
        assert 'sharpe' in first_val, \
            f"expected 'sharpe' (lowercase) in inner metric dict; got {list(first_val)}"
    else:
        # Metric-keyed: 'sharpe' must be a top-level key
        assert 'sharpe' in m, f"expected 'sharpe' key; got {list(m)}"


def test_load_pipeline_output_returns_dict():
    out = load_pipeline_output()
    assert isinstance(out, dict)
