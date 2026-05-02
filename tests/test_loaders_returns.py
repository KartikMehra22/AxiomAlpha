import pytest
import pandas as pd
from src.io.loaders import (
    load_log_returns, load_simple_returns,
    load_features_raw, load_features_normalized, load_feature_scaler,
)


def test_load_log_returns_shape_and_index():
    df = load_log_returns()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'Date'
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df.index.is_monotonic_increasing


def test_load_simple_returns_shape_and_index():
    df = load_simple_returns()
    assert df.index.name == 'Date'


def test_load_features_normalized_long_format():
    """Long-format: Date and ticker as columns (not index)."""
    df = load_features_normalized()
    assert 'Date' in df.columns
    assert 'ticker' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])


def test_load_features_raw_long_format():
    df = load_features_raw()
    assert 'Date' in df.columns
    assert 'ticker' in df.columns


def test_load_feature_scaler_returns_sklearn_scaler():
    scaler = load_feature_scaler()
    assert hasattr(scaler, 'transform')
    assert hasattr(scaler, 'mean_') or hasattr(scaler, 'scale_')
