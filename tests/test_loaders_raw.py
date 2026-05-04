import pytest
import pandas as pd
from src.io.loaders import (
    load_prices, load_macro, load_sector_map,
    PipelineDataError,
)

pytestmark = pytest.mark.requires_artifacts


def test_load_prices_returns_dataframe_with_date_index():
    df = load_prices()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'Date'
    assert pd.api.types.is_datetime64_any_dtype(df.index)
    assert df.index.is_monotonic_increasing
    assert df.shape[0] > 100  # at least 100 trading days
    assert df.shape[1] >= 25  # at least 25 tickers


def test_load_prices_no_nans():
    df = load_prices()
    assert df.isnull().sum().sum() == 0, "prices.csv should be cleaned"


def test_load_macro_returns_dataframe_with_date_index():
    df = load_macro()
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == 'Date'
    assert pd.api.types.is_datetime64_any_dtype(df.index)


def test_load_sector_map_returns_dict_with_string_values():
    sm = load_sector_map()
    assert isinstance(sm, dict)
    assert len(sm) >= 25
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in sm.items())


def test_load_prices_raises_when_file_missing(monkeypatch, tmp_path):
    # Temporarily redirect RAW to an empty dir
    from src.io import loaders
    monkeypatch.setattr(loaders, 'RAW', tmp_path)
    with pytest.raises(PipelineDataError, match="NB01_data_pipeline"):
        load_prices()
