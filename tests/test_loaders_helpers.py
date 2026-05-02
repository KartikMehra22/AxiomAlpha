import pytest
import pandas as pd
import numpy as np
from src.io.loaders import _require, _validate, PipelineDataError


def test_require_returns_path_when_exists(tmp_path):
    p = tmp_path / 'x.csv'
    p.write_text('hello')
    assert _require(p, producer_nb='NB01') == p


def test_require_raises_with_producer_in_message(tmp_path):
    p = tmp_path / 'missing.csv'
    with pytest.raises(PipelineDataError, match="NB01_data_pipeline"):
        _require(p, producer_nb='NB01_data_pipeline')


def test_validate_passes_when_columns_present():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    out = _validate(df, required_cols=['a', 'b'], name='toy')
    assert out is df


def test_validate_raises_when_column_missing():
    df = pd.DataFrame({'a': [1]})
    with pytest.raises(PipelineDataError, match="missing columns"):
        _validate(df, required_cols=['a', 'b'], name='toy')


def test_validate_monotonic_index_passes_when_sorted():
    df = pd.DataFrame({'x': [1, 2, 3]},
                      index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03']))
    _validate(df, required_cols=[], name='toy', monotonic_index=True)


def test_validate_monotonic_index_raises_when_unsorted():
    df = pd.DataFrame({'x': [1, 2]},
                      index=pd.to_datetime(['2024-01-02', '2024-01-01']))
    with pytest.raises(PipelineDataError, match="monotonic"):
        _validate(df, required_cols=[], name='toy', monotonic_index=True)
