import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('ijcnn_pilot', Path(__file__).parents[1] / 'src' / 'pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def sample():
    dates = pd.bdate_range('2020-09-01', '2023-02-28')
    rng = np.random.default_rng(123)
    changes = rng.normal(0, .012, (len(dates), 4))
    return pd.DataFrame(100 * np.exp(np.cumsum(changes, axis=0)), index=dates, columns=list('ABCD'))


def test_target_is_exactly_future_twenty_days():
    prices = sample()
    data = pilot.construct(prices)
    i = 130
    returns = prices.pct_change(fill_method=None).iloc[1:].mean(axis=1)
    market = np.log1p(returns)
    expected = np.sqrt(np.mean(np.square(market.iloc[i + 1:i + 21])))
    assert np.isclose(data.future_rms_20.iloc[i], expected)
    assert data.label_end.iloc[i] == data.index[i + 20]
    assert data.future_rms_20.iloc[-20:].isna().all()


def test_future_prices_do_not_change_existing_features():
    prices = sample()
    cutoff = prices.index[200]
    changed = prices.copy()
    changed.loc[changed.index > cutoff] *= np.linspace(1, 2, len(changed.columns))
    before = pilot.construct(prices).loc[cutoff]
    after = pilot.construct(changed).loc[cutoff]
    for key in ('vol_5', 'vol_20', 'vol_60', 'return_20', 'avg_correlation_60'):
        assert np.isclose(before[key], after[key])


def test_threshold_and_splits_exclude_unmatured_labels():
    prices = sample()
    data = pilot.construct(prices)
    train, valid, threshold = pilot.split(data)
    assert train.label_end.max() <= pilot.TRAIN_END
    assert valid.label_end.max() <= pilot.VALID_END
    assert valid.index.min() >= pilot.VALID_START
    modified = data.copy()
    modified.loc[modified.index >= pilot.VALID_START, 'future_rms_20'] *= 100
    _, _, new_threshold = pilot.split(modified)
    assert threshold == new_threshold


def test_invalid_or_unsorted_price_data_is_rejected(tmp_path):
    frame = sample().iloc[:3].reset_index(names='Date')
    path = tmp_path / 'bad.csv'
    frame.iloc[::-1].to_csv(path, index=False)
    import pytest
    with pytest.raises(ValueError, match='sorted'):
        pilot.read_prices(path)
