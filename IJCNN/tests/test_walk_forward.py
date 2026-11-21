import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path(__file__).parents[1] / 'src'
sys.path.insert(0, str(SRC))
spec = importlib.util.spec_from_file_location('ijcnn_walk_forward', SRC / 'walk_forward.py')
wf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wf)


def test_walk_forward_purges_at_each_year_and_freezes_threshold():
    dates = pd.bdate_range('2020-01-01', '2024-12-31')
    rng = np.random.default_rng(5)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, .015, (len(dates), 4)), axis=0)), index=dates)
    data = wf.construct(prices)
    for year in wf.FOLDS:
        train, validation, threshold = wf.make_fold(data, year)
        start = pd.Timestamp(year=year, month=1, day=1)
        assert train.label_end.max() < start
        assert validation.index.min() >= start
        assert validation.label_end.max() <= pd.Timestamp(year=year, month=12, day=31)
        modified = data.copy()
        modified.loc[modified.index >= start, 'future_rms_20'] *= 500
        _, _, next_threshold = wf.make_fold(modified, year)
        assert threshold == next_threshold


def test_paired_blocks_identical_predictions_have_zero_difference():
    y = np.tile([0, 1, 0, 0], 20)
    p = np.linspace(.1, .8, len(y))
    summary = wf.paired_block_interval(y, p, p, repetitions=100, seed=1)
    assert summary['ap_a_minus_b']['interval_95'] == [0.0, 0.0]
    assert summary['brier_a_minus_b']['interval_95'] == [0.0, 0.0]


def test_single_class_bootstrap_does_not_fabricate_ap():
    y = np.zeros(30, dtype=int)
    summary = wf.paired_block_interval(y, np.full(30, .1), np.full(30, .2), repetitions=20)
    assert summary['ap_a_minus_b']['valid_replicates'] == 0
    assert summary['ap_a_minus_b']['interval_95'] is None
