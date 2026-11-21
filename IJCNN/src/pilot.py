"""Reproducible development-only market-stress pilot; no portfolio data is modified."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data' / 'raw' / 'prices.csv'
HORIZON = 20
TRAIN_END = pd.Timestamp('2021-12-31')
VALID_START = pd.Timestamp('2022-01-01')
VALID_END = pd.Timestamp('2022-12-31')


def read_prices(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if 'Date' not in raw or raw.shape[1] < 3:
        raise ValueError('Expected Date and at least two price columns')
    dates = pd.to_datetime(raw.pop('Date'), errors='raise')
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be unique and sorted')
    prices = raw.apply(pd.to_numeric, errors='raise').set_axis(dates)
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise ValueError('Prices must be finite and positive; missingness is not imputed')
    return prices


def construct(prices: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    """At close t use features through t; label is RMS market log return t+1..t+H."""
    if horizon < 1:
        raise ValueError('horizon must be positive')
    returns = prices.pct_change(fill_method=None).iloc[1:]
    market_simple = returns.mean(axis=1)
    if (market_simple <= -1).any():
        raise ValueError('Invalid market return')
    market = np.log1p(market_simple)
    output = pd.DataFrame(index=market.index)
    for n in (5, 20, 60):
        output[f'vol_{n}'] = market.pow(2).rolling(n, min_periods=n).mean().pow(.5)
    output['return_20'] = market.rolling(20, min_periods=20).sum()
    # Mean of unique, signed pairwise correlations on the trailing 60 dates.
    values = returns.to_numpy(dtype=float)
    corrs = np.full(len(values), np.nan)
    for i in range(59, len(values)):
        window = values[i - 59:i + 1]
        if (window.std(axis=0) <= 1e-12).any():
            continue
        mat = np.corrcoef(window, rowvar=False)
        corrs[i] = mat[np.triu_indices_from(mat, k=1)].mean()
    output['avg_correlation_60'] = corrs
    output['future_rms_20'] = market.pow(2).iloc[::-1].rolling(horizon, min_periods=horizon).mean().iloc[::-1].shift(-1).pow(.5)
    output['label_end'] = pd.Series(market.index, index=market.index).shift(-horizon)
    return output


def split(data: pd.DataFrame):
    features = ['vol_5', 'vol_20', 'vol_60', 'return_20', 'avg_correlation_60']
    usable = data.dropna(subset=features + ['future_rms_20', 'label_end'])
    train = usable[(usable.index <= TRAIN_END) & (usable.label_end <= TRAIN_END)]
    # Validation labels must mature in 2022; 2023 onward is not evaluated.
    valid = usable[(usable.index >= VALID_START) & (usable.index <= VALID_END) & (usable.label_end <= VALID_END)]
    if train.empty or valid.empty:
        raise ValueError('Empty split after purging')
    threshold = float(train.future_rms_20.quantile(.8))
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError('Invalid training threshold')
    return train, valid, threshold


def run(source: Path, result_dir: Path):
    prices = read_prices(source)
    dataset = construct(prices)
    train, valid, threshold = split(dataset)
    y_train = (train.future_rms_20 > threshold).astype(int)
    y_valid = (valid.future_rms_20 > threshold).astype(int)
    if y_train.nunique() != 2:
        raise ValueError('Training split requires both classes')
    feature_sets = {
        'volatility': ['vol_5', 'vol_20', 'vol_60', 'return_20'],
        'volatility_and_correlation': ['vol_5', 'vol_20', 'vol_60', 'return_20', 'avg_correlation_60'],
    }
    predictions = {'historical_prevalence': np.full(len(valid), y_train.mean())}
    for name, features in feature_sets.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=17))
        model.fit(train[features], y_train)
        predictions[name] = model.predict_proba(valid[features])[:, 1]
    metrics = {}
    for name, p in predictions.items():
        metrics[name] = {'brier': float(brier_score_loss(y_valid, p)),
                         'average_precision': float(average_precision_score(y_valid, p)) if y_valid.nunique() == 2 else None,
                         'roc_auc': float(roc_auc_score(y_valid, p)) if y_valid.nunique() == 2 else None}
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({'date': valid.index, 'label_end': valid.label_end, 'label': y_valid, **predictions}).to_csv(result_dir / 'validation_predictions.csv', index=False)
    output = {'status': 'development_only_not_holdout', 'source': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'price_start': str(prices.index.min().date()), 'price_end': str(prices.index.max().date()), 'stocks': list(prices.columns),
        'market_proxy': 'daily equal-weight rebalanced arithmetic stock return, converted with log1p',
        'train_end': str(TRAIN_END.date()), 'validation_start': str(VALID_START.date()), 'validation_end': str(VALID_END.date()),
        'train_n': len(train), 'validation_n': len(valid), 'threshold_train_only': threshold,
        'train_positive': int(y_train.sum()), 'validation_positive': int(y_valid.sum()),
        'train_prevalence': float(y_train.mean()), 'validation_prevalence': float(y_valid.mean()),
        'feature_sets': feature_sets, 'metrics': metrics,
        'caveats': ['Pilot only; selected 30-stock universe, not point-in-time.', 'Equal-weight proxy overlaps the model stock universe.', 'Validation window has been inspected; it is not a final holdout.', 'Threshold fitted to matured training labels; no tuning or calibration performed.', 'No causal, neural-graph or investment claim.']}
    (result_dir / 'validation_metrics.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prices', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT / 'IJCNN' / 'results' / 'pilot')
    args = parser.parse_args()
    run(args.prices.resolve(), args.output.resolve())
