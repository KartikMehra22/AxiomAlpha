"""Exploratory yearly walk-forward baselines; all years are development data."""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pilot import ROOT, SOURCE, construct, read_prices

FEATURES = ['vol_5', 'vol_20', 'vol_60', 'return_20']
FOLDS = (2022, 2023, 2024)


def make_fold(data: pd.DataFrame, year: int):
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year, month=12, day=31)
    usable = data.dropna(subset=FEATURES + ['avg_correlation_60', 'future_rms_20', 'label_end'])
    train = usable[(usable.index < start) & (usable.label_end < start)]
    evaluation = usable[(usable.index >= start) & (usable.index <= end) & (usable.label_end <= end)]
    if train.empty or evaluation.empty:
        raise ValueError(f'Fold {year} empty after purging')
    cutoff = float(train.future_rms_20.quantile(.8))
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('Invalid threshold')
    return train, evaluation, cutoff


def paired_block_interval(y, prediction_a, prediction_b, block_length=25, repetitions=1000, seed=173):
    """Percentile moving-block intervals for AP and Brier A minus B, conditional on this fold."""
    y = np.asarray(y, dtype=int)
    a = np.asarray(prediction_a, dtype=float)
    b = np.asarray(prediction_b, dtype=float)
    n = len(y)
    if n == 0 or len(a) != n or len(b) != n or block_length < 1 or repetitions < 1:
        raise ValueError('Invalid bootstrap input')
    width = min(block_length, n)
    rng = np.random.default_rng(seed)
    ap, brier = [], []
    for _ in range(repetitions):
        starts = rng.integers(0, n - width + 1, size=int(np.ceil(n / width)))
        idx = (starts[:, None] + np.arange(width)).ravel()[:n]
        yy = y[idx]
        brier.append(brier_score_loss(yy, a[idx]) - brier_score_loss(yy, b[idx]))
        if np.unique(yy).size == 2:
            ap.append(average_precision_score(yy, a[idx]) - average_precision_score(yy, b[idx]))
    def summary(values):
        return {'valid_replicates': len(values), 'interval_95': [float(z) for z in np.quantile(values, [.025, .975])] if values else None}
    return {'block_length': width, 'repetitions': repetitions, 'seed': seed, 'ap_a_minus_b': summary(ap), 'brier_a_minus_b': summary(brier)}


def score(y, p):
    two_classes = np.unique(y).size == 2
    return {'average_precision': float(average_precision_score(y, p)) if two_classes else None,
            'roc_auc': float(roc_auc_score(y, p)) if two_classes else None,
            'brier': float(brier_score_loss(y, p))}


def run(source: Path, output: Path, repetitions: int):
    prices = read_prices(source)
    data = construct(prices)
    all_results = []
    prediction_frames = []
    for year in FOLDS:
        train, valid, threshold = make_fold(data, year)
        y_train = (train.future_rms_20 > threshold).astype(int)
        y_valid = (valid.future_rms_20 > threshold).astype(int)
        if y_train.nunique() != 2:
            raise ValueError(f'Fold {year} training data has only one class')
        feature_sets = {'logistic_volatility': FEATURES,
                        'logistic_volatility_correlation': FEATURES + ['avg_correlation_60'],
                        'boosted_volatility': FEATURES,
                        'boosted_volatility_correlation': FEATURES + ['avg_correlation_60']}
        preds = {'prevalence': np.full(len(valid), float(y_train.mean()))}
        for name, features in feature_sets.items():
            if name.startswith('logistic'):
                model = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=17))
            else:
                model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7, l2_regularization=1.0, random_state=17)
            model.fit(train[features], y_train)
            preds[name] = model.predict_proba(valid[features])[:, 1]
        y = y_valid.to_numpy()
        runs = int(((y == 1) & np.r_[True, y[:-1] == 0]).sum())
        fold = {'year': year, 'train_n': len(train), 'evaluation_n': len(valid),
                'last_training_label': str(train.label_end.max().date()),
                'first_evaluation_origin': str(valid.index.min().date()),
                'last_evaluation_label': str(valid.label_end.max().date()),
                'stress_threshold': threshold, 'train_prevalence': float(y_train.mean()),
                'evaluation_prevalence': float(y_valid.mean()), 'positive_days': int(y.sum()),
                'positive_runs_not_independent_episodes': runs,
                'metrics': {name: score(y, p) for name, p in preds.items()},
                'bootstrap_logistic_correlation_minus_volatility': paired_block_interval(
                    y, preds['logistic_volatility_correlation'], preds['logistic_volatility'], repetitions=repetitions)}
        all_results.append(fold)
        prediction_frames.append(pd.DataFrame({'date': valid.index, 'label_end': valid.label_end, 'fold': year,
                                                'label': y_valid, 'threshold': threshold, **preds}))
    output.mkdir(parents=True, exist_ok=True)
    pd.concat(prediction_frames).to_csv(output / 'development_predictions.csv', index=False)
    report = {'status': 'exploratory_development_not_untouched_holdout',
              'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'training': 'expanding annual; labels mature before each evaluation year; no tuning',
              'target': '20-trading-day future RMS of equal-weight proxy log returns > training-fold 80th percentile',
              'notes': ['Annual evaluation labels must mature within their calendar year; boundary observations are omitted.',
                        'Threshold changes between folds; AP depends on prevalence; do not pool fold labels as one fixed event.',
                        'Block-bootstrap intervals are exploratory within-fold sensitivity summaries, not confirmation; overlap and few events limit inference.',
                        'All years are development data; no independent holdout, graph or neural model.',
                        'Fixed selected stock universe; no point-in-time constituent, availability or economic-value claim.'],
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'scikit_learn': sklearn.__version__},
              'folds': all_results}
    (output / 'development_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    for fold in all_results:
        print(f"{fold['year']} n={fold['evaluation_n']} positives={fold['positive_days']} prevalence={fold['evaluation_prevalence']:.3f} runs={fold['positive_runs_not_independent_episodes']}")
        for model, metrics in fold['metrics'].items():
            print(f"  {model}: AP={metrics['average_precision']} Brier={metrics['brier']:.4f}")
        print('  correlation AP difference interval:', fold['bootstrap_logistic_correlation_minus_volatility']['ap_a_minus_b'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prices', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=ROOT / 'IJCNN' / 'results' / 'walk_forward')
    parser.add_argument('--bootstrap-repetitions', type=int, default=1000)
    args = parser.parse_args()
    run(args.prices.resolve(), args.output.resolve(), args.bootstrap_repetitions)
