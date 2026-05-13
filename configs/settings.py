"""Centralized configuration for the AxiomAlpha pipeline.

Notebooks import from here instead of hardcoding constants.
"""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / 'data' / 'raw'
DATA_PROC = PROJECT_ROOT / 'data' / 'processed'
DATA_OUT = PROJECT_ROOT / 'data' / 'outputs'

# --- Universe (30 tickers, 6 sectors × 5 tickers) ---
# Single source of truth. NB01 imports STOCK_UNIVERSE from here instead of
# hardcoding its own copy; data/raw/sector_map.json is regenerated to match.
STOCK_UNIVERSE = {
    'Technology':  ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
    'Financials':  ['JPM',  'BAC',  'GS',   'MS',    'WFC'],
    'Healthcare':  ['JNJ',  'UNH',  'PFE',  'ABBV',  'MRK'],
    'Energy':      ['XOM',  'CVX',  'COP',  'SLB',   'EOG'],
    'Consumer':    ['AMZN', 'TSLA', 'HD',   'MCD',   'NKE'],
    'Industrials': ['BA',   'CAT',  'GE',   'MMM',   'UPS'],
}
TICKERS = [t for tickers in STOCK_UNIVERSE.values() for t in tickers]
SECTOR_MAP = {t: sector for sector, tickers in STOCK_UNIVERSE.items() for t in tickers}

# --- Date range ---
START_DATE = '2019-01-01'
END_DATE = '2024-12-31'

# --- Reproducibility ---
RANDOM_SEED = 42

# --- Statistical / quant ---
GARCH_P = 1
GARCH_Q = 1
ROLLING_GRAPH_WINDOW = 60
N_REGIMES = 3  # Bear, Volatile, Bull

# --- Machine Learning ---
ML_N_SPLITS = 5
RF_PARAMS = dict(
    n_estimators=200, max_depth=6, min_samples_leaf=20,
    random_state=RANDOM_SEED, n_jobs=1,  # n_jobs=1 for determinism
)
XGB_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_SEED, n_jobs=1, eval_metric='logloss',
)
RIDGE_PARAMS = dict(alpha=1.0, random_state=RANDOM_SEED)

# --- NLP ---
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # sentence-transformers
FINBERT_MODEL = 'ProsusAI/finbert'
SYNTHETIC_NEWS_N = 1000

# --- Optimization ---
OPT_RISK_AVERSION = 2.0
OPT_MAX_WEIGHT = 0.10
OPT_LONG_ONLY = True
CVAR_ALPHA = 0.95
COVAR_SHRINKAGE = 'ledoit_wolf'

# --- Backtest ---
BACKTEST_REBALANCE = 'ME'  # month-end (pandas >= 2.2 alias; 'M' is deprecated)
TRANSACTION_COST_BPS = 5      # linear cost per unit of L1 turnover
SLIPPAGE_BPS = 5              # square-root market-impact coefficient (per asset);
                              # total slippage = SLIPPAGE_BPS * sum_i |Δw_i|^1.5.
                              # Set to 0 to disable.
INITIAL_CAPITAL = 1_000_000

# Annualized risk-free rate, used for Sharpe / Sortino / Calmar.
# Roughly the average 3-month T-bill yield over START_DATE..END_DATE.
# Daily rf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR.
RISK_FREE_RATE = 0.04
TRADING_DAYS_PER_YEAR = 252

# --- Visualization ---
DARK_BG = '#0d0d0d'
COLORS = ['#00D4FF', '#FF6B6B', '#51CF66', '#FFD43B', '#FF922B', '#888888']
