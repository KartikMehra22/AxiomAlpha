from configs import settings


def test_tickers_is_list_of_30_strings():
    assert isinstance(settings.TICKERS, list)
    assert len(settings.TICKERS) == 30
    assert all(isinstance(t, str) and t.isupper() for t in settings.TICKERS)


def test_date_range_is_strings():
    assert settings.START_DATE == '2019-01-01'
    assert settings.END_DATE == '2024-12-31'


def test_random_seed_is_42():
    assert settings.RANDOM_SEED == 42


def test_paths_resolve():
    """Identify root by markers, not directory name (works on forks/renames)."""
    assert (settings.PROJECT_ROOT / 'configs' / 'settings.py').exists()
    assert (settings.PROJECT_ROOT / 'src').is_dir()
    assert settings.DATA_RAW.is_dir()
    assert settings.DATA_PROC.is_dir()


def test_model_params_are_dicts_with_seed():
    assert settings.RF_PARAMS['random_state'] == 42
    assert settings.XGB_PARAMS['random_state'] == 42


def test_risk_free_rate_is_centralized():
    """RFR is load-bearing for Sharpe/Sortino/Calmar — must live in settings."""
    assert hasattr(settings, 'RISK_FREE_RATE')
    assert 0.0 <= settings.RISK_FREE_RATE <= 0.20  # plausible annual rate
    assert settings.TRADING_DAYS_PER_YEAR == 252
