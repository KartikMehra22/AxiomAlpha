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
    assert settings.PROJECT_ROOT.name == 'AxiomAlpha'
    assert settings.DATA_RAW.is_dir()
    assert settings.DATA_PROC.is_dir()


def test_model_params_are_dicts_with_seed():
    assert settings.RF_PARAMS['random_state'] == 42
    assert settings.XGB_PARAMS['random_state'] == 42
