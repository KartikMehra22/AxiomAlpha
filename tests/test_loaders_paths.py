import pytest
from pathlib import Path
from src.io.loaders import _find_project_root, PipelineDataError, RAW, PROC


def test_find_project_root_returns_axiomalpha_dir():
    root = _find_project_root()
    assert root.name == 'AxiomAlpha'
    assert (root / 'data').is_dir()


def test_raw_and_proc_constants_exist():
    assert RAW.exists() and RAW.is_dir()
    assert PROC.exists() and PROC.is_dir()


def test_pipeline_data_error_is_filenotfound_subclass():
    assert issubclass(PipelineDataError, FileNotFoundError)
    err = PipelineDataError("test message")
    assert "test message" in str(err)
