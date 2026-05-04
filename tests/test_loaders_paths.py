import pytest
from pathlib import Path
from src.io.loaders import (
    _find_project_root,
    PipelineDataError, MissingArtifactError, SchemaError,
    RAW, PROC,
)


def test_find_project_root_returns_repo_root():
    """Project root is identified by markers, not directory name —
    so this works on forks/clones renamed to anything."""
    root = _find_project_root()
    assert (root / 'configs' / 'settings.py').exists()
    assert (root / 'src').is_dir()
    assert (root / 'data').is_dir()


def test_raw_and_proc_constants_exist():
    assert RAW.exists() and RAW.is_dir()
    assert PROC.exists() and PROC.is_dir()


def test_missing_artifact_is_filenotfound_subclass():
    """Existing `except FileNotFoundError` blocks must keep catching missing files."""
    assert issubclass(MissingArtifactError, FileNotFoundError)
    assert issubclass(MissingArtifactError, PipelineDataError)


def test_schema_error_is_value_error_subclass():
    """Schema problems aren't 'file not found' — they're value errors."""
    assert issubclass(SchemaError, ValueError)
    assert issubclass(SchemaError, PipelineDataError)


def test_pipeline_data_error_is_base_exception():
    assert issubclass(PipelineDataError, Exception)
    err = PipelineDataError("test message")
    assert "test message" in str(err)
