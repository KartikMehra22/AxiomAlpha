"""End-to-end gate: run scripts/check_pipeline.py against the real notebooks.

This is the "no fabricated data, no direct file reads" smoke gate. It runs
on every `pytest` invocation when notebooks are present.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.check_pipeline import check_notebook


REPO_ROOT = Path(__file__).resolve().parent.parent
NB_DIR = REPO_ROOT / 'notebooks'


def _all_notebook_paths():
    return sorted(NB_DIR.glob('*/*.ipynb'))


@pytest.mark.skipif(not NB_DIR.is_dir(), reason='notebooks/ not present')
@pytest.mark.parametrize(
    'nb_path',
    _all_notebook_paths(),
    ids=lambda p: p.parent.name,
)
def test_notebook_passes_static_check(nb_path):
    """Each notebook must pass scripts/check_pipeline.py."""
    errors = check_notebook(nb_path)
    if errors:
        pytest.fail(
            f"{nb_path.name} failed static check:\n  - "
            + "\n  - ".join(errors)
        )
