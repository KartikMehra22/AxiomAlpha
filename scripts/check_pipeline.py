"""Smoke test: scan every notebook for forbidden patterns.

Run: python scripts/check_pipeline.py

Exits 0 if all notebooks pass, 1 otherwise. Use as a pre-commit hook
or CI check (also wired into `tests/test_check_pipeline_runs.py`).

Cells tagged `# noqa: pipeline-check` are skipped (for legitimate
Monte Carlo simulations, optimizer warm-starts, etc.).
"""
import json
import re
import sys
from pathlib import Path

# (regex, human-readable reason). Patterns target both the `np.` and
# `numpy.` prefixes, plus the `rng.` style (rng = np.random.default_rng(...)).
FORBIDDEN = [
    # Data-fabrication in numpy.random (any prefix).
    (r'\b(?:np|numpy)\.random\.normal\(',
        'np.random.normal — fabricates data; use real upstream read'),
    (r'\b(?:np|numpy)\.random\.standard_normal\(',
        'np.random.standard_normal — fabricates data'),
    (r'\b(?:np|numpy)\.random\.randn\(',
        'np.random.randn — fabricates data'),
    (r'\b(?:np|numpy)\.random\.choice\(',
        'np.random.choice — fabricates data'),
    (r'\b(?:np|numpy)\.random\.uniform\(',
        'np.random.uniform — fabricates data'),
    (r'\b(?:np|numpy)\.random\.rand\(',
        'np.random.rand — fabricates data'),
    (r'\b(?:np|numpy)\.random\.dirichlet\(',
        'np.random.dirichlet — fabricates data'),
    # Generator-style fabrication: `rng = np.random.default_rng(...)` then
    # `rng.normal(...)`. This is the modern API but equally bad as a
    # data source unless tagged for Monte Carlo.
    (r'\brng\.(?:normal|standard_normal|standard_t|randn|choice|uniform|rand|dirichlet)\(',
        'rng.* call — fabricates data; use real upstream read'),
    # Direct file reads bypassing the loaders.
    (r'parse_dates=\[\'date\'\]',
        "use 'Date' (capitalized) consistently"),
    (r'\bpd\.read_csv\(',
        'use src.io.loaders, not direct pd.read_csv'),
    (r'\bpd\.read_parquet\(',
        'use src.io.loaders, not direct pd.read_parquet'),
    (r'\bjoblib\.load\(',
        'use src.io.loaders, not direct joblib.load'),
]

ALLOWLIST_TAG = '# noqa: pipeline-check'

# NB01 ingests from yfinance/FRED — exempt from "must use loaders" rule and
# from the pd.read_csv rule (it validates its own freshly written outputs).
EXEMPT_NOTEBOOKS = {'01_data_pipeline'}

# Patterns that NB01 is allowed to use for self-validation.
NB01_EXEMPT_PATTERNS = {'pd.read_csv', 'pd.read_parquet'}


def _strip_comments_and_strings(line: str) -> str:
    """Return the line with anything after a `#` removed (best-effort).
    Doesn't try to handle strings — fine for our regex needs."""
    if '#' not in line:
        return line
    # Cheap split — string `#` content gets stripped too, but our forbidden
    # patterns don't include `#` anyway, so false negatives are unlikely.
    return line.split('#', 1)[0]


def check_notebook(nb_path: Path) -> list[str]:
    errors: list[str] = []
    nb = json.loads(nb_path.read_text())
    nb_stem = nb_path.stem

    for cell_idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue
        src = ''.join(cell.get('source', []))
        if ALLOWLIST_TAG in src:
            continue
        for pattern, msg in FORBIDDEN:
            if re.search(pattern, src):
                # NB01 is allowed pd.read_csv / pd.read_parquet for its own
                # validation reads of freshly written ingestion outputs.
                if nb_stem in EXEMPT_NOTEBOOKS and any(
                    p in pattern for p in NB01_EXEMPT_PATTERNS
                ):
                    continue
                errors.append(f"{nb_path.name} cell {cell_idx}: {msg}")

    # Every notebook (except NB01) must import from src.io.loaders. Check on
    # comment-stripped lines so a docstring/comment can't satisfy the rule.
    if nb_stem not in EXEMPT_NOTEBOOKS:
        has_import = False
        for cell in nb.get('cells', []):
            if cell.get('cell_type') != 'code':
                continue
            for line in ''.join(cell.get('source', [])).splitlines():
                if 'from src.io.loaders' in _strip_comments_and_strings(line):
                    has_import = True
                    break
            if has_import:
                break
        if not has_import:
            errors.append(
                f"{nb_path.name}: missing 'from src.io.loaders import ...' "
                f"(every notebook must read inputs through the loader module)"
            )
    return errors


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    nb_dir = repo_root / 'notebooks'
    if not nb_dir.is_dir():
        print(f"❌ notebooks/ not found at {nb_dir}")
        return 1

    all_errors: list[str] = []
    nb_paths = sorted(nb_dir.glob('*/*.ipynb'))
    if not nb_paths:
        print(f"❌ no notebooks found under {nb_dir}")
        return 1

    for nb_path in nb_paths:
        all_errors.extend(check_notebook(nb_path))

    if all_errors:
        print('❌ Pipeline check failed:')
        for e in all_errors:
            print(f'  - {e}')
        return 1

    print(f'✅ All {len(nb_paths)} notebooks pass pipeline check.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
