"""Smoke test: scan every notebook for forbidden patterns.

Run: python scripts/check_pipeline.py

Exits 0 if all notebooks pass, 1 otherwise. Use as a pre-commit hook
or CI check.

Cells tagged `# noqa: pipeline-check` are skipped (for legitimate
Monte Carlo simulations, optimizer warm-starts, etc.).
"""
import json
import re
import sys
from pathlib import Path

FORBIDDEN = [
    (r'np\.random\.normal\(',      'np.random.normal — fabricates data; use real upstream read'),
    (r'np\.random\.randn\(',       'np.random.randn — fabricates data; use real upstream read'),
    (r'np\.random\.choice\(',      'np.random.choice — fabricates data; use real upstream read'),
    (r'np\.random\.uniform\(',     'np.random.uniform — fabricates data; use real upstream read'),
    (r'np\.random\.rand\(',        'np.random.rand — fabricates data; use real upstream read'),
    (r"parse_dates=\['date'\]",    "use 'Date' (capitalized) consistently"),
    (r'pd\.read_csv\(',            'use src.io.loaders, not direct pd.read_csv'),
    (r'pd\.read_parquet\(',        'use src.io.loaders, not direct pd.read_parquet'),
    (r'joblib\.load\(',            'use src.io.loaders, not direct joblib.load'),
]

ALLOWLIST_TAG = '# noqa: pipeline-check'

# NB01 ingests from yfinance/FRED — exempt from "must use loaders" rule.
# It uses raw pd.read_csv only when reading back its own freshly written outputs
# during validation, which is fine.
EXEMPT_NOTEBOOKS = {'01_data_pipeline'}


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
                # NB01 is allowed pd.read_csv for its own validation reads.
                if nb_stem in EXEMPT_NOTEBOOKS and 'pd.read_csv' in pattern:
                    continue
                errors.append(f"{nb_path.name} cell {cell_idx}: {msg}")

    # Every notebook (except NB01) must import from src.io.loaders.
    if nb_stem not in EXEMPT_NOTEBOOKS:
        full_src = '\n'.join(
            ''.join(c.get('source', []))
            for c in nb.get('cells', [])
            if c.get('cell_type') == 'code'
        )
        if 'from src.io.loaders' not in full_src:
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
