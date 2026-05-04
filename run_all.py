"""End-to-end pipeline runner.

Usage:
    python run_all.py              # run all 15 notebooks in order
    python run_all.py --from 06    # resume from NB06
    python run_all.py --skip-check # skip the static check_pipeline.py gate

Each notebook is executed in-place (output cells updated). Fails fast on the
first error.

Reproducibility: this script seeds numpy + python random + PYTHONHASHSEED
from configs.settings.RANDOM_SEED before papermill is invoked. NB01 itself
hits external APIs (yfinance / FRED) so it remains non-deterministic;
`scripts/check_determinism.py` therefore runs `--from 02`.
"""
from __future__ import annotations

import argparse
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import papermill as pm

from configs.settings import RANDOM_SEED


NOTEBOOKS = [
    '01_data_pipeline',
    '02_data_processing',
    '03_quant_core',
    '04_graph_engine',
    '05_graph_analysis',
    '06_ml_models',
    '07_ml_analysis',
    '08_nlp_sentiment',
    '09_rag_setup',
    '10_nlp_analysis',
    '11_portfolio_optimization',
    '12_backtesting',
    '13_agent_system',
    '14_agent_orchestration',
    '15_agent_demo',
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--from', dest='start_from', default=None,
                    help='Resume from this notebook prefix (e.g. "06" or "6").')
    ap.add_argument('--skip-check', action='store_true',
                    help='Skip the scripts/check_pipeline.py gate.')
    return ap.parse_args()


def normalize_start(start: str) -> str:
    """Accept '6', '06', or '06_data_processing'. Returns canonical 2-digit prefix."""
    m = re.match(r'^(\d{1,2})', start)
    if not m:
        raise SystemExit(f"--from must start with a number, got {start!r}")
    n = int(m.group(1))
    if not 1 <= n <= len(NOTEBOOKS):
        raise SystemExit(f"--from must be 1..{len(NOTEBOOKS)}, got {n}")
    return f"{n:02d}"


def seed_everything(seed: int) -> None:
    """Seed every RNG papermill might inherit before kernel start."""
    os.environ.setdefault('PYTHONHASHSEED', str(seed))
    random.seed(seed)
    np.random.seed(seed)


def run_check_pipeline(repo_root: Path) -> int:
    """Run the static notebook check as a pre-gate."""
    print("▶ Running scripts/check_pipeline.py …")
    result = subprocess.run(
        [sys.executable, 'scripts/check_pipeline.py'],
        cwd=str(repo_root),
    )
    return result.returncode


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    nb_dir = repo_root / 'notebooks'

    seed_everything(RANDOM_SEED)

    if not args.skip_check:
        rc = run_check_pipeline(repo_root)
        if rc != 0:
            print("\n❌ check_pipeline.py failed — fix the violations or "
                  "re-run with --skip-check.")
            return rc

    notebooks = NOTEBOOKS
    if args.start_from:
        prefix = normalize_start(args.start_from)
        notebooks = [n for n in NOTEBOOKS if n[:2] >= prefix]
        if not notebooks:
            print(f"❌ No notebooks at or after {prefix}.")
            return 1
        print(f"▶ Resuming from {notebooks[0]}")

    total_start = time.time()
    for nb_name in notebooks:
        src = nb_dir / nb_name / f'{nb_name}.ipynb'
        if not src.exists():
            print(f"❌ Notebook not found: {src}")
            return 1

        print(f"\n{'='*70}\n▶ {nb_name}\n{'='*70}")
        t0 = time.time()
        try:
            pm.execute_notebook(
                str(src), str(src),  # in-place
                kernel_name='python3',
                progress_bar=False,
                cwd=str(src.parent),
            )
        except pm.PapermillExecutionError as e:
            print(f"\n❌ {nb_name} FAILED:")
            cell = getattr(e, 'exec_count', '?')
            ename = getattr(e, 'ename', type(e).__name__)
            evalue = getattr(e, 'evalue', str(e))
            print(f"   Cell {cell}: {ename}: {evalue}")
            return 1
        elapsed = time.time() - t0
        print(f"✅ {nb_name} OK ({elapsed:.1f}s)")

    total = time.time() - total_start
    print(f"\n🎉 Pipeline complete in {total:.1f}s")
    print(f"   See data/processed/RESULTS.md")
    return 0


if __name__ == '__main__':
    sys.exit(main())
