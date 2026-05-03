"""End-to-end pipeline runner.

Usage:
    python run_all.py            # run all 15 notebooks in order
    python run_all.py --from 06  # resume from NB06

Each notebook is executed in-place (output cells updated). Fails fast
on the first error.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

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
    ap = argparse.ArgumentParser()
    ap.add_argument('--from', dest='start_from', default=None,
                    help='Resume from this notebook prefix (e.g. "06")')
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    nb_dir = repo_root / 'notebooks'

    notebooks = NOTEBOOKS
    if args.start_from:
        notebooks = [n for n in NOTEBOOKS if n[:2] >= args.start_from]
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
                cwd=str(src.parent),  # so notebook's `Path().absolute()` walks correctly
            )
        except pm.PapermillExecutionError as e:
            print(f"\n❌ {nb_name} FAILED:")
            print(f"   Cell {e.exec_count}: {e.ename}: {e.evalue}")
            return 1
        elapsed = time.time() - t0
        print(f"✅ {nb_name} OK ({elapsed:.1f}s)")

    total = time.time() - total_start
    print(f"\n🎉 Pipeline complete in {total:.1f}s")
    print(f"   See data/processed/RESULTS.md")
    return 0


if __name__ == '__main__':
    sys.exit(main())
