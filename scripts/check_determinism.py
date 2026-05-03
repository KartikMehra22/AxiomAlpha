"""Verify the pipeline is deterministic from NB02 onward.

NB01 is excluded because it hits yfinance/FRED — external APIs that may
return slightly different bytes across calls (split adjustments, backfills).
For determinism we hold `data/raw/prices.csv` constant and re-run NB02-15.

Workflow:
  1. Run `python run_all.py --from 02` once (snapshot 1).
  2. Snapshot SHA-256 of every CSV in data/processed/.
  3. Run `python run_all.py --from 02` again (snapshot 2).
  4. Compare hashes; fail if any CSV differs.

JSON files (which embed timestamps) are excluded from the check.

Run: python scripts/check_determinism.py
"""
import hashlib
import subprocess
import sys
from pathlib import Path


def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def snapshot(proc_dir: Path) -> dict[str, str]:
    return {p.name: file_hash(p) for p in sorted(proc_dir.glob('*.csv'))}


def run_pipeline(repo_root: Path):
    print(f"▶ Running run_all.py --from 02 …")
    result = subprocess.run(
        [sys.executable, 'run_all.py', '--from', '02'],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"❌ run_all.py failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    proc = repo_root / 'data' / 'processed'

    run_pipeline(repo_root)
    before = snapshot(proc)
    if not before:
        print("❌ no CSVs in data/processed/ after first run")
        return 1

    run_pipeline(repo_root)
    after = snapshot(proc)

    diffs = [k for k in before if before[k] != after.get(k)]
    if diffs:
        print(f"❌ {len(diffs)} files differ between runs (non-deterministic):")
        for k in diffs:
            print(f"  - {k}: {before[k][:12]} → {after[k][:12]}")
        return 1

    print(f"✅ All {len(before)} CSVs are bit-identical across runs.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
