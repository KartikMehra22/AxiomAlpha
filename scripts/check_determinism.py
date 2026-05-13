"""Verify the pipeline is deterministic from NB02 onward.

NB01 is excluded because it hits yfinance/FRED — external APIs that may
return slightly different bytes across calls (split adjustments, backfills).
For determinism we hold `data/raw/prices.csv` constant and re-run NB02-15.

Workflow:
  1. Run `python run_all.py --from 02` once (snapshot 1).
  2. Snapshot a content hash of every CSV/NPY/JSON in data/processed/.
  3. Run `python run_all.py --from 02` again (snapshot 2).
  4. Compare hashes; fail if any artifact differs.

Coverage:
  - CSV     : raw bytes hash (canonical line-ending after pandas to_csv).
  - NPY     : raw bytes hash (numpy save format is layout-stable).
  - JSON    : parsed → volatile keys stripped → re-serialized canonically
              → hashed. Strips known timestamp keys
              (`generated_at`, `timestamp`, `created_at`, `run_at`).
  - PKL/JOBLIB : excluded. Joblib output embeds file mtimes and sklearn
              objects sometimes carry non-deterministic memo'd state.
              These should be regenerated rather than diffed.
  - BIN     : excluded (faiss index byte layout depends on build flags).

Run: python scripts/check_determinism.py
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


# Volatile JSON keys stripped before hashing so a wall-clock timestamp can't
# show up as non-determinism. Match by key name anywhere in the tree.
VOLATILE_JSON_KEYS = {
    'generated_at', 'timestamp', 'created_at', 'run_at',
    'date_generated', 'now', 'today',
}


def _strip_volatile(obj):
    """Recursively remove keys whose names are in VOLATILE_JSON_KEYS."""
    if isinstance(obj, dict):
        return {
            k: _strip_volatile(v)
            for k, v in obj.items()
            if k not in VOLATILE_JSON_KEYS
        }
    if isinstance(obj, list):
        return [_strip_volatile(x) for x in obj]
    return obj


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(p: Path) -> str:
    suffix = p.suffix.lower()
    if suffix == '.json':
        try:
            obj = json.loads(p.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            return _hash_bytes(p.read_bytes())
        canonical = json.dumps(_strip_volatile(obj), sort_keys=True,
                              separators=(',', ':'))
        return _hash_bytes(canonical.encode('utf-8'))
    return _hash_bytes(p.read_bytes())


def snapshot(proc_dir: Path) -> dict[str, str]:
    """Hash every CSV / NPY / JSON in the processed dir."""
    out: dict[str, str] = {}
    for ext in ('*.csv', '*.npy', '*.json'):
        for p in sorted(proc_dir.glob(ext)):
            out[p.name] = _hash_file(p)
    return out


def run_pipeline(repo_root: Path) -> None:
    print(">>> Running run_all.py --from 02 ...")
    result = subprocess.run(
        [sys.executable, 'run_all.py', '--from', '02', '--skip-check'],
        cwd=str(repo_root), capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[FAIL] run_all.py failed:\n{result.stdout}\n{result.stderr}")
        sys.exit(1)


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    proc = repo_root / 'data' / 'processed'

    run_pipeline(repo_root)
    before = snapshot(proc)
    if not before:
        print("[FAIL] no CSVs/NPYs/JSONs in data/processed/ after first run")
        return 1

    run_pipeline(repo_root)
    after = snapshot(proc)

    diffs = [k for k in before if before[k] != after.get(k)]
    if diffs:
        print(f"[FAIL] {len(diffs)} files differ between runs (non-deterministic):")
        for k in diffs:
            print(f"  - {k}: {before[k][:12]} -> {after[k][:12]}")
        return 1

    new_files = sorted(set(after) - set(before))
    if new_files:
        print(f"[WARN] {len(new_files)} files appeared in run 2 only "
              f"(non-deterministic file set): {new_files}")
        return 1

    print(f"[ OK ] All {len(before)} CSV/NPY/JSON artifacts are bit-identical "
          "across runs.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
