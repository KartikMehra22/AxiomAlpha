import json
import subprocess
import sys
from pathlib import Path


def test_direct_runner_runs_cells_in_order(tmp_path):
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'execute_notebook_direct.py'
    nb = tmp_path / 'example.ipynb'
    nb.write_text(json.dumps({'cells': [
        {'cell_type': 'code', 'source': ['value = 4\n']},
        {'cell_type': 'markdown', 'source': ['ignored']},
        {'cell_type': 'code', 'source': ["open('result.txt', 'w').write(str(value + 3))\n"]},
    ]}))
    subprocess.run([sys.executable, str(script), str(nb)], check=True, cwd=tmp_path)
    assert (tmp_path / 'result.txt').read_text() == '7'
