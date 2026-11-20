"""Execute notebook code cells in an isolated process without a Jupyter kernel.

This fallback preserves a fresh namespace and the notebook working directory;
it does not save notebook outputs. It supports the repository's Python-only
cells and raises on the first error.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('MPLBACKEND', 'Agg')
from IPython.display import display

path = Path(sys.argv[1]).resolve()
os.chdir(path.parent)
notebook = json.loads(path.read_text())
namespace = {'__name__': '__main__', '__file__': str(path), 'display': display}
for number, cell in enumerate(notebook['cells']):
    if cell.get('cell_type') != 'code':
        continue
    source = ''.join(cell.get('source', []))
    print(f'  cell {number}', flush=True)
    exec(compile(source, f'{path.name}:cell{number}', 'exec'), namespace)
