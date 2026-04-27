import json
import os
from pathlib import Path

# Load the notebook
with open('notebooks/02_data_processing.ipynb', 'r') as f:
    nb = json.load(f)

# Clear outputs
for cell in nb.get('cells', []):
    if 'outputs' in cell:
        cell['outputs'] = []

cells = nb['cells']

# 1. Path Standardization Cell
path_cell_source = [
    "import os\n",
    "from pathlib import Path\n",
    "\n",
    "# Standardized paths — works from any directory\n",
    "NOTEBOOK_DIR  = Path().absolute()\n",
    "PROJECT_ROOT  = NOTEBOOK_DIR.parent\n",
    "RAW_DIR       = str(PROJECT_ROOT / 'data' / 'raw')\n",
    "PROCESSED_DIR = str(PROJECT_ROOT / 'data' / 'processed')\n",
    "OUTPUTS_DIR   = str(PROJECT_ROOT / 'data' / 'outputs')\n",
    "\n",
    "# Create directories if they don't exist\n",
    "os.makedirs(RAW_DIR, exist_ok=True)\n",
    "os.makedirs(PROCESSED_DIR, exist_ok=True)\n",
    "os.makedirs(OUTPUTS_DIR, exist_ok=True)\n",
    "\n",
    "print(f\"Project root : {PROJECT_ROOT}\")\n",
    "print(f\"Raw data     : {RAW_DIR}\")\n",
    "print(f\"Processed    : {PROCESSED_DIR}\")\n"
]

path_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": path_cell_source
}

# 2. Loading Cell
loading_cell_source = [
    "# ════════════════════════════════════════\n",
    "# LOAD DATA FROM NOTEBOOK 01\n",
    "# ════════════════════════════════════════\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import json\n",
    "import os\n",
    "\n",
    "# Paths (Standardized)\n",
    "RAW_DIR       = '../data/raw'\n",
    "PROCESSED_DIR = '../data/processed'\n",
    "OUTPUTS_DIR   = '../data/outputs'\n",
    "\n",
    "os.makedirs(PROCESSED_DIR, exist_ok=True)\n",
    "os.makedirs(OUTPUTS_DIR, exist_ok=True)\n",
    "\n",
    "# Load raw data from notebook 01\n",
    "prices = pd.read_csv(\n",
    "    f'{RAW_DIR}/prices.csv',\n",
    "    index_col=0,\n",
    "    parse_dates=True\n",
    ")\n",
    "macro = pd.read_csv(\n",
    "    f'{RAW_DIR}/macro.csv',\n",
    "    index_col=0,\n",
    "    parse_dates=True\n",
    ")\n",
    "with open(f'{RAW_DIR}/sector_map.json') as f:\n",
    "    SECTOR_MAP = json.load(f)\n",
    "with open(f'{RAW_DIR}/tickers.json') as f:\n",
    "    ALL_TICKERS = json.load(f)\n",
    "\n",
    "print(\"✅ Data loaded from notebook 01\")\n",
    "print(f\"   Prices  : {prices.shape}\")\n",
    "print(f\"   Macro   : {macro.shape}\")\n",
    "print(f\"   Tickers : {len(ALL_TICKERS)}\")\n",
    "print(f\"   Date range: {prices.index[0].date()} → {prices.index[-1].date()}\")\n",
    "\n",
    "assert len(prices) > 0, \"Empty prices dataframe loaded\"\n",
    "assert len(macro) > 0, \"Empty macro dataframe loaded\"\n"
]

loading_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": loading_cell_source
}

# Find insertion point (after imports)
first_code_idx = -1
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        first_code_idx = i
        break

if first_code_idx != -1:
    cells.insert(first_code_idx + 1, path_cell)
    cells.insert(first_code_idx + 2, loading_cell)

# 3. Fix savefig/show order and paths
for cell in cells:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'plt.savefig' in line:
                # Replace with the standard savefig format
                filename_match = line.split('plt.savefig(')[1].split(',')[0].strip("'\" )")
                if '/' in filename_match:
                    filename = filename_match.split('/')[-1]
                else:
                    filename = filename_match
                
                # Check if plt.tight_layout() is already there in previous lines of this cell
                if not any('plt.tight_layout()' in l for l in new_source):
                    new_source.append("plt.tight_layout()\n")
                
                new_source.append(f"plt.savefig(f'{{OUTPUTS_DIR}}/{filename}', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')\n")
            elif 'plt.show()' in line:
                # If show comes before savefig in the original, we already moved savefig logic
                # But typically we want Save -> Show -> Close
                # We'll handle savefig separately above.
                new_source.append(line)
                if not any('plt.close()' in l for l in cell['source']):
                    new_source.append("plt.close()\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

# 4. Saving Cell
saving_cell_source = [
    "# ════════════════════════════════════════\n",
    "# SAVE ALL OUTPUTS — Required for notebook 03+\n",
    "# ════════════════════════════════════════\n",
    "\n",
    "# Log returns\n",
    "log_returns.to_csv(f'{PROCESSED_DIR}/log_returns.csv')\n",
    "\n",
    "# Simple returns  \n",
    "simple_returns.to_csv(f'{PROCESSED_DIR}/simple_returns.csv')\n",
    "\n",
    "# Raw features (long format)\n",
    "features_raw.to_csv(\n",
    "    f'{PROCESSED_DIR}/features_raw.csv', \n",
    "    index=False\n",
    ")\n",
    "\n",
    "# Normalized features\n",
    "features_normalized.to_csv(\n",
    "    f'{PROCESSED_DIR}/features_normalized.csv',\n",
    "    index=False\n",
    ")\n",
    "\n",
    "print(\"✅ Notebook 02 outputs saved\")\n",
    "print(f\"   log_returns.csv          : {log_returns.shape}\")\n",
    "print(f\"   simple_returns.csv       : {simple_returns.shape}\")\n",
    "print(f\"   features_raw.csv         : {features_raw.shape}\")\n",
    "print(f\"   features_normalized.csv  : {features_normalized.shape}\")\n"
]

saving_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": saving_cell_source
}

cells.append(saving_cell)

# Save the fixed notebook
with open('notebooks/02_data_processing_fixed.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Saved to 02_data_processing_fixed.ipynb")
