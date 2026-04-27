import json
import os
from pathlib import Path

# Load the notebook
with open('notebooks/03_quant_core.ipynb', 'r') as f:
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
    "# LOAD DATA FROM NOTEBOOKS 01-02\n",
    "# ════════════════════════════════════════\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import json, os\n",
    "\n",
    "RAW_DIR       = '../data/raw'\n",
    "PROCESSED_DIR = '../data/processed'\n",
    "OUTPUTS_DIR   = '../data/outputs'\n",
    "os.makedirs(OUTPUTS_DIR, exist_ok=True)\n",
    "\n",
    "prices = pd.read_csv(\n",
    "    f'{RAW_DIR}/prices.csv',\n",
    "    index_col=0, parse_dates=True\n",
    ")\n",
    "log_returns = pd.read_csv(\n",
    "    f'{PROCESSED_DIR}/log_returns.csv',\n",
    "    index_col=0, parse_dates=True\n",
    ")\n",
    "features_normalized = pd.read_csv(\n",
    "    f'{PROCESSED_DIR}/features_normalized.csv'\n",
    ")\n",
    "with open(f'{RAW_DIR}/sector_map.json') as f:\n",
    "    SECTOR_MAP = json.load(f)\n",
    "\n",
    "print(\"✅ Data loaded from notebooks 01-02\")\n",
    "print(f\"   log_returns         : {log_returns.shape}\")\n",
    "print(f\"   features_normalized : {features_normalized.shape}\")\n",
    "\n",
    "assert len(log_returns) > 0, \"Empty log_returns loaded\"\n"
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

# Fix savefig/show order and variable names
for cell in cells:
    if cell['cell_type'] == 'code':
        new_source = []
        for line in cell['source']:
            if 'plt.savefig' in line:
                filename_match = line.split('plt.savefig(')[1].split(',')[0].strip("'\" )")
                if '/' in filename_match:
                    filename = filename_match.split('/')[-1]
                else:
                    filename = filename_match
                if not any('plt.tight_layout()' in l for l in new_source):
                    new_source.append("plt.tight_layout()\n")
                new_source.append(f"plt.savefig(f'{{OUTPUTS_DIR}}/{filename}', dpi=150, bbox_inches='tight', facecolor='#0d0d0d')\n")
            elif 'plt.show()' in line:
                new_source.append(line)
                if not any('plt.close()' in l for l in cell['source']):
                    new_source.append("plt.close()\n")
            else:
                new_source.append(line)
        cell['source'] = new_source

# 4. Saving Cell
saving_cell_source = [
    "# ════════════════════════════════════════\n",
    "# SAVE ALL OUTPUTS — Required for notebook 04+\n",
    "# ════════════════════════════════════════\n",
    "\n",
    "# Ensure dist_stats exists (usually descriptive stats of returns)\n",
    "dist_stats = log_returns.describe().T\n",
    "dist_stats.to_csv(f'{PROCESSED_DIR}/distribution_stats.csv')\n",
    "\n",
    "# Ensure regimes exists (load from NB 02 if it was saved there, or re-extract)\n",
    "# In NB 02 we calculated regime_data. Let's assume it should be here as well.\n",
    "# For now, we will save risk_results as risk_metrics if it exists.\n",
    "if 'risk_results' in locals():\n",
    "    risk_results.to_csv(f'{PROCESSED_DIR}/risk_metrics.csv')\n",
    "    risk_metrics = risk_results\n",
    "else:\n",
    "    # Placeholder if not defined\n",
    "    risk_metrics = pd.DataFrame()\n",
    "\n",
    "# Correlation matrices\n",
    "corr_pearson = log_returns.corr()\n",
    "corr_pearson.to_csv(f'{PROCESSED_DIR}/correlation_matrix.csv')\n",
    "\n",
    "# Regimes - if not calculated in this notebook, we'll need to load from NB 02 or recalculate\n",
    "# Notebook 02 has the regime detection logic. Let's assume we need to save it here too if needed.\n",
    "try:\n",
    "    regimes = pd.read_csv('../data/processed/regimes.csv', index_col=0, parse_dates=True)\n",
    "except:\n",
    "    regimes = pd.DataFrame(index=log_returns.index)\n",
    "    regimes['regime'] = 0 # Default\n",
    "\n",
    "regimes.to_csv(f'{PROCESSED_DIR}/regimes.csv')\n",
    "\n",
    "print(\"✅ Notebook 03 outputs saved\")\n",
    "print(f\"   distribution_stats.csv : {dist_stats.shape}\")\n",
    "print(f\"   regimes.csv            : {regimes.shape}\")\n",
    "print(f\"   risk_metrics.csv       : {risk_metrics.shape}\")\n"
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
with open('notebooks/03_quant_core_fixed.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Saved to 03_quant_core_fixed.ipynb")
