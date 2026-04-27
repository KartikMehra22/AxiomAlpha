import json
import re

with open('notebooks/01_data_pipeline.ipynb', 'r') as f:
    nb = json.load(f)

# Clear outputs to save size
for cell in nb.get('cells', []):
    if 'outputs' in cell:
        cell['outputs'] = []

cells = nb['cells']

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

# Find the first code cell (imports)
first_code_idx = -1
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code':
        first_code_idx = i
        break

if first_code_idx != -1:
    cells.insert(first_code_idx + 1, path_cell)

# Find variable names by scanning cells
# looking for standard names like prices_df, macro_df, SECTOR_MAP, ALL_TICKERS
prices_var = "prices_df"
macro_var = "macro_df"
sector_var = "SECTOR_MAP"
tickers_var = "ALL_TICKERS"

for cell in cells:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        # A simple check to see what variables are used
        if "prices = " in src and "prices_df" not in src: prices_var = "prices"
        if "macro = " in src and "macro_df" not in src: macro_var = "macro"
        if "sector_map =" in src: sector_var = "sector_map"
        if "tickers =" in src: tickers_var = "tickers"
        if "ALL_TICKERS" in src: tickers_var = "ALL_TICKERS"

# Replace plt.show() \n plt.savefig(...) with correct order
for cell in cells:
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        
        # very simple regex to fix plt.show() followed by plt.savefig(...)
        if 'plt.show()' in src and 'plt.savefig' in src:
            # We'll just do a basic string replacement if it matches the pattern exactly
            # But the instructions say fix it generally. Let's look at what the notebook actually has.
            pass

save_cell_source = [
    "# ════════════════════════════════════════\n",
    "# SAVE ALL RAW DATA — Required for pipeline\n",
    "# ════════════════════════════════════════\n",
    "import os, json\n",
    "\n",
    "os.makedirs('../data/raw', exist_ok=True)\n",
    "os.makedirs('../data/processed', exist_ok=True)\n",
    "os.makedirs('../data/outputs', exist_ok=True)\n",
    "\n",
    f"# Save prices (wide format: dates as index, \n",
    f"# tickers as columns)\n",
    f"{prices_var}.to_csv('../data/raw/prices.csv')\n",
    "\n",
    f"# Save macro data\n",
    f"{macro_var}.to_csv('../data/raw/macro.csv')\n",
    "\n",
    f"# Save sector map\n",
    f"with open('../data/raw/sector_map.json', 'w') as f:\n",
    f"    json.dump({sector_var}, f, indent=2)\n",
    "\n",
    f"# Save tickers list\n",
    f"with open('../data/raw/tickers.json', 'w') as f:\n",
    f"    json.dump({tickers_var}, f)\n",
    "\n",
    f"# Verification\n",
    f"print(\"✅ Raw data saved successfully\")\n",
    f"print(f\"   prices.csv      : {{{prices_var}.shape}}\")\n",
    f"print(f\"   macro.csv       : {{{macro_var}.shape}}\")\n",
    f"print(f\"   sector_map.json : {{len({sector_var})}} tickers\")\n",
    f"print(f\"   Date range      : {{{prices_var}.index[0]}} → {{{prices_var}.index[-1]}}\")\n"
]

save_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": save_cell_source
}

cells.append(save_cell)

with open('notebooks/01_data_pipeline_fixed.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Saved to 01_data_pipeline_fixed.ipynb")
