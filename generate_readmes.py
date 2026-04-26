import os
import json
import glob

NOTEBOOKS_DIR = "notebooks"
DOCS_DIR = "notebook_docs"

os.makedirs(DOCS_DIR, exist_ok=True)

notebook_files = glob.glob(os.path.join(NOTEBOOKS_DIR, "*.ipynb"))

for nb_file in sorted(notebook_files):
    base_name = os.path.basename(nb_file).replace('.ipynb', '')
    
    # Create a separate folder for this notebook
    nb_doc_dir = os.path.join(DOCS_DIR, base_name)
    os.makedirs(nb_doc_dir, exist_ok=True)
    
    with open(nb_file, 'r', encoding='utf-8') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"Skipping {nb_file} due to JSON error: {e}")
            continue
            
    markdown_content = []
    
    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'markdown':
            source = cell.get('source', [])
            if isinstance(source, list):
                markdown_content.append("".join(source))
            else:
                markdown_content.append(source)
                
    readme_path = os.path.join(nb_doc_dir, "README.md")
    
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(markdown_content))
        
    print(f"Generated {readme_path}")
