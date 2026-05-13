import json
with open('Final_model.ipynb', encoding='utf-8') as f:
    cells = json.load(f)['cells']
code_cells = ["".join(c['source']) for c in cells if c['cell_type'] == 'code']
with open('extracted_final.py', 'w', encoding='utf-8') as f:
    f.write("\n\n".join(code_cells))
