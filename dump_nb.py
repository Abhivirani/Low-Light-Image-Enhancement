import json

with open("notebooke3d7a332db (1).ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("nb_source.py", "w", encoding="utf-8") as out:
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") == "code":
            out.write(f"\n# --- CELL {i} ---\n")
            out.write("".join(cell.get("source", [])))
            out.write("\n")
