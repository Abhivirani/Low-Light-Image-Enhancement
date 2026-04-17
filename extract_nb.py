import json
import sys

try:
    with open("notebooke3d7a332db (1).ipynb", "r", encoding="utf-8") as f:
        nb = json.load(f)
    with open("model_notebook.py", "w", encoding="utf-8") as out:
        out.write("import torch\nimport torch.nn as nn\nfrom torch.ao.quantization import QuantStub, DeQuantStub\n\n")
        
        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code":
                source = "".join(cell.get("source", []))
                if "class" in source and "nn.Module" in source:
                    out.write(source)
                    out.write("\n\n")

    print("Successfully wrote python models to model_notebook.py")
except Exception as e:
    print(f"Error: {e}")
