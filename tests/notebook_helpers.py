"""Execute maintained notebook cells directly in renderer contract tests."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def notebook_namespace(name, *, cells=None):
    notebook = json.loads((ROOT / "pzbao/figures" / (name + ".ipynb")).read_text())
    namespace = {}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and (cells is None or cell["id"] in cells):
            assert cell["outputs"] == [] and cell["execution_count"] is None
            exec(compile("".join(cell["source"]), name + ":" + cell["id"], "exec"), namespace)
    return namespace
