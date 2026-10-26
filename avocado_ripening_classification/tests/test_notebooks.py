import ast
from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_notebook_json_and_python_cells_are_valid():
    notebook_paths = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    assert len(notebook_paths) == 4

    for notebook_path in notebook_paths:
        notebook = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(notebook)
        assert notebook["nbformat"] == 4
        for cell_number, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            ast.parse(source, filename=f"{notebook_path.name}:cell-{cell_number}")
