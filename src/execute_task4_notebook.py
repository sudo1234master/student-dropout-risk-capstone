"""Execute the Task 4 notebook from the project root and save its outputs."""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "task4_model_implementation.ipynb"


def main() -> None:
    with NOTEBOOK.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)

    client = NotebookClient(
        notebook,
        timeout=1200,
        kernel_name="python3",
        # Execute from the notebook directory to match a normal interactive
        # Jupyter "Run All" session and verify the notebook's root discovery.
        resources={"metadata": {"path": str(NOTEBOOK.parent)}},
        allow_errors=False,
    )
    client.execute()

    with NOTEBOOK.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
