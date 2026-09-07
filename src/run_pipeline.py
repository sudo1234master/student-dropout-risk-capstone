"""Run or validate the complete student-dropout capstone pipeline.

Run this file from any working directory. By default it regenerates Tasks 3-5
and then validates their required artifacts. Use ``--validate-only`` to check
the committed outputs without retraining the models.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(script_name: str) -> None:
    script = ROOT / "src" / script_name
    environment = os.environ.copy()
    environment.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    print(f"\n>>> {script_name}", flush=True)
    subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing outputs without regenerating models and reports.",
    )
    parser.add_argument(
        "--execute-notebooks",
        action="store_true",
        help="After regenerating outputs, execute and save the Task 3-5 notebooks.",
    )
    arguments = parser.parse_args()

    if not arguments.validate_only:
        for script_name in [
            "task3_analysis.py",
            "task4_modeling.py",
            "task5_fairness.py",
        ]:
            run(script_name)
        if arguments.execute_notebooks:
            for script_name in [
                "execute_task3_notebook.py",
                "execute_task4_notebook.py",
                "execute_task5_notebook.py",
            ]:
                run(script_name)

    for script_name in [
        "validate_task3_artifacts.py",
        "validate_task4_artifacts.py",
        "validate_task5_artifacts.py",
    ]:
        run(script_name)

    print("\nAll requested pipeline checks passed.")


if __name__ == "__main__":
    main()

