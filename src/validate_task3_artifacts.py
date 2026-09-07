"""Integrity checks for the Task 3 capstone deliverables."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import nbformat
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed" / "task3"
MODELS = ROOT / "models" / "task3"
REPORTS = ROOT / "reports"

SECOND_SEMESTER_COLUMNS = {
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
}

AUDIT_ONLY_COLUMNS = {
    "Marital Status",
    "Nationality",
    "Displaced",
    "Educational special needs",
    "Gender",
    "Age at enrollment",
    "International",
    "nontraditional_age_flag",
}


def main() -> None:
    required_files = [
        PROCESSED / "primary_first_semester_dataset.csv",
        PROCESSED / "train_primary.csv",
        PROCESSED / "test_primary.csv",
        PROCESSED / "train_selected_features.csv",
        PROCESSED / "test_selected_features.csv",
        MODELS / "preprocessor_first_semester.joblib",
        MODELS / "selected_features.json",
        MODELS / "pca_full.joblib",
        REPORTS / "task3_analysis_summary.json",
        REPORTS / "task3_eda_feature_engineering.html",
        ROOT / "notebooks" / "task3_eda_feature_engineering.ipynb",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required_files)

    primary = pd.read_csv(PROCESSED / "primary_first_semester_dataset.csv")
    train = pd.read_csv(PROCESSED / "train_primary.csv")
    test = pd.read_csv(PROCESSED / "test_primary.csv")
    train_selected = pd.read_csv(PROCESSED / "train_selected_features.csv")
    test_selected = pd.read_csv(PROCESSED / "test_selected_features.csv")

    assert (len(primary), len(train), len(test)) == (4424, 3539, 885)
    assert train["record_id"].is_unique and test["record_id"].is_unique
    assert set(train["record_id"]).isdisjoint(set(test["record_id"]))
    assert set(train["record_id"]) | set(test["record_id"]) == set(primary["record_id"])
    assert not SECOND_SEMESTER_COLUMNS.intersection(train.columns)
    assert not SECOND_SEMESTER_COLUMNS.intersection(test.columns)
    assert AUDIT_ONLY_COLUMNS.issubset(train.columns) and AUDIT_ONLY_COLUMNS.issubset(test.columns)

    full_share = primary["Target"].value_counts(normalize=True)
    train_share = train["Target"].value_counts(normalize=True)
    test_share = test["Target"].value_counts(normalize=True)
    assert (train_share - full_share).abs().max() < 0.002
    assert (test_share - full_share).abs().max() < 0.002

    with (MODELS / "selected_features.json").open(encoding="utf-8") as handle:
        selection_record = json.load(handle)
    selected_features = selection_record["selected_features"]
    assert selection_record["top_k"] == 30
    assert len(selected_features) == 30 and len(set(selected_features)) == 30
    expected_selected_columns = ["record_id", *selected_features, "Target"]
    assert train_selected.columns.tolist() == expected_selected_columns
    assert test_selected.columns.tolist() == expected_selected_columns

    preprocessor = joblib.load(MODELS / "preprocessor_first_semester.joblib")
    fitted_inputs = set(preprocessor.feature_names_in_)
    assert not AUDIT_ONLY_COLUMNS.intersection(fitted_inputs)
    assert not SECOND_SEMESTER_COLUMNS.intersection(fitted_inputs)
    assert len(preprocessor.get_feature_names_out()) == 125

    notebook_path = ROOT / "notebooks" / "task3_eda_feature_engineering.ipynb"
    with notebook_path.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert not errors

    with (REPORTS / "task3_analysis_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["data_scope"]["train_rows"] == len(train)
    assert summary["data_scope"]["test_rows"] == len(test)
    assert summary["data_scope"]["selected_features"] == len(selected_features)

    result = {
        "status": "PASS",
        "rows": len(primary),
        "train_rows": len(train),
        "test_rows": len(test),
        "id_overlap": 0,
        "second_semester_columns_in_primary_split": 0,
        "audit_columns_in_default_preprocessor": 0,
        "encoded_features": len(preprocessor.get_feature_names_out()),
        "selected_features": len(selected_features),
        "executed_notebook_code_cells": len(code_cells),
        "notebook_cell_errors": 0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
