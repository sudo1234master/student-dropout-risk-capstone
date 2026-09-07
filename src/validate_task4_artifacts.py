"""Integrity and reproducibility checks for Task 4 deliverables."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

try:
    from src.task4_modeling import threshold_predictions
except ModuleNotFoundError:
    from task4_modeling import threshold_predictions


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "task4"
TABLE_DIR = ROOT / "reports" / "tables" / "task4"
FIGURE_DIR = ROOT / "reports" / "figures" / "task4"

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

SECOND_SEMESTER_COLUMNS = {
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
}


def main() -> None:
    required = [
        MODEL_DIR / "best_model_bundle.joblib",
        MODEL_DIR / "logistic_regression_pipeline.joblib",
        MODEL_DIR / "decision_tree_pipeline.joblib",
        MODEL_DIR / "random_forest_pipeline.joblib",
        MODEL_DIR / "histogram_gradient_boosting_pipeline.joblib",
        ROOT / "configs" / "task4_model_config.json",
        ROOT / "reports" / "task4_model_summary.json",
        ROOT / "reports" / "task4_model_implementation.html",
        ROOT / "notebooks" / "task4_model_implementation.ipynb",
        TABLE_DIR / "cross_validated_model_comparison.csv",
        TABLE_DIR / "heldout_test_metrics.csv",
        TABLE_DIR / "heldout_test_predictions.csv",
        TABLE_DIR / "oof_threshold_search.csv",
        TABLE_DIR / "top_30pct_capacity_metrics.csv",
        TABLE_DIR / "test_bootstrap_confidence_intervals.csv",
        *[FIGURE_DIR / f"0{index}_{name}.png" for index, name in [
            (1, "cross_validated_model_comparison"),
            (2, "threshold_tradeoff"),
            (3, "heldout_confusion_matrices"),
            (4, "heldout_roc_and_precision_recall"),
            (5, "heldout_capacity_and_calibration"),
        ]],
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)

    comparison = pd.read_csv(TABLE_DIR / "cross_validated_model_comparison.csv")
    assert len(comparison) == 5
    assert comparison["rank"].is_unique
    assert comparison.iloc[0]["model"] == "Random Forest"
    assert comparison.iloc[-1]["model"] == "Dummy Baseline"
    assert comparison.iloc[-1]["cv_dropout_recall"] == 0

    threshold_table = pd.read_csv(TABLE_DIR / "oof_threshold_search.csv")
    assert threshold_table["selected"].sum() == 1
    selected_threshold = float(
        threshold_table.loc[threshold_table["selected"], "threshold"].iloc[0]
    )
    with (ROOT / "configs" / "task4_model_config.json").open(
        encoding="utf-8"
    ) as handle:
        model_config = json.load(handle)
    assert np.isclose(
        selected_threshold, float(model_config["dropout_alert_threshold"])
    )

    bundle = joblib.load(MODEL_DIR / "best_model_bundle.joblib")
    assert bundle["model_name"] == "Random Forest"
    assert np.isclose(bundle["dropout_alert_threshold"], selected_threshold)
    assert len(bundle["model_input_columns"]) == 31
    assert not AUDIT_ONLY_COLUMNS.intersection(bundle["model_input_columns"])
    assert not SECOND_SEMESTER_COLUMNS.intersection(bundle["model_input_columns"])

    test = pd.read_csv(
        ROOT / "data" / "processed" / "task3" / "test_primary.csv"
    )
    probabilities = bundle["pipeline"].predict_proba(
        test[bundle["model_input_columns"]]
    )
    classes = np.asarray(bundle["classes"])
    assert probabilities.shape == (885, 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    recalculated = threshold_predictions(probabilities, classes, selected_threshold)

    saved_predictions = pd.read_csv(TABLE_DIR / "heldout_test_predictions.csv")
    assert len(saved_predictions) == 885
    assert saved_predictions["record_id"].is_unique
    assert set(saved_predictions["dropout_risk_rank"]) == set(range(1, 886))
    assert int(saved_predictions["top_30pct_risk_flag"].sum()) == 266
    aligned = (
        saved_predictions.set_index("record_id")
        .loc[test["record_id"], "thresholded_prediction"]
        .to_numpy()
    )
    assert np.array_equal(recalculated, aligned)

    actual_dropout = test["Target"].to_numpy() == "Dropout"
    predicted_dropout = recalculated == "Dropout"
    recall = recall_score(actual_dropout, predicted_dropout)
    precision = precision_score(actual_dropout, predicted_dropout)
    assert recall >= 0.75 and precision >= 0.50

    capacity = pd.read_csv(TABLE_DIR / "top_30pct_capacity_metrics.csv")
    heldout_capacity = capacity.loc[capacity["partition"] == "Held-out test"].iloc[0]
    assert heldout_capacity["dropout_capture_rate"] >= 0.60
    top_capacity_ids = set(
        saved_predictions.loc[
            saved_predictions["top_30pct_risk_flag"].astype(bool), "record_id"
        ]
    )
    expected_captured = int(
        test.loc[test["record_id"].isin(top_capacity_ids), "Target"].eq("Dropout").sum()
    )
    assert int(heldout_capacity["dropouts_captured"]) == expected_captured

    metrics = pd.read_csv(TABLE_DIR / "heldout_test_metrics.csv")
    tuned = metrics[metrics["prediction_rule"].str.startswith("Dropout threshold")].iloc[0]
    assert np.isclose(tuned["dropout_recall"], recall)
    assert np.isclose(tuned["dropout_precision"], precision)

    with (ROOT / "reports" / "task4_model_summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    assert summary["data"]["heldout_used_once_after_selection"] is True
    assert all(
        [
            summary["heldout_test"]["meets_dropout_recall_target"],
            summary["heldout_test"]["meets_dropout_precision_floor"],
            summary["heldout_test"]["meets_top_30pct_capture_target"],
        ]
    )

    notebook_path = ROOT / "notebooks" / "task4_model_implementation.ipynb"
    with notebook_path.open(encoding="utf-8") as handle:
        notebook = nbformat.read(handle, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    assert not errors

    result = {
        "status": "PASS",
        "best_model": bundle["model_name"],
        "dropout_threshold": selected_threshold,
        "heldout_rows": len(test),
        "heldout_dropout_recall": float(recall),
        "heldout_dropout_precision": float(precision),
        "heldout_top_30pct_capture": float(heldout_capacity["dropout_capture_rate"]),
        "all_task1_targets_met": True,
        "notebook_code_cells": len(code_cells),
        "notebook_errors": 0,
        "probability_rows_reproduced": len(test),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
