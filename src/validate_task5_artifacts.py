"""Integrity and calculation checks for Task 5 deliverables."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import nbformat
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLE_DIR = ROOT / "reports" / "tables" / "task5"
FIGURE_DIR = ROOT / "reports" / "figures" / "task5"

SENSITIVE_AUDIT_ONLY = {
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
    required = [
        ROOT / "models" / "task5" / "heldout_dropout_shap_values.npy",
        ROOT / "configs" / "task5_audit_config.json",
        ROOT / "reports" / "task5_bias_fairness_summary.json",
        ROOT / "reports" / "task5_bias_fairness_analysis.md",
        ROOT / "reports" / "task5_bias_fairness_analysis.html",
        ROOT / "notebooks" / "task5_bias_fairness_analysis.ipynb",
        TABLE_DIR / "shap_grouped_feature_importance.csv",
        TABLE_DIR / "shap_encoded_feature_importance.csv",
        TABLE_DIR / "shap_local_case_explanations.csv",
        TABLE_DIR / "pdp_ice_curves.csv",
        TABLE_DIR / "heldout_audit_predictions.csv",
        TABLE_DIR / "fairness_group_metrics.csv",
        TABLE_DIR / "fairness_gap_metrics.csv",
        TABLE_DIR / "generalization_gap_review.csv",
        TABLE_DIR / "mitigation_plan.csv",
        *[
            FIGURE_DIR / f"0{index}_{name}.png"
            for index, name in [
                (1, "shap_global_importance"),
                (2, "shap_direction_summary"),
                (3, "shap_local_explanations"),
                (4, "pdp_ice_academic_progress"),
                (5, "group_dropout_and_alert_rates"),
                (6, "group_true_and_false_positive_rates"),
                (7, "fairness_gap_summary"),
            ]
        ],
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)

    test = pd.read_csv(
        ROOT / "data" / "processed" / "task3" / "test_primary.csv"
    )
    bundle = joblib.load(ROOT / "models" / "task4" / "best_model_bundle.joblib")
    assert not SENSITIVE_AUDIT_ONLY.intersection(bundle["model_input_columns"])
    probabilities = bundle["pipeline"].predict_proba(
        test[bundle["model_input_columns"]]
    )
    classes = np.asarray(bundle["classes"])
    dropout_index = int(np.where(classes == "Dropout")[0][0])

    dropout_shap = np.load(
        ROOT / "models" / "task5" / "heldout_dropout_shap_values.npy"
    )
    assert dropout_shap.shape == (885, 125)
    assert np.isfinite(dropout_shap).all()
    with (ROOT / "reports" / "task5_bias_fairness_summary.json").open(
        encoding="utf-8"
    ) as handle:
        summary = json.load(handle)
    expected = float(summary["explainability"]["expected_dropout_value"])
    reconstructed = expected + dropout_shap.sum(axis=1)
    shap_max_error = float(
        np.max(np.abs(reconstructed - probabilities[:, dropout_index]))
    )
    assert shap_max_error < 1e-8

    grouped_shap = pd.read_csv(TABLE_DIR / "shap_grouped_feature_importance.csv")
    expected_top_feature = summary["explainability"]["top_10_grouped_features"][0][
        "original_feature"
    ]
    assert grouped_shap.iloc[0]["original_feature"] == expected_top_feature
    assert grouped_shap["rank"].tolist() == list(range(1, len(grouped_shap) + 1))
    local = pd.read_csv(TABLE_DIR / "shap_local_case_explanations.csv")
    assert local["case_type"].nunique() == 3
    assert len(local) == 36

    curves = pd.read_csv(TABLE_DIR / "pdp_ice_curves.csv")
    assert set(curves["feature"]) == {
        "first_sem_progress_index",
        "first_sem_approval_rate",
    }
    assert len(curves) == 2 * 25 * 61
    assert curves["predicted_dropout_probability"].between(0, 1).all()

    audit = pd.read_csv(TABLE_DIR / "heldout_audit_predictions.csv")
    assert len(audit) == 885 and audit["record_id"].is_unique
    assert np.allclose(
        audit["dropout_probability"], probabilities[:, dropout_index]
    )

    metrics = pd.read_csv(TABLE_DIR / "fairness_group_metrics.csv")
    dimensions = metrics["audit_dimension"].unique()
    for dimension in dimensions:
        assert int(metrics.loc[metrics["audit_dimension"] == dimension, "n"].sum()) == 885
    assert metrics["alert_rate"].between(0, 1).all()
    assert metrics["dropout_recall_tpr"].dropna().between(0, 1).all()
    international = metrics[
        (metrics["audit_dimension"] == "International status")
        & (metrics["group"] == "International")
    ].iloc[0]
    assert int(international["n"]) == 24 and bool(international["small_sample_warning"])

    gaps = pd.read_csv(TABLE_DIR / "fairness_gap_metrics.csv")
    for _, row in gaps.iterrows():
        subset = metrics[metrics["audit_dimension"] == row["audit_dimension"]]
        group = subset[subset["group"] == row["group"]].iloc[0]
        reference = subset[subset["group"] == row["reference_group"]].iloc[0]
        assert np.isclose(
            row["demographic_parity_difference"],
            group["alert_rate"] - reference["alert_rate"],
        )
        assert np.isclose(
            row["equal_opportunity_tpr_difference"],
            group["dropout_recall_tpr"] - reference["dropout_recall_tpr"],
        )
        expected_equalized_gap = max(
            abs(group["dropout_recall_tpr"] - reference["dropout_recall_tpr"]),
            abs(group["false_positive_rate"] - reference["false_positive_rate"]),
        )
        assert np.isclose(row["equalized_odds_max_gap"], expected_equalized_gap)

    assert int(gaps["four_fifths_review_flag"].sum()) == summary["fairness_results"]["four_fifths_review_flags"]
    assert int(gaps["equalized_odds_review_flag"].sum()) == summary["fairness_results"]["equalized_odds_review_flags"]
    assert "Not possible" in summary["fairness_results"]["race_audit_status"]

    generalization = pd.read_csv(TABLE_DIR / "generalization_gap_review.csv")
    assert generalization["heldout_minus_training"].between(-0.03, 0.01).all()
    mitigations = pd.read_csv(TABLE_DIR / "mitigation_plan.csv")
    assert len(mitigations) == 6 and mitigations["priority"].tolist() == list(range(1, 7))

    report_text = (ROOT / "reports" / "task5_bias_fairness_analysis.md").read_text(
        encoding="utf-8"
    )
    for heading in [
        "## Model explanations",
        "## Fairness metrics",
        "## Imbalance, leakage, and overfitting",
        "## Data and ethical limitations",
        "## Mitigation plan",
        "## Decision",
    ]:
        assert heading in report_text
    assert "Nationality or international status is not treated as race" in report_text

    notebook_path = ROOT / "notebooks" / "task5_bias_fairness_analysis.ipynb"
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
        "heldout_records_audited": len(test),
        "direct_sensitive_model_inputs": 0,
        "shap_records": dropout_shap.shape[0],
        "shap_features": dropout_shap.shape[1],
        "shap_max_additivity_error": shap_max_error,
        "fairness_dimensions": len(dimensions),
        "four_fifths_review_flags": int(gaps["four_fifths_review_flag"].sum()),
        "equalized_odds_review_flags": int(gaps["equalized_odds_review_flag"].sum()),
        "notebook_code_cells": len(code_cells),
        "notebook_errors": 0,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
