"""Build the reproducible Task 5 ethical AI and fairness notebook."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "task5_bias_fairness_analysis.ipynb"


def code(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(dedent(source).strip())


def markdown(source: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(dedent(source).strip())


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3 (Capstone)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
        "title": "Task 5: Bias and Fairness Analysis",
    }

    notebook["cells"] = [
        markdown(
            """
            # Task 5 — Critical Thinking, Ethical AI, and Bias Auditing

            **Model:** Locked Task 4 Random Forest  
            **Audit partition:** 885 held-out test records  
            **Adverse selection event:** `Dropout` alert at the saved Task 4 threshold reported below

            This notebook addresses the Task 5 rubric through SHAP explanations, PDP/ICE diagnostics, imbalance/leakage/overfitting review, subgroup fairness metrics, intersectional screening, limitations, and mitigation recommendations.

            A dropout alert is treated as adverse for measurement because it can create stigma, extra scrutiny, or punitive misuse. It may also provide a benefit when it triggers respectful support. Fairness therefore cannot be decided by one parity statistic alone.
            """
        ),
        markdown(
            """
            ## 1. Audit boundaries

            Gender, age, and international status were excluded from model inputs and retained only for auditing. Financial-risk and scholarship variables remain model inputs or related features and are audited as incomplete socioeconomic proxies.

            Race is not available. Nationality and international status are not equivalent to race, so no racial fairness claim is made. The source gender field is binary and may not represent gender identity or nonbinary students.
            """
        ),
        code(
            """
            from pathlib import Path
            import json
            import sys

            import pandas as pd
            from IPython.display import Image, display

            def find_project_root(start: Path) -> Path:
                # Locate the extracted project root from Jupyter's working directory.
                start = start.resolve()
                for candidate in (start, *start.parents):
                    if (candidate / "src" / "task5_fairness.py").is_file():
                        return candidate
                raise FileNotFoundError(
                    "Project root not found. Keep the notebook inside the extracted Task 5 folder."
                )

            ROOT = find_project_root(Path.cwd())
            if str(ROOT) not in sys.path:
                sys.path.insert(0, str(ROOT))

            from src.task5_fairness import run_task5

            artifacts = run_task5()
            with (ROOT / "reports" / "task5_bias_fairness_summary.json").open(encoding="utf-8") as handle:
                summary = json.load(handle)

            print(f"Model: {summary['model']['name']}")
            print(f"Dropout alert threshold: {summary['model']['dropout_alert_threshold']:.2f}")
            print(f"Held-out records audited: {summary['model']['heldout_records_audited']:,}")
            print(summary['fairness_results']['race_audit_status'])
            """
        ),
        markdown(
            """
            ## 2. Global SHAP explanations

            Tree SHAP explains the locked Random Forest for the `Dropout` class. Mean absolute SHAP values summarize average contribution magnitude. Positive values increase the model's dropout probability relative to its expected value, while negative values decrease it.

            The grouped chart combines one-hot components belonging to the same source variable. Because summing components can favor high-cardinality fields such as course and application mode, both grouped and encoded rankings are saved. SHAP explains model behavior; it does not establish that a feature causes dropout or that changing it would improve outcomes.
            """
        ),
        code(
            """
            grouped_shap = pd.read_csv(ROOT / "reports" / "tables" / "task5" / "shap_grouped_feature_importance.csv")
            encoded_shap = pd.read_csv(ROOT / "reports" / "tables" / "task5" / "shap_encoded_feature_importance.csv")
            display(grouped_shap.head(15).round(4))
            display(Image(filename=str(ROOT / "reports" / "figures" / "task5" / "01_shap_global_importance.png")))
            display(Image(filename=str(ROOT / "reports" / "figures" / "task5" / "02_shap_direction_summary.png")))
            """
        ),
        markdown(
            """
            ## 3. Individual explanations

            Three cases illustrate how the same model can correctly flag a dropout, incorrectly flag a non-dropout, or miss a dropout. These are diagnostic examples, not student-facing explanations. A production explanation should translate codes into plain language, disclose uncertainty, avoid blame, and support correction or appeal.
            """
        ),
        code(
            """
            local = pd.read_csv(ROOT / "reports" / "tables" / "task5" / "shap_local_case_explanations.csv")
            display(local[[
                "case_type", "record_id", "actual_outcome", "thresholded_prediction",
                "dropout_probability", "rank_within_case", "encoded_feature",
                "dropout_shap_value", "effect_direction"
            ]].head(15).round(4))
            display(Image(filename=str(ROOT / "reports" / "figures" / "task5" / "03_shap_local_explanations.png")))
            """
        ),
        markdown(
            """
            ## 4. PDP and ICE diagnostics

            Partial dependence shows the model's average predicted response to first-semester progress and approval rate. ICE lines show heterogeneity across 60 held-out students. Both features are engineered from related academic variables, so changing one while holding its components fixed can create unrealistic combinations. The curves are model diagnostics rather than causal intervention estimates.
            """
        ),
        code(
            """
            pdp_ice = pd.read_csv(ROOT / "reports" / "tables" / "task5" / "pdp_ice_curves.csv")
            pdp_average = pdp_ice[pdp_ice["curve_type"] == "PDP average"]
            display(pdp_average.groupby("feature").agg(
                minimum_grid_value=("feature_value", "min"),
                maximum_grid_value=("feature_value", "max"),
                minimum_average_risk=("predicted_dropout_probability", "min"),
                maximum_average_risk=("predicted_dropout_probability", "max"),
            ).round(3))
            display(Image(filename=str(ROOT / "reports" / "figures" / "task5" / "04_pdp_ice_academic_progress.png")))
            """
        ),
        markdown(
            """
            ## 5. Fairness definitions

            The audit reports:

            - **Demographic parity difference:** difference in adverse alert rates.
            - **Disparate-impact ratio:** group adverse alert rate divided by the reference group's rate. A symmetric parity ratio is also reported for four-fifths screening.
            - **Equal opportunity difference:** difference in dropout true-positive rate or recall.
            - **Equalized odds:** differences in both true-positive and false-positive rates.
            - **Calibration and precision:** whether risk scores and alerts have comparable meaning across groups.

            Reference groups are documented comparison anchors chosen for stable interpretation, not assertions that those groups are inherently privileged. A four-fifths parity ratio below 0.80 or a maximum TPR/FPR gap above 0.10 triggers review. These are screening rules, not legal findings.
            """
        ),
        code(
            """
            group_metrics = pd.read_csv(ROOT / "reports" / "tables" / "task5" / "fairness_group_metrics.csv")
            core_metrics = group_metrics[group_metrics["audit_dimension"] != "Gender x financial risk"]
            display(core_metrics[[
                "audit_dimension", "group", "reference_group", "n", "actual_dropout_n",
                "observed_dropout_rate", "alert_rate", "dropout_precision",
                "dropout_recall_tpr", "false_positive_rate", "brier_score",
                "small_sample_warning"
            ]].round(3))
            display(Image(filename=str(ROOT / "reports" / "figures" / "task5" / "05_group_dropout_and_alert_rates.png")))
            display(Image(filename=str(ROOT / "reports" / "figures" / "task5" / "06_group_true_and_false_positive_rates.png")))
            """
        ),
        markdown(
            """
            ## 6. Fairness findings

            Material screening differences are present. The table and chart below are regenerated from the locked Task 4 model, so they remain aligned with the saved alert threshold.

            Particular attention is required for age, immediate financial-risk indicators, scholarship status, and the intersection of gender and financial risk. Small international and scholarship-positive samples make some estimates unstable. Alert-rate and equalized-odds flags are screening signals rather than legal findings.

            Outcome-rate differences partly explain alert-rate differences, but historical outcomes may themselves reflect unequal access and institutional conditions. A disparity that mirrors historical labels is not automatically fair.
            """
        ),
        code(
            """
            gap_metrics = pd.read_csv(ROOT / "reports" / "tables" / "task5" / "fairness_gap_metrics.csv")
            display(gap_metrics.round(3))
            display(Image(filename=str(ROOT / "reports" / "figures" / "task5" / "07_fairness_gap_summary.png")))

            print("Small-sample groups")
            display(group_metrics.loc[group_metrics["small_sample_warning"], [
                "audit_dimension", "group", "n", "actual_dropout_n",
                "dropout_recall_tpr", "false_positive_rate"
            ]].round(3))
            """
        ),
        markdown(
            """
            ## 7. Imbalance, leakage, and overfitting

            - **Imbalance:** the `Enrolled` class is about 18% and `Dropout` about 32%. Task 4 used stratification, class weighting, macro metrics, dropout-specific metrics, and fixed-capacity capture rather than accuracy alone.
            - **Leakage:** all six second-semester variables were removed. Learned preprocessing, feature selection, model selection, and threshold selection used training data only. Task 5 audits the already locked model on held-out data.
            - **Overfitting:** the generalization table below compares cross-validated or out-of-fold estimates with the locked held-out results. A modest gap on one split cannot replace external and temporal validation.
            """
        ),
        code(
            """
            generalization = pd.read_csv(ROOT / "reports" / "tables" / "task5" / "generalization_gap_review.csv")
            display(generalization.round(3))
            """
        ),
        markdown(
            """
            ## 8. Ethical and data limitations

            - One Portuguese institution limits external validity.
            - Race is unavailable and cannot be audited.
            - Gender is binary and may not represent student identity.
            - International and educational-special-needs samples are small.
            - Financial variables do not measure income, wealth, caregiving, accessibility costs, or access to support.
            - Historical dropout labels can encode institutional and structural inequities.
            - SHAP, PDP, ICE, subgroup relationships, and fairness metrics are descriptive rather than causal.
            - Increasing dropout recall also increases false-positive outreach and reduces overall multiclass performance.
            """
        ),
        markdown(
            """
            ## 9. Mitigation plan

            Mitigations must be developed with training or new validation data rather than tuned on this audited test set. Group-specific thresholds should be considered only after legal and ethical review because improving one fairness criterion can worsen another.
            """
        ),
        code(
            """
            mitigations = pd.read_csv(ROOT / "reports" / "tables" / "task5" / "mitigation_plan.csv")
            display(mitigations)
            """
        ),
        markdown(
            """
            ## 10. Recommendation

            **Do not use this model for automated adverse decisions.** It may be suitable for a controlled, non-punitive student-support pilot only after:

            1. review of the flagged financial, age, scholarship, and intersectional disparities;
            2. training-only reweighting or resampling experiments;
            3. probability calibration and validation on a new cohort;
            4. human review, plain-language explanations, correction and appeal;
            5. explicit prohibition on use for exclusion, discipline, grading, or financial penalties; and
            6. ongoing performance, fairness, calibration, and drift monitoring.

            A separate written `Bias & Fairness Analysis` section, all detailed metrics, explanations, figures, and the audit configuration are saved with this notebook.
            """
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_notebook()
