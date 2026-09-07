"""Build a ten-slide Jupyter presentation for the technical capstone audience."""

from __future__ import annotations

import base64
from pathlib import Path
from textwrap import dedent

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "task6_technical_presentation.ipynb"


def slide(source: str, image_paths: list[Path] | None = None) -> nbf.NotebookNode:
    cell = nbf.v4.new_markdown_cell(dedent(source).strip())
    cell.metadata["slideshow"] = {"slide_type": "slide"}
    if image_paths:
        cell["attachments"] = {}
        for image_path in image_paths:
            cell["attachments"][image_path.name] = {
                "image/png": base64.b64encode(image_path.read_bytes()).decode("ascii")
            }
    return cell


def build_notebook() -> None:
    figures = ROOT / "reports" / "figures"
    notebook = nbf.v4.new_notebook()
    notebook.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (Capstone)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
        "celltoolbar": "Slideshow",
        "rise": {"scroll": True, "theme": "white", "transition": "fade"},
        "title": "Predicting Student Dropout Risk — Technical Presentation",
    }

    notebook.cells = [
        slide(
            """
            <style>
            .reveal { color: #243447; font-family: Arial, sans-serif; }
            .reveal .slides section { text-align: left; }
            .reveal h1 { color: #102A43; font-size: 1.75em; margin-bottom: 0.25em; }
            .reveal h2 { color: #102A43; font-size: 1.25em; margin-bottom: 0.35em; }
            .reveal h3 { color: #0F766E; font-size: 0.9em; }
            .reveal p, .reveal li { font-size: 0.66em; line-height: 1.32; }
            .reveal strong { color: #0F766E; }
            .reveal img { border: 0; box-shadow: none; max-height: 480px; display: block; margin: 10px auto; }
            .reveal .metric { color: #0F766E; font-size: 1.35em; font-weight: 700; }
            .reveal .subtle { color: #60758A; font-size: 0.52em; }
            .reveal table { font-size: 0.52em; }
            </style>

            # Predicting student dropout risk

            ### End-to-end machine learning capstone

            Early warning after the first semester  
            Multiclass outcome: Dropout, Enrolled, Graduate

            <p class="subtle">Technical presentation for peer review</p>

            <aside class="notes">Project domain and task requirements come from Capstone - Instructions.pdf.</aside>
            """
        ),
        slide(
            """
            ## Problem framing and success criteria

            **Business problem**  
            Student-support teams need an evidence-based way to prioritize outreach after first-semester results become available.

            **Data science task**  
            Supervised multiclass classification with `P(Dropout)` used as the operational risk score.

            **Predefined success criteria**

            - Dropout recall at least **75%**
            - Dropout precision at least **50%**
            - Capture at least **60%** of dropouts in the highest-risk 30% of students

            Accuracy remains secondary because the class distribution is uneven.

            <aside class="notes">Source: reports/task4_model_summary.json and Task 1 project framing.</aside>
            """
        ),
        slide(
            """
            ## Dataset and leakage controls

            <span class="metric">4,424</span> students from one Portuguese higher-education institution

            - 36 original predictors and one three-class target
            - 49.9% Graduate, 32.1% Dropout, 17.9% Enrolled
            - No machine-readable nulls or exact duplicate rows
            - Six second-semester fields removed from the first-semester prediction scope
            - Sensitive demographic fields retained for audit but excluded from model inputs

            **Development split:** 3,539 training records and 885 held-out test records  
            All learned preprocessing, model selection, and threshold tuning used training data only.

            <aside class="notes">Dataset: UCI Predict Students' Dropout and Academic Success, DOI 10.24432/C5MC89. Source: reports/task3_analysis_summary.json.</aside>
            """
        ),
        slide(
            """
            ## EDA and feature engineering

            ![Target distribution](attachment:01_target_distribution.png)

            Nine prediction-time features captured academic progress, evaluation activity, preparation gaps, program preference, and immediate financial risk.

            First-semester approval rate and the combined progress index became the strongest screening features.

            <aside class="notes">Source figure: reports/figures/task3/01_target_distribution.png. Full definitions: notebooks/task3_eda_feature_engineering.ipynb.</aside>
            """,
            [figures / "task3" / "01_target_distribution.png"],
        ),
        slide(
            """
            ## Cross-validated model comparison

            ![Cross-validated model comparison](attachment:01_cross_validated_model_comparison.png)

            Random Forest produced the highest five-fold dropout F2. Logistic Regression had slightly higher macro F1, but model selection followed the predefined dropout-focused objective.

            <aside class="notes">Source figure: reports/figures/task4/01_cross_validated_model_comparison.png.</aside>
            """,
            [figures / "task4" / "01_cross_validated_model_comparison.png"],
        ),
        slide(
            """
            ## Threshold selection

            ![Threshold tradeoff](attachment:02_threshold_tradeoff.png)

            Out-of-fold training predictions selected a **0.23 dropout threshold**. This threshold maximized dropout F2 among settings that met the recall and precision constraints.

            The threshold changes the alert decision, not the underlying three-class probabilities.

            <aside class="notes">Source figure: reports/figures/task4/02_threshold_tradeoff.png.</aside>
            """,
            [figures / "task4" / "02_threshold_tradeoff.png"],
        ),
        slide(
            """
            ## Held-out performance and support capacity

            ![Capacity and calibration](attachment:05_heldout_capacity_and_calibration.png)

            - Dropout recall: **83.8%**
            - Dropout precision: **62.6%**
            - Dropout F2: **78.5%**
            - Top 30% captured **207 of 284 dropouts**, or **72.9%**
            - Precision within the top-risk list: **77.8%**, a **2.43× lift**

            <aside class="notes">Source figure: reports/figures/task4/05_heldout_capacity_and_calibration.png. Metrics: reports/tables/task4/heldout_test_metrics.csv.</aside>
            """,
            [figures / "task4" / "05_heldout_capacity_and_calibration.png"],
        ),
        slide(
            """
            ## Model explanations

            ![SHAP global importance](attachment:01_shap_global_importance.png)

            Tree SHAP identified progress index, approval rate, tuition status, approved units, and financial-risk indicators as the largest global contributors to modeled dropout risk.

            SHAP, PDP, and ICE explain model behavior. They do not establish causal effects or prescribe interventions.

            <aside class="notes">Source figure: reports/figures/task5/01_shap_global_importance.png. SHAP computed for all 885 held-out records.</aside>
            """,
            [figures / "task5" / "01_shap_global_importance.png"],
        ),
        slide(
            """
            ## Fairness audit

            ![Fairness screening](attachment:07_fairness_gap_summary.png)

            Age, financial-risk, scholarship, and intersectional comparisons triggered equalized-odds review flags. Gender triggered alert-rate parity review but stayed below the 10% equalized-odds screening gap.

            Race cannot be audited because the dataset does not contain it. The international subgroup contains only 24 test records.

            <aside class="notes">Source figure: reports/figures/task5/07_fairness_gap_summary.png. Flags are screening signals, not legal findings.</aside>
            """,
            [figures / "task5" / "07_fairness_gap_summary.png"],
        ),
        slide(
            """
            ## Conclusion and reproducibility

            The Random Forest met all three predefined held-out success criteria, but fairness gaps and limited external validity prevent an unrestricted deployment recommendation.

            **Recommended use:** controlled, non-punitive support pilot with human review, correction and appeal, probability calibration, and subgroup monitoring.

            **Reproducibility assets**

            - Complete preprocessing and model bundle
            - Fixed random seed and parameter grids
            - Executed Task 3–5 notebooks
            - Held-out predictions, metrics, SHAP values, and fairness tables

            A new institutional cohort should validate any mitigation or threshold change.

            <aside class="notes">Code and artifacts are stored in src/, notebooks/, models/, configs/, and reports/.</aside>
            """
        ),
    ]

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    build_notebook()
