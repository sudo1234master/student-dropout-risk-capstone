"""Task 5: explainability, ethical AI review, and held-out fairness audit."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import sklearn
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import accuracy_score, brier_score_loss, precision_score, recall_score

try:
    from src.task3_analysis import define_model_columns
    from src.task4_modeling import threshold_predictions
except ModuleNotFoundError:
    from task3_analysis import define_model_columns
    from task4_modeling import threshold_predictions


ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = ROOT / "data" / "processed" / "task3" / "test_primary.csv"
BUNDLE_PATH = ROOT / "models" / "task4" / "best_model_bundle.joblib"
TASK4_SUMMARY_PATH = ROOT / "reports" / "task4_model_summary.json"
MODEL_DIR = ROOT / "models" / "task5"
TABLE_DIR = ROOT / "reports" / "tables" / "task5"
FIGURE_DIR = ROOT / "reports" / "figures" / "task5"
CONFIG_DIR = ROOT / "configs"
SUMMARY_PATH = ROOT / "reports" / "task5_bias_fairness_summary.json"
REPORT_PATH = ROOT / "reports" / "task5_bias_fairness_analysis.md"

RANDOM_STATE = 42
MIN_GROUP_SIZE = 50
MIN_POSITIVE_CASES = 25
PARITY_RATIO_FLOOR = 0.80
EQUALIZED_ODDS_GAP_REVIEW = 0.10


@dataclass
class Task5Artifacts:
    summary: dict[str, Any]
    group_metrics: pd.DataFrame
    fairness_gaps: pd.DataFrame
    shap_importance: pd.DataFrame
    local_explanations: pd.DataFrame


def ensure_directories() -> None:
    for directory in [MODEL_DIR, TABLE_DIR, FIGURE_DIR, CONFIG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 12,
            "figure.titlesize": 15,
            "legend.frameon": False,
        }
    )


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return np.nan, np.nan
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return float(max(0, centre - margin)), float(min(1, centre + margin))


def transformed_feature_mapping(pipeline: Any, test: pd.DataFrame) -> tuple[np.ndarray, list[str], list[str]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    selector = pipeline.named_steps.get("selector")
    transformed_names = np.asarray(preprocessor.get_feature_names_out())
    if selector is not None:
        support = selector.get_support()
        transformed_names = transformed_names[support]

    _, categorical_columns, _ = define_model_columns(test)
    sorted_categorical = sorted(categorical_columns, key=len, reverse=True)
    original_names: list[str] = []
    for encoded_name in transformed_names:
        if encoded_name.startswith("numeric__"):
            original_names.append(encoded_name.removeprefix("numeric__"))
            continue
        remainder = encoded_name.removeprefix("categorical__")
        match = next(
            (
                column
                for column in sorted_categorical
                if remainder == column or remainder.startswith(f"{column}_")
            ),
            remainder,
        )
        original_names.append(match)
    return transformed_names, original_names, categorical_columns


def compute_shap_explanations(
    bundle: dict[str, Any], test: pd.DataFrame, probabilities: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    pipeline = bundle["pipeline"]
    model_columns = bundle["model_input_columns"]
    transformed = np.asarray(pipeline[:-1].transform(test[model_columns]))
    transformed_names, original_names, _ = transformed_feature_mapping(pipeline, test)
    model = pipeline.named_steps["model"]
    classes = np.asarray(bundle["classes"])
    dropout_index = int(np.where(classes == "Dropout")[0][0])

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(
        explainer.shap_values(transformed, check_additivity=False)
    )
    if shap_values.ndim != 3:
        raise AssertionError(f"Expected multiclass SHAP array, received {shap_values.shape}")
    dropout_shap = shap_values[:, :, dropout_index]
    np.save(MODEL_DIR / "heldout_dropout_shap_values.npy", dropout_shap)

    encoded_importance = pd.DataFrame(
        {
            "encoded_feature": transformed_names,
            "original_feature": original_names,
            "mean_absolute_shap": np.mean(np.abs(dropout_shap), axis=0),
            "mean_signed_shap": np.mean(dropout_shap, axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    encoded_importance["rank"] = np.arange(1, len(encoded_importance) + 1)
    encoded_importance.to_csv(TABLE_DIR / "shap_encoded_feature_importance.csv", index=False)

    grouped_importance = (
        encoded_importance.groupby("original_feature", as_index=False)
        .agg(
            mean_absolute_shap=("mean_absolute_shap", "sum"),
            encoded_components=("encoded_feature", "count"),
        )
        .sort_values("mean_absolute_shap", ascending=False)
        .reset_index(drop=True)
    )
    grouped_importance["rank"] = np.arange(1, len(grouped_importance) + 1)
    grouped_importance.to_csv(TABLE_DIR / "shap_grouped_feature_importance.csv", index=False)

    threshold = float(bundle["dropout_alert_threshold"])
    predictions = threshold_predictions(probabilities, classes, threshold)
    dropout_probability = probabilities[:, dropout_index]
    actual_dropout = test["Target"].astype(str).to_numpy() == "Dropout"
    alerted = predictions == "Dropout"

    candidate_definitions = {
        "True positive - high risk": np.where(actual_dropout & alerted)[0],
        "False positive - high risk": np.where((~actual_dropout) & alerted)[0],
        "False negative - missed dropout": np.where(actual_dropout & (~alerted))[0],
    }
    chosen: dict[str, int] = {}
    for case_type, candidates in candidate_definitions.items():
        if len(candidates) == 0:
            continue
        if case_type.startswith("False negative"):
            chosen[case_type] = int(candidates[np.argmax(dropout_probability[candidates])])
        else:
            chosen[case_type] = int(candidates[np.argmax(dropout_probability[candidates])])

    local_rows: list[dict[str, Any]] = []
    for case_type, row_index in chosen.items():
        order = np.argsort(-np.abs(dropout_shap[row_index]))[:12]
        for local_rank, feature_index in enumerate(order, start=1):
            value = transformed[row_index, feature_index]
            shap_value = dropout_shap[row_index, feature_index]
            local_rows.append(
                {
                    "case_type": case_type,
                    "record_id": int(test.iloc[row_index]["record_id"]),
                    "actual_outcome": str(test.iloc[row_index]["Target"]),
                    "thresholded_prediction": str(predictions[row_index]),
                    "dropout_probability": float(dropout_probability[row_index]),
                    "rank_within_case": local_rank,
                    "encoded_feature": str(transformed_names[feature_index]),
                    "original_feature": str(original_names[feature_index]),
                    "transformed_feature_value": float(value),
                    "dropout_shap_value": float(shap_value),
                    "effect_direction": "Raises modeled dropout risk"
                    if shap_value > 0
                    else "Lowers modeled dropout risk",
                }
            )
    local_explanations = pd.DataFrame(local_rows)
    local_explanations.to_csv(TABLE_DIR / "shap_local_case_explanations.csv", index=False)

    shap_metadata = {
        "method": "Tree SHAP on the locked Random Forest and held-out test records",
        "explained_class": "Dropout",
        "records_explained": int(len(test)),
        "transformed_features": int(transformed.shape[1]),
        "expected_dropout_value": float(np.asarray(explainer.expected_value)[dropout_index]),
        "local_case_record_ids": {
            case_type: int(test.iloc[index]["record_id"])
            for case_type, index in chosen.items()
        },
        "interpretation": "Positive SHAP values raise the model's dropout probability relative to its expected value; negative values lower it. SHAP explains the model, not causal effects.",
    }

    plot_shap_importance(grouped_importance)
    plot_shap_beeswarm(dropout_shap, transformed, transformed_names)
    plot_local_shap(local_explanations)
    return grouped_importance, local_explanations, shap_metadata


def plot_shap_importance(grouped_importance: pd.DataFrame) -> None:
    top = grouped_importance.head(15).sort_values("mean_absolute_shap")
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.barh(top["original_feature"], top["mean_absolute_shap"], color="#1f807d")
    axis.set(
        xlabel="Sum of mean absolute SHAP values across encoded components",
        ylabel="",
        title="Global drivers of modeled dropout risk",
    )
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "01_shap_global_importance.png", bbox_inches="tight")
    plt.close(figure)


def plot_shap_beeswarm(
    dropout_shap: np.ndarray, transformed: np.ndarray, feature_names: np.ndarray
) -> None:
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        dropout_shap,
        transformed,
        feature_names=feature_names,
        max_display=18,
        show=False,
    )
    plt.title("Direction and magnitude of SHAP contributions to dropout risk", pad=18)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "02_shap_direction_summary.png", bbox_inches="tight", dpi=180)
    plt.close()


def plot_local_shap(local_explanations: pd.DataFrame) -> None:
    case_types = local_explanations["case_type"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(case_types), figsize=(6.5 * len(case_types), 7), squeeze=False)
    for axis, case_type in zip(axes[0], case_types):
        subset = local_explanations[local_explanations["case_type"] == case_type].copy()
        subset = subset.sort_values("dropout_shap_value")
        colors = np.where(subset["dropout_shap_value"] >= 0, "#d1495b", "#1f807d")
        axis.barh(subset["encoded_feature"], subset["dropout_shap_value"], color=colors)
        record_id = int(subset["record_id"].iloc[0])
        probability = float(subset["dropout_probability"].iloc[0])
        axis.axvline(0, color="#555555", linewidth=0.8)
        axis.set(
            xlabel="SHAP contribution to dropout probability",
            ylabel="",
            title=f"{case_type}\nRecord {record_id}, risk {probability:.1%}",
        )
    figure.suptitle("Individual prediction explanations")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "03_shap_local_explanations.png", bbox_inches="tight")
    plt.close(figure)


def compute_pdp_ice(
    bundle: dict[str, Any], test: pd.DataFrame
) -> pd.DataFrame:
    pipeline = bundle["pipeline"]
    model_columns = bundle["model_input_columns"]
    classes = np.asarray(bundle["classes"])
    dropout_index = int(np.where(classes == "Dropout")[0][0])
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = np.sort(rng.choice(len(test), size=min(60, len(test)), replace=False))
    features = ["first_sem_progress_index", "first_sem_approval_rate"]
    rows: list[dict[str, Any]] = []

    for feature in features:
        lower, upper = test[feature].quantile([0.05, 0.95])
        grid = np.linspace(float(lower), float(upper), 25)
        for grid_index, value in enumerate(grid, start=1):
            modified = test[model_columns].copy()
            modified[feature] = value
            probability = pipeline.predict_proba(modified)[:, dropout_index]
            rows.append(
                {
                    "feature": feature,
                    "grid_index": grid_index,
                    "feature_value": value,
                    "curve_type": "PDP average",
                    "record_id": np.nan,
                    "predicted_dropout_probability": float(probability.mean()),
                }
            )
            for row_index in sample_indices:
                rows.append(
                    {
                        "feature": feature,
                        "grid_index": grid_index,
                        "feature_value": value,
                        "curve_type": "ICE individual",
                        "record_id": int(test.iloc[row_index]["record_id"]),
                        "predicted_dropout_probability": float(probability[row_index]),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "pdp_ice_curves.csv", index=False)
    plot_pdp_ice(result)
    return result


def plot_pdp_ice(curves: pd.DataFrame) -> None:
    features = curves["feature"].drop_duplicates().tolist()
    figure, axes = plt.subplots(1, len(features), figsize=(13, 5.5), sharey=True)
    for axis, feature in zip(np.atleast_1d(axes), features):
        subset = curves[curves["feature"] == feature]
        ice = subset[subset["curve_type"] == "ICE individual"]
        for _, group in ice.groupby("record_id"):
            axis.plot(
                group["feature_value"],
                group["predicted_dropout_probability"],
                color="#9eb7b6",
                alpha=0.14,
                linewidth=0.7,
            )
        average = subset[subset["curve_type"] == "PDP average"]
        axis.plot(
            average["feature_value"],
            average["predicted_dropout_probability"],
            color="#d1495b",
            linewidth=3,
            label="Partial dependence",
        )
        axis.set(
            xlabel="Feature value",
            ylabel="Predicted dropout probability",
            title=feature.replace("_", " ").title(),
            ylim=(0, 1),
        )
        axis.legend()
    figure.suptitle("PDP and ICE diagnostics for first-semester progress")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "04_pdp_ice_academic_progress.png", bbox_inches="tight")
    plt.close(figure)


def build_audit_frame(
    test: pd.DataFrame, probabilities: np.ndarray, bundle: dict[str, Any]
) -> pd.DataFrame:
    classes = np.asarray(bundle["classes"])
    dropout_index = int(np.where(classes == "Dropout")[0][0])
    predictions = threshold_predictions(
        probabilities, classes, float(bundle["dropout_alert_threshold"])
    )
    audit = pd.DataFrame(
        {
            "record_id": test["record_id"].astype(int),
            "actual_dropout": test["Target"].astype(str).eq("Dropout"),
            "dropout_alert": predictions == "Dropout",
            "dropout_probability": probabilities[:, dropout_index],
            "Gender": test["Gender"].map({0: "Female", 1: "Male"}),
            "Age group": np.where(
                test["Age at enrollment"] < 25, "Under 25", "25 or older"
            ),
            "International status": test["International"].map(
                {0: "Domestic", 1: "International"}
            ),
            "Financial-risk proxy": np.where(
                (test["Debtor"] == 1) | (test["Tuition fees up to date"] == 0),
                "Financial risk indicated",
                "No immediate financial risk",
            ),
            "Scholarship status": test["Scholarship holder"].map(
                {0: "No scholarship", 1: "Scholarship holder"}
            ),
        }
    )
    audit["Gender x financial risk"] = (
        audit["Gender"].astype(str) + " | " + audit["Financial-risk proxy"].astype(str)
    )
    if audit.isna().any().any():
        raise AssertionError("Unexpected missing audit group label")
    return audit


def binary_group_metrics(group: pd.DataFrame) -> dict[str, Any]:
    actual = group["actual_dropout"].to_numpy(dtype=bool)
    alert = group["dropout_alert"].to_numpy(dtype=bool)
    true_positive = int(np.sum(actual & alert))
    false_positive = int(np.sum((~actual) & alert))
    true_negative = int(np.sum((~actual) & (~alert)))
    false_negative = int(np.sum(actual & (~alert)))
    positives = int(actual.sum())
    negatives = int((~actual).sum())
    alert_count = int(alert.sum())
    alert_low, alert_high = wilson_interval(alert_count, len(group))
    recall_low, recall_high = wilson_interval(true_positive, positives)
    return {
        "n": int(len(group)),
        "actual_dropout_n": positives,
        "observed_dropout_rate": float(actual.mean()),
        "alert_n": alert_count,
        "alert_rate": float(alert.mean()),
        "alert_rate_ci_95_lower": alert_low,
        "alert_rate_ci_95_upper": alert_high,
        "dropout_precision": float(true_positive / alert_count) if alert_count else np.nan,
        "dropout_recall_tpr": float(true_positive / positives) if positives else np.nan,
        "dropout_recall_ci_95_lower": recall_low,
        "dropout_recall_ci_95_upper": recall_high,
        "false_positive_rate": float(false_positive / negatives) if negatives else np.nan,
        "false_negative_rate": float(false_negative / positives) if positives else np.nan,
        "specificity": float(true_negative / negatives) if negatives else np.nan,
        "accuracy": float(accuracy_score(actual, alert)),
        "mean_predicted_dropout_probability": float(group["dropout_probability"].mean()),
        "brier_score": float(brier_score_loss(actual, group["dropout_probability"])),
        "small_sample_warning": bool(
            len(group) < MIN_GROUP_SIZE or positives < MIN_POSITIVE_CASES
        ),
    }


def compute_fairness_metrics(
    audit: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    reference_groups = {
        "Gender": "Female",
        "Age group": "Under 25",
        "International status": "Domestic",
        "Financial-risk proxy": "No immediate financial risk",
        "Scholarship status": "No scholarship",
    }
    intersection_dimension = "Gender x financial risk"
    reference_groups[intersection_dimension] = str(
        audit[intersection_dimension].value_counts().idxmax()
    )

    metric_rows: list[dict[str, Any]] = []
    for dimension, reference in reference_groups.items():
        for group_name, group in audit.groupby(dimension, observed=True):
            metric_rows.append(
                {
                    "audit_dimension": dimension,
                    "group": str(group_name),
                    "reference_group": reference,
                    **binary_group_metrics(group),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(TABLE_DIR / "fairness_group_metrics.csv", index=False)

    gap_rows: list[dict[str, Any]] = []
    for dimension, reference in reference_groups.items():
        subset = metrics[metrics["audit_dimension"] == dimension]
        reference_row = subset[subset["group"] == reference].iloc[0]
        for _, row in subset[subset["group"] != reference].iterrows():
            alert_rate = float(row["alert_rate"])
            reference_alert_rate = float(reference_row["alert_rate"])
            adverse_ratio = (
                alert_rate / reference_alert_rate
                if reference_alert_rate > 0
                else np.nan
            )
            parity_ratio = (
                min(adverse_ratio, 1 / adverse_ratio)
                if adverse_ratio > 0
                else np.nan
            )
            tpr_gap = float(row["dropout_recall_tpr"] - reference_row["dropout_recall_tpr"])
            fpr_gap = float(row["false_positive_rate"] - reference_row["false_positive_rate"])
            gap_rows.append(
                {
                    "audit_dimension": dimension,
                    "group": row["group"],
                    "reference_group": reference,
                    "group_n": int(row["n"]),
                    "reference_n": int(reference_row["n"]),
                    "demographic_parity_difference": float(
                        alert_rate - reference_alert_rate
                    ),
                    "disparate_impact_ratio_adverse_alert": adverse_ratio,
                    "four_fifths_parity_ratio": parity_ratio,
                    "equal_opportunity_tpr_difference": tpr_gap,
                    "false_positive_rate_difference": fpr_gap,
                    "average_odds_difference": float((tpr_gap + fpr_gap) / 2),
                    "equalized_odds_max_gap": float(max(abs(tpr_gap), abs(fpr_gap))),
                    "precision_difference": float(
                        row["dropout_precision"] - reference_row["dropout_precision"]
                    ),
                    "observed_dropout_rate_difference": float(
                        row["observed_dropout_rate"]
                        - reference_row["observed_dropout_rate"]
                    ),
                    "four_fifths_review_flag": bool(
                        np.isfinite(parity_ratio) and parity_ratio < PARITY_RATIO_FLOOR
                    ),
                    "equalized_odds_review_flag": bool(
                        max(abs(tpr_gap), abs(fpr_gap))
                        > EQUALIZED_ODDS_GAP_REVIEW
                    ),
                    "small_sample_warning": bool(
                        row["small_sample_warning"]
                        or reference_row["small_sample_warning"]
                    ),
                }
            )
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(TABLE_DIR / "fairness_gap_metrics.csv", index=False)
    plot_fairness_rates(metrics)
    plot_equalized_odds(metrics)
    plot_fairness_gap_summary(gaps)
    return metrics, gaps, reference_groups


def plot_fairness_rates(metrics: pd.DataFrame) -> None:
    dimensions = [
        "Gender",
        "Age group",
        "International status",
        "Financial-risk proxy",
        "Scholarship status",
    ]
    long = metrics[metrics["audit_dimension"].isin(dimensions)].melt(
        id_vars=["audit_dimension", "group", "n"],
        value_vars=["observed_dropout_rate", "alert_rate"],
        var_name="rate_type",
        value_name="rate",
    )
    long["rate_type"] = long["rate_type"].map(
        {
            "observed_dropout_rate": "Observed dropout rate",
            "alert_rate": "Model alert rate",
        }
    )
    figure, axes = plt.subplots(3, 2, figsize=(15, 14))
    axes_flat = axes.flatten()
    for axis, dimension in zip(axes_flat, dimensions):
        subset = long[long["audit_dimension"] == dimension]
        sns.barplot(
            data=subset,
            x="rate",
            y="group",
            hue="rate_type",
            palette=["#6f5aa8", "#d1495b"],
            ax=axis,
        )
        axis.set(xlim=(0, 1), xlabel="Rate", ylabel="", title=dimension)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.legend(loc="lower right", fontsize=8)
    axes_flat[-1].axis("off")
    figure.suptitle("Observed dropout and model alert rates across audit groups")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "05_group_dropout_and_alert_rates.png", bbox_inches="tight")
    plt.close(figure)


def plot_equalized_odds(metrics: pd.DataFrame) -> None:
    dimensions = [
        "Gender",
        "Age group",
        "International status",
        "Financial-risk proxy",
        "Scholarship status",
    ]
    long = metrics[metrics["audit_dimension"].isin(dimensions)].melt(
        id_vars=["audit_dimension", "group", "small_sample_warning"],
        value_vars=["dropout_recall_tpr", "false_positive_rate"],
        var_name="rate_type",
        value_name="rate",
    )
    long["rate_type"] = long["rate_type"].map(
        {
            "dropout_recall_tpr": "True-positive rate",
            "false_positive_rate": "False-positive rate",
        }
    )
    long["group_label"] = long.apply(
        lambda row: f"{row['group']}*" if row["small_sample_warning"] else row["group"],
        axis=1,
    )
    figure, axes = plt.subplots(3, 2, figsize=(15, 14))
    axes_flat = axes.flatten()
    for axis, dimension in zip(axes_flat, dimensions):
        subset = long[long["audit_dimension"] == dimension]
        sns.barplot(
            data=subset,
            x="rate",
            y="group_label",
            hue="rate_type",
            palette=["#1f807d", "#e3b23c"],
            ax=axis,
        )
        axis.set(xlim=(0, 1), xlabel="Rate", ylabel="", title=dimension)
        axis.xaxis.set_major_formatter(PercentFormatter(1.0))
        axis.legend(loc="lower right", fontsize=8)
    axes_flat[-1].axis("off")
    figure.suptitle("Equalized-odds components by audit group (* small sample)")
    figure.tight_layout(rect=(0.08, 0, 1, 0.96), w_pad=3.5, h_pad=3.0)
    figure.savefig(FIGURE_DIR / "06_group_true_and_false_positive_rates.png", bbox_inches="tight")
    plt.close(figure)


def plot_fairness_gap_summary(gaps: pd.DataFrame) -> None:
    plot_data = gaps[gaps["audit_dimension"] != "Gender x financial risk"].copy()
    plot_data["comparison"] = (
        plot_data["audit_dimension"] + ": " + plot_data["group"]
    )
    plot_data = plot_data.sort_values("equalized_odds_max_gap")
    figure, axes = plt.subplots(1, 2, figsize=(15, 7))
    colors = np.where(plot_data["four_fifths_review_flag"], "#d1495b", "#1f807d")
    axes[0].barh(
        plot_data["comparison"], plot_data["four_fifths_parity_ratio"], color=colors
    )
    axes[0].axvline(PARITY_RATIO_FLOOR, linestyle="--", color="#222222")
    axes[0].set(
        xlim=(0, 1.05),
        xlabel="Symmetric alert-rate parity ratio",
        ylabel="",
        title="Demographic-parity screening",
    )
    colors = np.where(plot_data["equalized_odds_review_flag"], "#d1495b", "#1f807d")
    axes[1].barh(
        plot_data["comparison"], plot_data["equalized_odds_max_gap"], color=colors
    )
    axes[1].axvline(EQUALIZED_ODDS_GAP_REVIEW, linestyle="--", color="#222222")
    axes[1].set(
        xlim=(0, max(0.35, float(plot_data["equalized_odds_max_gap"].max()) + 0.05)),
        xlabel="Maximum absolute TPR/FPR gap",
        ylabel="",
        title="Equalized-odds screening",
    )
    figure.suptitle("Fairness screening flags require contextual review")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "07_fairness_gap_summary.png", bbox_inches="tight")
    plt.close(figure)


def build_generalization_table() -> pd.DataFrame:
    comparison = pd.read_csv(
        ROOT / "reports" / "tables" / "task4" / "cross_validated_model_comparison.csv"
    )
    test_metrics = pd.read_csv(
        ROOT / "reports" / "tables" / "task4" / "heldout_test_metrics.csv"
    )
    with TASK4_SUMMARY_PATH.open(encoding="utf-8") as handle:
        task4_summary = json.load(handle)
    cv_best = comparison[comparison["model"] == task4_summary["best_model"]["name"]].iloc[0]
    default_test = test_metrics[test_metrics["prediction_rule"] == "Default argmax"].iloc[0]
    tuned_test = test_metrics[test_metrics["prediction_rule"].str.startswith("Dropout threshold")].iloc[0]
    oof = task4_summary["best_model"]["oof_threshold_metrics"]
    rows = [
        ("Default dropout F2", cv_best["cv_dropout_f2"], default_test["dropout_f2"]),
        ("Default macro F1", cv_best["cv_f1_macro"], default_test["macro_f1"]),
        (
            "Default weighted OVR ROC-AUC",
            cv_best["cv_roc_auc_ovr_weighted"],
            default_test["roc_auc_ovr_weighted"],
        ),
        ("Thresholded dropout F2", oof["dropout_f2"], tuned_test["dropout_f2"]),
        (
            "Thresholded dropout recall",
            oof["dropout_recall"],
            tuned_test["dropout_recall"],
        ),
        (
            "Thresholded dropout precision",
            oof["dropout_precision"],
            tuned_test["dropout_precision"],
        ),
    ]
    table = pd.DataFrame(rows, columns=["metric", "training_cv_or_oof", "heldout_test"])
    table["heldout_minus_training"] = table["heldout_test"] - table["training_cv_or_oof"]
    table.to_csv(TABLE_DIR / "generalization_gap_review.csv", index=False)
    return table


def mitigation_plan() -> pd.DataFrame:
    rows = [
        {
            "priority": 1,
            "action": "Use alerts only for supportive outreach with human review, documented reasons, and an appeal path.",
            "risk_addressed": "False positives, stigma, and automated adverse decisions",
            "validation_before_use": "Confirm no alert is used for exclusion, discipline, grading, or financial penalties.",
        },
        {
            "priority": 2,
            "action": "Collect better representation data under consent and governance, including self-described gender and race/ethnicity where lawful.",
            "risk_addressed": "Binary gender field and missing race prevent a complete audit",
            "validation_before_use": "Publish purpose, retention, access, minimum cell size, and deletion rules before collection.",
        },
        {
            "priority": 3,
            "action": "Evaluate reweighting or group-aware resampling inside nested cross-validation.",
            "risk_addressed": "Unequal false-negative and false-positive rates",
            "validation_before_use": "Compare recall, precision, calibration, and equalized-odds gaps on a new validation cohort.",
        },
        {
            "priority": 4,
            "action": "Calibrate dropout probabilities using training-only cross-validation.",
            "risk_addressed": "Probability miscalibration across risk ranges and groups",
            "validation_before_use": "Check Brier score and reliability curves overall and by sufficiently large group.",
        },
        {
            "priority": 5,
            "action": "Consider group-specific thresholds only after legal and ethical review.",
            "risk_addressed": "Recall gaps that a single threshold cannot resolve",
            "validation_before_use": "Tune on validation data, document the fairness objective, and test for new precision or burden disparities.",
        },
        {
            "priority": 6,
            "action": "Monitor performance, alert rates, TPR, FPR, calibration, and data drift by cohort and group.",
            "risk_addressed": "Temporal drift and degradation after deployment",
            "validation_before_use": "Set minimum sample sizes, confidence intervals, escalation thresholds, and rollback ownership.",
        },
    ]
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "mitigation_plan.csv", index=False)
    return result


def format_percent(value: float) -> str:
    return "not estimable" if not np.isfinite(value) else f"{value:.1%}"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = frame.columns.tolist()
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        cleaned = [str(value).replace("|", "\\|").replace("\n", " ") for value in values]
        rows.append("| " + " | ".join(cleaned) + " |")
    return "\n".join([header, separator, *rows])


def write_report(
    summary: dict[str, Any],
    metrics: pd.DataFrame,
    gaps: pd.DataFrame,
    grouped_shap: pd.DataFrame,
    generalization: pd.DataFrame,
    mitigations: pd.DataFrame,
) -> None:
    def metric(dimension: str, group: str, column: str) -> float:
        return float(
            metrics.loc[
                (metrics["audit_dimension"] == dimension) & (metrics["group"] == group),
                column,
            ].iloc[0]
        )

    top_features = ", ".join(grouped_shap.head(5)["original_feature"].tolist())
    gender_gap = gaps[
        (gaps["audit_dimension"] == "Gender") & (gaps["group"] == "Male")
    ].iloc[0]
    age_gap = gaps[
        (gaps["audit_dimension"] == "Age group")
        & (gaps["group"] == "25 or older")
    ].iloc[0]
    financial_gap = gaps[
        (gaps["audit_dimension"] == "Financial-risk proxy")
        & (gaps["group"] == "Financial risk indicated")
    ].iloc[0]

    mitigation_markdown = markdown_table(mitigations)
    report = f"""# Bias & Fairness Analysis

## Scope and decision context

This analysis audits the locked Task 4 Random Forest on the 885-record held-out test set. A `Dropout` prediction at the saved {summary['model']['dropout_alert_threshold']:.2f} probability threshold is treated as the adverse selection event for fairness measurement. In practice, an alert can create both benefit (earlier support) and harm (stigma, unnecessary contact, or punitive misuse), so alert-rate parity alone does not determine fairness.

Gender, age, and international status were excluded from the model inputs and retained only for auditing. Financial-risk and scholarship indicators are audited as incomplete socioeconomic proxies. Race is absent from the dataset. Nationality or international status is not treated as race.

## Model explanations

Tree SHAP was computed for all held-out records. The leading grouped drivers of modeled dropout risk were {top_features}. Positive SHAP values raise modeled dropout probability relative to the model's expected value; negative values lower it. Grouped importance sums one-hot component contributions, so high-cardinality variables such as course and application mode require cautious comparison. These explanations describe model behavior, not causal effects or recommended interventions.

PDP and ICE curves were generated for first-semester progress and approval rate. They show the model's average and individual response when each feature changes while other inputs remain fixed. Because engineered academic features are related to their source variables, these plots are diagnostic and should not be interpreted as feasible or causal interventions.

## Fairness metrics

The audit reports observed dropout rate, alert rate, precision, true-positive rate (dropout recall), false-positive rate, calibration, demographic-parity difference, adverse-alert disparate-impact ratio, equal-opportunity difference, and equalized-odds gaps. Reference groups are chosen explicitly for stable comparison, not labeled inherently privileged.

- **Gender:** female alert rate {format_percent(metric('Gender', 'Female', 'alert_rate'))}; male alert rate {format_percent(metric('Gender', 'Male', 'alert_rate'))}. Female dropout recall is {format_percent(metric('Gender', 'Female', 'dropout_recall_tpr'))}; male recall is {format_percent(metric('Gender', 'Male', 'dropout_recall_tpr'))}. The maximum TPR/FPR gap is {format_percent(float(gender_gap['equalized_odds_max_gap']))}.
- **Age:** under-25 alert rate {format_percent(metric('Age group', 'Under 25', 'alert_rate'))}; age-25-or-older alert rate {format_percent(metric('Age group', '25 or older', 'alert_rate'))}. The maximum TPR/FPR gap is {format_percent(float(age_gap['equalized_odds_max_gap']))}.
- **Financial-risk proxy:** no-immediate-risk alert rate {format_percent(metric('Financial-risk proxy', 'No immediate financial risk', 'alert_rate'))}; financial-risk-indicated alert rate {format_percent(metric('Financial-risk proxy', 'Financial risk indicated', 'alert_rate'))}. The adverse-alert rate ratio is {float(financial_gap['disparate_impact_ratio_adverse_alert']):.2f}.
- **International status:** the international subgroup has only {int(metric('International status', 'International', 'n'))} test records. Its metrics are reported but are too unstable for a strong conclusion.

Demographic-parity gaps partly track real outcome-rate differences in this historical dataset. That does not prove the alert policy is fair: historical outcomes may themselves reflect unequal access, financial constraints, or institutional processes. Equalized-odds components and the practical consequences of false positives and false negatives deserve more weight for a supportive intervention.

## Imbalance, leakage, and overfitting

The minority `Enrolled` class represents about 18% of records and dropout about 32%. Task 4 used stratification, class weighting, macro metrics, dropout-specific metrics, and a fixed-capacity capture metric rather than accuracy alone.

Temporal leakage was reduced by excluding all second-semester academic variables. Preprocessing, feature selection, model selection, and threshold selection used only the training partition. Task 5 uses the held-out data only to audit the already locked model.

The held-out-minus-training gaps range from {generalization['heldout_minus_training'].min():.3f} to {generalization['heldout_minus_training'].max():.3f} across the reviewed metrics. This suggests a modest generalization gap in this split, not proof that overfitting is absent. External and temporal validation are still required.

## Data and ethical limitations

- The data comes from one Portuguese higher-education institution, limiting external validity.
- Race is unavailable. The audit cannot assess racial fairness, and nationality is not a valid replacement.
- Gender is encoded as a binary field and may not reflect identity or nonbinary students.
- International and educational-special-needs groups are small, producing unstable estimates.
- Financial-risk and scholarship indicators do not measure income, wealth, caregiving, disability-related costs, or access to support.
- Historical dropout labels can encode institutional and structural inequities.
- SHAP, PDP, subgroup associations, and fairness gaps are descriptive rather than causal.
- A threshold that improves recall increases false-positive outreach and reduces overall multiclass performance.

## Mitigation plan

{mitigation_markdown}

## Decision

The model should not be deployed for automated adverse decisions. It is suitable only for a controlled, non-punitive support pilot after review of the flagged group disparities, training-only mitigation experiments, probability calibration, human oversight, an appeal process, and prospective monitoring. A new cohort should be reserved for validating any mitigation or threshold change.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def run_task5() -> Task5Artifacts:
    ensure_directories()
    configure_plots()
    test = pd.read_csv(TEST_PATH)
    bundle = joblib.load(BUNDLE_PATH)
    pipeline = bundle["pipeline"]
    model_columns = bundle["model_input_columns"]
    probabilities = pipeline.predict_proba(test[model_columns])

    grouped_shap, local_explanations, shap_metadata = compute_shap_explanations(
        bundle, test, probabilities
    )
    pdp_ice = compute_pdp_ice(bundle, test)
    audit = build_audit_frame(test, probabilities, bundle)
    audit.to_csv(TABLE_DIR / "heldout_audit_predictions.csv", index=False)
    group_metrics, fairness_gaps, reference_groups = compute_fairness_metrics(audit)
    generalization = build_generalization_table()
    mitigations = mitigation_plan()

    highest_eo = fairness_gaps.loc[
        fairness_gaps["equalized_odds_max_gap"].idxmax()
    ]
    parity_flags = fairness_gaps[fairness_gaps["four_fifths_review_flag"]]
    eo_flags = fairness_gaps[fairness_gaps["equalized_odds_review_flag"]]
    small_groups = group_metrics[group_metrics["small_sample_warning"]][
        ["audit_dimension", "group", "n", "actual_dropout_n"]
    ].to_dict(orient="records")

    summary = {
        "task": "Task 5: Critical Thinking, Ethical AI, and Bias Auditing",
        "model": {
            "name": bundle["model_name"],
            "dropout_alert_threshold": float(bundle["dropout_alert_threshold"]),
            "heldout_records_audited": int(len(test)),
            "sensitive_fields_in_model_inputs": [],
            "sensitive_fields_used_for_audit_only": [
                "Gender",
                "Age at enrollment",
                "International",
            ],
            "socioeconomic_proxies_audited": [
                "Debtor",
                "Tuition fees up to date",
                "Scholarship holder",
            ],
        },
        "explainability": {
            **shap_metadata,
            "top_10_grouped_features": grouped_shap.head(10).to_dict(orient="records"),
            "pdp_ice_features": pdp_ice["feature"].drop_duplicates().tolist(),
        },
        "fairness_definition": {
            "event": "Dropout alert is the adverse selection event",
            "reference_groups": reference_groups,
            "demographic_parity": "Difference and ratio of adverse alert rates",
            "disparate_impact": "Group adverse alert rate divided by reference adverse alert rate; the symmetric four-fifths ratio is also reported",
            "equal_opportunity": "Difference in dropout true-positive rate",
            "equalized_odds": "True-positive and false-positive rate gaps",
            "screening_thresholds": {
                "four_fifths_parity_ratio_below": PARITY_RATIO_FLOOR,
                "equalized_odds_max_gap_above": EQUALIZED_ODDS_GAP_REVIEW,
            },
        },
        "fairness_results": {
            "four_fifths_review_flags": int(len(parity_flags)),
            "equalized_odds_review_flags": int(len(eo_flags)),
            "largest_equalized_odds_gap": {
                "audit_dimension": str(highest_eo["audit_dimension"]),
                "group": str(highest_eo["group"]),
                "reference_group": str(highest_eo["reference_group"]),
                "gap": float(highest_eo["equalized_odds_max_gap"]),
            },
            "small_sample_groups": small_groups,
            "race_audit_status": "Not possible: race is absent. Nationality and international status are not treated as race.",
            "interpretation": "Flags are screening signals, not legal findings or proof of discrimination. Base-rate differences and intervention harms/benefits require contextual review.",
        },
        "risk_controls": {
            "imbalance": "Stratification, class weighting, macro metrics, dropout-specific metrics, and capacity-based capture metrics were used.",
            "leakage": "Second-semester variables were excluded; all learned Task 3 and Task 4 steps used training data before held-out evaluation.",
            "overfitting": "Cross-validated or OOF metrics were compared with held-out results; external and temporal validation remain required.",
        },
        "recommendation": "Do not use the model for automated adverse decisions. Consider only a non-punitive support pilot after mitigation testing, human review, appeal, calibration, and prospective monitoring.",
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "shap": shap.__version__,
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    config = {
        "random_state": RANDOM_STATE,
        "audit_partition": "Held-out test set; locked model and threshold",
        "audit_event": "Thresholded Dropout alert",
        "minimum_group_size": MIN_GROUP_SIZE,
        "minimum_positive_cases": MIN_POSITIVE_CASES,
        "four_fifths_parity_ratio_floor": PARITY_RATIO_FLOOR,
        "equalized_odds_review_gap": EQUALIZED_ODDS_GAP_REVIEW,
        "reference_groups": reference_groups,
        "race_handling": "Unavailable; no proxy substitution",
    }
    (CONFIG_DIR / "task5_audit_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    write_report(
        summary,
        group_metrics,
        fairness_gaps,
        grouped_shap,
        generalization,
        mitigations,
    )

    return Task5Artifacts(
        summary=summary,
        group_metrics=group_metrics,
        fairness_gaps=fairness_gaps,
        shap_importance=grouped_shap,
        local_explanations=local_explanations,
    )


def main() -> None:
    artifacts = run_task5()
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
