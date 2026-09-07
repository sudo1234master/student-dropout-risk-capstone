"""Task 4: cross-validated model comparison and held-out evaluation."""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

try:
    from src.task3_analysis import define_model_columns, make_preprocessor
except ModuleNotFoundError:  # Support direct execution as ``python src/task4_modeling.py``.
    from task3_analysis import define_model_columns, make_preprocessor


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "task3" / "train_primary.csv"
TEST_PATH = ROOT / "data" / "processed" / "task3" / "test_primary.csv"
MODEL_DIR = ROOT / "models" / "task4"
CONFIG_DIR = ROOT / "configs"
TABLE_DIR = ROOT / "reports" / "tables" / "task4"
FIGURE_DIR = ROOT / "reports" / "figures" / "task4"
SUMMARY_PATH = ROOT / "reports" / "task4_model_summary.json"

RANDOM_STATE = 42
CV_FOLDS = 5
TOP_RISK_SHARE = 0.30
TARGET_DROPOUT_RECALL = 0.75
MIN_DROPOUT_PRECISION = 0.50
LABELS = ["Dropout", "Enrolled", "Graduate"]


@dataclass
class Task4Artifacts:
    summary: dict[str, Any]
    comparison: pd.DataFrame
    test_metrics: pd.DataFrame
    threshold_table: pd.DataFrame
    best_model_name: str
    best_threshold: float


def ensure_directories() -> None:
    for directory in [MODEL_DIR, CONFIG_DIR, TABLE_DIR, FIGURE_DIR]:
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


def dropout_precision_metric(y_true: Any, y_pred: Any) -> float:
    return float(
        precision_score(
            np.asarray(y_true) == "Dropout",
            np.asarray(y_pred) == "Dropout",
            zero_division=0,
        )
    )


def dropout_recall_metric(y_true: Any, y_pred: Any) -> float:
    return float(
        recall_score(
            np.asarray(y_true) == "Dropout",
            np.asarray(y_pred) == "Dropout",
            zero_division=0,
        )
    )


def dropout_f2_metric(y_true: Any, y_pred: Any) -> float:
    return float(
        fbeta_score(
            np.asarray(y_true) == "Dropout",
            np.asarray(y_pred) == "Dropout",
            beta=2,
            zero_division=0,
        )
    )


def model_scoring() -> dict[str, Any]:
    from sklearn.metrics import make_scorer

    return {
        "dropout_f2": make_scorer(dropout_f2_metric),
        "dropout_recall": make_scorer(dropout_recall_metric),
        "dropout_precision": make_scorer(dropout_precision_metric),
        "f1_macro": "f1_macro",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "roc_auc_ovr_weighted": "roc_auc_ovr_weighted",
    }


def selector() -> SelectKBest:
    score_function = partial(mutual_info_classif, random_state=RANDOM_STATE)
    return SelectKBest(score_func=score_function, k=30)


def build_pipeline(
    primary: pd.DataFrame, estimator: Any, include_selector: bool = True
) -> tuple[Pipeline, list[str]]:
    model_columns, categorical_columns, numeric_columns = define_model_columns(primary)
    steps: list[tuple[str, Any]] = [
        ("preprocessor", make_preprocessor(categorical_columns, numeric_columns))
    ]
    if include_selector:
        steps.append(("selector", selector()))
    steps.append(("model", estimator))
    return Pipeline(steps=steps), model_columns


def candidate_specs() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(
                max_iter=3000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "selector__k": [30, "all"],
                "model__C": [0.1, 1.0, 10.0],
            },
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                class_weight="balanced",
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),
            {
                "selector__k": [30, "all"],
                "model__max_depth": [4, 8, None],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced_subsample",
                n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            {
                "selector__k": [30, "all"],
                "model__min_samples_leaf": [2, 5],
                "model__max_features": ["sqrt", 0.5],
            },
        ),
        "Histogram Gradient Boosting": (
            HistGradientBoostingClassifier(
                max_iter=250,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "selector__k": [30, "all"],
                "model__learning_rate": [0.05, 0.10],
                "model__max_leaf_nodes": [15, 31],
            },
        ),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def compare_models(
    train: pd.DataFrame,
    cv: StratifiedKFold,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, pd.DataFrame], list[str]]:
    scoring = model_scoring()
    y_train = train["Target"].astype(str)

    dummy, model_columns = build_pipeline(
        train,
        DummyClassifier(strategy="prior", random_state=RANDOM_STATE),
        include_selector=False,
    )
    dummy_scores = cross_validate(
        dummy,
        train[model_columns],
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score="raise",
    )
    dummy.fit(train[model_columns], y_train)

    metric_names = list(scoring)
    comparison_rows = [
        {
            "model": "Dummy Baseline",
            "best_parameters": json.dumps({"strategy": "prior"}),
            **{
                f"cv_{metric}": float(np.mean(dummy_scores[f"test_{metric}"]))
                for metric in metric_names
            },
            **{
                f"cv_{metric}_std": float(np.std(dummy_scores[f"test_{metric}"], ddof=1))
                for metric in metric_names
            },
        }
    ]
    fitted_models: dict[str, Pipeline] = {"Dummy Baseline": dummy}
    detailed_results: dict[str, pd.DataFrame] = {}

    for name, (estimator, grid) in candidate_specs().items():
        pipeline, candidate_columns = build_pipeline(train, estimator)
        if candidate_columns != model_columns:
            raise AssertionError("Candidate model columns do not match")
        search = GridSearchCV(
            pipeline,
            param_grid=grid,
            scoring=scoring,
            refit="dropout_f2",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )
        search.fit(train[model_columns], y_train)
        index = search.best_index_
        row: dict[str, Any] = {
            "model": name,
            "best_parameters": json.dumps(json_safe(search.best_params_), sort_keys=True),
        }
        for metric in metric_names:
            row[f"cv_{metric}"] = float(search.cv_results_[f"mean_test_{metric}"][index])
            row[f"cv_{metric}_std"] = float(search.cv_results_[f"std_test_{metric}"][index])
        comparison_rows.append(row)
        fitted_models[name] = search.best_estimator_

        details = pd.DataFrame(search.cv_results_)
        keep = [
            "params",
            "rank_test_dropout_f2",
            *[f"mean_test_{metric}" for metric in metric_names],
            *[f"std_test_{metric}" for metric in metric_names],
        ]
        details = details[keep].copy()
        details["params"] = details["params"].map(
            lambda item: json.dumps(json_safe(item), sort_keys=True)
        )
        detailed_results[name] = details.sort_values("rank_test_dropout_f2")

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["cv_dropout_f2", "cv_f1_macro"], ascending=False
    )
    comparison.insert(0, "rank", np.arange(1, len(comparison) + 1))
    comparison.to_csv(TABLE_DIR / "cross_validated_model_comparison.csv", index=False)
    for name, details in detailed_results.items():
        slug = name.lower().replace(" ", "_")
        details.to_csv(TABLE_DIR / f"grid_results_{slug}.csv", index=False)

    for name, pipeline in fitted_models.items():
        slug = name.lower().replace(" ", "_")
        joblib.dump(pipeline, MODEL_DIR / f"{slug}_pipeline.joblib")

    return comparison, fitted_models, detailed_results, model_columns


def select_best_model(comparison: pd.DataFrame) -> str:
    candidates = comparison[comparison["model"] != "Dummy Baseline"].copy()
    eligible = candidates[candidates["cv_dropout_precision"] >= MIN_DROPOUT_PRECISION]
    pool = eligible if not eligible.empty else candidates
    return str(
        pool.sort_values(["cv_dropout_f2", "cv_f1_macro"], ascending=False)
        .iloc[0]["model"]
    )


def threshold_predictions(
    probabilities: np.ndarray, classes: np.ndarray, threshold: float
) -> np.ndarray:
    class_array = np.asarray(classes)
    dropout_index = int(np.where(class_array == "Dropout")[0][0])
    non_dropout_index = np.where(class_array != "Dropout")[0]
    prediction = class_array[
        non_dropout_index[np.argmax(probabilities[:, non_dropout_index], axis=1)]
    ].astype(object)
    prediction[probabilities[:, dropout_index] >= threshold] = "Dropout"
    return prediction.astype(str)


def evaluate_prediction_rule(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
    predictions: np.ndarray,
    rule_name: str,
) -> dict[str, Any]:
    return {
        "prediction_rule": rule_name,
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted")),
        "dropout_precision": dropout_precision_metric(y_true, predictions),
        "dropout_recall": dropout_recall_metric(y_true, predictions),
        "dropout_f1": float(
            f1_score(
                np.asarray(y_true) == "Dropout",
                np.asarray(predictions) == "Dropout",
                zero_division=0,
            )
        ),
        "dropout_f2": dropout_f2_metric(y_true, predictions),
        "multiclass_log_loss": float(log_loss(y_true, probabilities, labels=classes)),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="macro",
            )
        ),
        "roc_auc_ovr_weighted": float(
            roc_auc_score(
                y_true,
                probabilities,
                labels=classes,
                multi_class="ovr",
                average="weighted",
            )
        ),
    }


def threshold_search(
    y_true: pd.Series, probabilities: np.ndarray, classes: np.ndarray
) -> tuple[pd.DataFrame, float, str]:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        prediction = threshold_predictions(probabilities, classes, float(threshold))
        rows.append(
            {
                "threshold": float(threshold),
                "dropout_precision": dropout_precision_metric(y_true, prediction),
                "dropout_recall": dropout_recall_metric(y_true, prediction),
                "dropout_f1": float(
                    f1_score(
                        np.asarray(y_true) == "Dropout",
                        prediction == "Dropout",
                        zero_division=0,
                    )
                ),
                "dropout_f2": dropout_f2_metric(y_true, prediction),
                "macro_f1": float(f1_score(y_true, prediction, average="macro")),
                "alert_share": float(np.mean(prediction == "Dropout")),
            }
        )
    table = pd.DataFrame(rows)
    meets_both = table[
        (table["dropout_precision"] >= MIN_DROPOUT_PRECISION)
        & (table["dropout_recall"] >= TARGET_DROPOUT_RECALL)
    ]
    if not meets_both.empty:
        pool = meets_both
        reason = "Maximised out-of-fold dropout F2 among thresholds meeting the Task 1 recall and precision targets."
    else:
        precision_pool = table[
            table["dropout_precision"] >= MIN_DROPOUT_PRECISION
        ]
        pool = precision_pool if not precision_pool.empty else table
        reason = "No threshold met both Task 1 targets; selected the best out-of-fold dropout F2 under the precision constraint when feasible."
    selected = pool.sort_values(
        ["dropout_f2", "macro_f1", "dropout_recall"], ascending=False
    ).iloc[0]
    table["selected"] = np.isclose(table["threshold"], selected["threshold"])
    table.to_csv(TABLE_DIR / "oof_threshold_search.csv", index=False)
    return table, float(selected["threshold"]), reason


def capacity_metrics(
    y_true: pd.Series,
    dropout_probabilities: np.ndarray,
    share: float = TOP_RISK_SHARE,
) -> dict[str, Any]:
    count = int(np.ceil(len(y_true) * share))
    ordering = np.argsort(-dropout_probabilities)
    selected = ordering[:count]
    actual_dropout = (np.asarray(y_true) == "Dropout").astype(int)
    captured = int(actual_dropout[selected].sum())
    total_dropout = int(actual_dropout.sum())
    precision_at_capacity = float(captured / count)
    prevalence = float(actual_dropout.mean())
    return {
        "capacity_share": float(share),
        "students_flagged": count,
        "dropouts_captured": captured,
        "total_dropouts": total_dropout,
        "dropout_capture_rate": float(captured / total_dropout),
        "precision_at_capacity": precision_at_capacity,
        "lift_at_capacity": float(precision_at_capacity / prevalence),
        "baseline_dropout_prevalence": prevalence,
    }


def bootstrap_intervals(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
    threshold: float,
    iterations: int = 1000,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    values: dict[str, list[float]] = {
        "dropout_precision": [],
        "dropout_recall": [],
        "dropout_f2": [],
        "macro_f1": [],
    }
    y_array = np.asarray(y_true)
    for _ in range(iterations):
        indices = rng.integers(0, len(y_array), size=len(y_array))
        y_sample = y_array[indices]
        probability_sample = probabilities[indices]
        prediction = threshold_predictions(probability_sample, classes, threshold)
        values["dropout_precision"].append(
            dropout_precision_metric(y_sample, prediction)
        )
        values["dropout_recall"].append(dropout_recall_metric(y_sample, prediction))
        values["dropout_f2"].append(dropout_f2_metric(y_sample, prediction))
        values["macro_f1"].append(
            float(f1_score(y_sample, prediction, average="macro"))
        )
    rows = []
    for metric, samples in values.items():
        rows.append(
            {
                "metric": metric,
                "bootstrap_mean": float(np.mean(samples)),
                "ci_95_lower": float(np.quantile(samples, 0.025)),
                "ci_95_upper": float(np.quantile(samples, 0.975)),
                "iterations": iterations,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(TABLE_DIR / "test_bootstrap_confidence_intervals.csv", index=False)
    return result


def save_test_outputs(
    test: pd.DataFrame,
    probabilities: np.ndarray,
    classes: np.ndarray,
    tuned_prediction: np.ndarray,
    default_prediction: np.ndarray,
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dropout_index = int(np.where(classes == "Dropout")[0][0])
    risk = probabilities[:, dropout_index]
    top_count = int(np.ceil(len(test) * TOP_RISK_SHARE))
    top_order = np.argsort(-risk)
    top_flag = np.zeros(len(test), dtype=bool)
    top_flag[top_order[:top_count]] = True
    rank = np.empty(len(test), dtype=int)
    rank[top_order] = np.arange(1, len(test) + 1)

    predictions = pd.DataFrame(
        {
            "record_id": test["record_id"].to_numpy(),
            "actual_outcome": test["Target"].astype(str).to_numpy(),
            **{
                f"probability_{label.lower()}": probabilities[:, index]
                for index, label in enumerate(classes)
            },
            "default_prediction": default_prediction,
            "thresholded_prediction": tuned_prediction,
            "dropout_alert_threshold": threshold,
            "dropout_risk_rank": rank,
            "top_30pct_risk_flag": top_flag,
        }
    ).sort_values("dropout_risk_rank")
    predictions.to_csv(TABLE_DIR / "heldout_test_predictions.csv", index=False)

    default_cm = pd.DataFrame(
        confusion_matrix(test["Target"], default_prediction, labels=LABELS),
        index=LABELS,
        columns=LABELS,
    )
    tuned_cm = pd.DataFrame(
        confusion_matrix(test["Target"], tuned_prediction, labels=LABELS),
        index=LABELS,
        columns=LABELS,
    )
    tidy_confusion = pd.concat(
        [
            default_cm.rename_axis("actual").reset_index().melt(
                id_vars="actual", var_name="predicted", value_name="count"
            ).assign(prediction_rule="Default argmax"),
            tuned_cm.rename_axis("actual").reset_index().melt(
                id_vars="actual", var_name="predicted", value_name="count"
            ).assign(prediction_rule="Tuned dropout threshold"),
        ],
        ignore_index=True,
    )
    tidy_confusion.to_csv(TABLE_DIR / "heldout_confusion_matrices.csv", index=False)

    report = pd.DataFrame(
        classification_report(
            test["Target"],
            tuned_prediction,
            labels=LABELS,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
    report.index.name = "class_or_average"
    report.to_csv(TABLE_DIR / "heldout_thresholded_classification_report.csv")
    return predictions, default_cm, tuned_cm


def plot_cv_comparison(comparison: pd.DataFrame) -> None:
    ordered = comparison.sort_values("cv_dropout_f2")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    specs = [
        ("cv_dropout_f2", "cv_dropout_f2_std", "Dropout F2", "#1f807d"),
        ("cv_f1_macro", "cv_f1_macro_std", "Macro F1", "#6f5aa8"),
    ]
    for axis, (metric, std, title, color) in zip(axes, specs):
        axis.barh(
            ordered["model"],
            ordered[metric],
            xerr=ordered[std],
            color=color,
            alpha=0.9,
            capsize=3,
        )
        axis.set_xlim(0, 1)
        axis.set_xlabel("Five-fold cross-validation score")
        axis.set_title(title)
    figure.suptitle("Cross-validated model comparison on the training partition")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "01_cross_validated_model_comparison.png", bbox_inches="tight")
    plt.close(figure)


def plot_threshold_tradeoff(table: pd.DataFrame, selected_threshold: float) -> None:
    figure, axis = plt.subplots(figsize=(10, 6))
    for metric, color in [
        ("dropout_precision", "#33658a"),
        ("dropout_recall", "#d1495b"),
        ("dropout_f2", "#1f807d"),
    ]:
        axis.plot(table["threshold"], table[metric], label=metric.replace("_", " ").title(), color=color)
    axis.axvline(selected_threshold, color="#222222", linestyle="--", label=f"Selected: {selected_threshold:.2f}")
    axis.axhline(TARGET_DROPOUT_RECALL, color="#d1495b", linestyle=":", alpha=0.6)
    axis.axhline(MIN_DROPOUT_PRECISION, color="#33658a", linestyle=":", alpha=0.6)
    axis.set(xlabel="Dropout alert threshold", ylabel="Out-of-fold score", ylim=(0, 1))
    axis.set_title("Training-only threshold trade-off")
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "02_threshold_tradeoff.png", bbox_inches="tight")
    plt.close(figure)


def plot_confusion_matrices(default_cm: pd.DataFrame, tuned_cm: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, matrix, title in [
        (axes[0], default_cm, "Default argmax"),
        (axes[1], tuned_cm, "Tuned dropout threshold"),
    ]:
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
        axis.set(xlabel="Predicted outcome", ylabel="Actual outcome", title=title)
    figure.suptitle("Held-out test confusion matrices")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "03_heldout_confusion_matrices.png", bbox_inches="tight")
    plt.close(figure)


def plot_roc_and_precision_recall(
    y_true: pd.Series,
    probabilities: np.ndarray,
    classes: np.ndarray,
    threshold: float,
) -> None:
    y_binary = label_binarize(y_true, classes=classes)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    palette = {"Dropout": "#d1495b", "Enrolled": "#e3b23c", "Graduate": "#1f807d"}
    for index, label in enumerate(classes):
        false_positive, true_positive, _ = roc_curve(y_binary[:, index], probabilities[:, index])
        auc = roc_auc_score(y_binary[:, index], probabilities[:, index])
        axes[0].plot(false_positive, true_positive, label=f"{label} (AUC {auc:.3f})", color=palette[label])
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777")
    axes[0].set(xlabel="False-positive rate", ylabel="True-positive rate", title="One-vs-rest ROC curves")
    axes[0].legend()

    dropout_index = int(np.where(classes == "Dropout")[0][0])
    dropout_true = y_binary[:, dropout_index]
    precision, recall, thresholds = precision_recall_curve(dropout_true, probabilities[:, dropout_index])
    axes[1].plot(recall, precision, color=palette["Dropout"], label="Dropout precision-recall")
    if len(thresholds):
        nearest = int(np.argmin(np.abs(thresholds - threshold)))
        axes[1].scatter(recall[nearest], precision[nearest], color="#222222", s=55, zorder=5, label=f"Threshold {threshold:.2f}")
    axes[1].axhline(float(np.mean(dropout_true)), linestyle="--", color="#777777", label="Dropout prevalence")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="Held-out dropout precision-recall", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    figure.suptitle("Held-out discrimination performance")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "04_heldout_roc_and_precision_recall.png", bbox_inches="tight")
    plt.close(figure)


def plot_capacity_and_calibration(
    y_true: pd.Series, dropout_probabilities: np.ndarray
) -> None:
    actual = (np.asarray(y_true) == "Dropout").astype(int)
    ordering = np.argsort(-dropout_probabilities)
    cumulative = np.cumsum(actual[ordering]) / actual.sum()
    population = np.arange(1, len(actual) + 1) / len(actual)
    capacity_index = int(np.ceil(TOP_RISK_SHARE * len(actual))) - 1

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(population, cumulative, color="#1f807d", linewidth=2)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Random ranking")
    axes[0].axvline(TOP_RISK_SHARE, linestyle=":", color="#222222")
    axes[0].scatter(population[capacity_index], cumulative[capacity_index], color="#d1495b", zorder=5)
    axes[0].annotate(
        f"{cumulative[capacity_index]:.1%} captured\nat {TOP_RISK_SHARE:.0%} capacity",
        (population[capacity_index], cumulative[capacity_index]),
        xytext=(12, -35),
        textcoords="offset points",
    )
    axes[0].set(xlabel="Share of students reviewed", ylabel="Share of dropouts captured", title="Cumulative dropout capture", xlim=(0, 1), ylim=(0, 1))
    axes[0].legend()

    observed, predicted = calibration_curve(actual, dropout_probabilities, n_bins=10, strategy="quantile")
    axes[1].plot(predicted, observed, marker="o", color="#6f5aa8", label="Best model")
    axes[1].plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect calibration")
    axes[1].set(xlabel="Mean predicted dropout probability", ylabel="Observed dropout rate", title="Dropout calibration", xlim=(0, 1), ylim=(0, 1))
    axes[1].legend()
    figure.suptitle("Held-out operational performance")
    figure.tight_layout()
    figure.savefig(FIGURE_DIR / "05_heldout_capacity_and_calibration.png", bbox_inches="tight")
    plt.close(figure)


def run_task4() -> Task4Artifacts:
    ensure_directories()
    configure_plots()

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    train["Target"] = train["Target"].astype(str)
    test["Target"] = test["Target"].astype(str)
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    comparison, fitted_models, detailed_results, model_columns = compare_models(train, cv)
    best_model_name = select_best_model(comparison)
    best_model = fitted_models[best_model_name]

    oof_probabilities = cross_val_predict(
        clone(best_model),
        train[model_columns],
        train["Target"],
        cv=cv,
        method="predict_proba",
        n_jobs=-1,
    )
    classes = np.asarray(best_model.classes_)
    threshold_table, best_threshold, threshold_reason = threshold_search(
        train["Target"], oof_probabilities, classes
    )
    oof_capacity = capacity_metrics(
        train["Target"], oof_probabilities[:, int(np.where(classes == "Dropout")[0][0])]
    )

    test_probabilities = best_model.predict_proba(test[model_columns])
    default_prediction = best_model.predict(test[model_columns])
    tuned_prediction = threshold_predictions(
        test_probabilities, classes, best_threshold
    )
    test_metric_rows = [
        evaluate_prediction_rule(
            test["Target"],
            test_probabilities,
            classes,
            default_prediction,
            "Default argmax",
        ),
        evaluate_prediction_rule(
            test["Target"],
            test_probabilities,
            classes,
            tuned_prediction,
            f"Dropout threshold {best_threshold:.2f}",
        ),
    ]
    test_metrics = pd.DataFrame(test_metric_rows)
    test_metrics.to_csv(TABLE_DIR / "heldout_test_metrics.csv", index=False)

    dropout_index = int(np.where(classes == "Dropout")[0][0])
    test_capacity = capacity_metrics(
        test["Target"], test_probabilities[:, dropout_index]
    )
    pd.DataFrame([{"partition": "Training OOF", **oof_capacity}, {"partition": "Held-out test", **test_capacity}]).to_csv(
        TABLE_DIR / "top_30pct_capacity_metrics.csv", index=False
    )

    predictions, default_cm, tuned_cm = save_test_outputs(
        test,
        test_probabilities,
        classes,
        tuned_prediction,
        default_prediction,
        best_threshold,
    )
    intervals = bootstrap_intervals(
        test["Target"], test_probabilities, classes, best_threshold
    )

    best_parameters = json.loads(
        comparison.loc[
            comparison["model"] == best_model_name, "best_parameters"
        ].iloc[0]
    )
    model_bundle = {
        "pipeline": best_model,
        "model_name": best_model_name,
        "classes": classes.tolist(),
        "dropout_alert_threshold": best_threshold,
        "model_input_columns": model_columns,
        "prediction_rule": "Flag Dropout when P(Dropout) meets the saved threshold; otherwise choose the higher-probability non-dropout class.",
        "top_risk_capacity_share": TOP_RISK_SHARE,
    }
    joblib.dump(model_bundle, MODEL_DIR / "best_model_bundle.joblib")

    plot_cv_comparison(comparison)
    plot_threshold_tradeoff(threshold_table, best_threshold)
    plot_confusion_matrices(default_cm, tuned_cm)
    plot_roc_and_precision_recall(
        test["Target"], test_probabilities, classes, best_threshold
    )
    plot_capacity_and_calibration(test["Target"], test_probabilities[:, dropout_index])

    tuned_metrics = test_metrics.iloc[1].to_dict()
    chosen_threshold_row = threshold_table.loc[threshold_table["selected"]].iloc[0]
    config = {
        "random_state": RANDOM_STATE,
        "cross_validation": {
            "type": "StratifiedKFold",
            "folds": CV_FOLDS,
            "shuffle": True,
        },
        "selection_metric": "dropout_f2",
        "business_constraints": {
            "target_dropout_recall": TARGET_DROPOUT_RECALL,
            "minimum_dropout_precision": MIN_DROPOUT_PRECISION,
            "top_risk_capacity_share": TOP_RISK_SHARE,
            "target_top_risk_capture": 0.60,
        },
        "best_model": best_model_name,
        "best_parameters": best_parameters,
        "dropout_alert_threshold": best_threshold,
        "threshold_selection_reason": threshold_reason,
        "candidate_parameter_grids": {
            name: json_safe(grid) for name, (_, grid) in candidate_specs().items()
        },
        "model_input_columns": model_columns,
        "audit_only_exclusion": [
            "Marital Status",
            "Nationality",
            "Displaced",
            "Educational special needs",
            "Gender",
            "Age at enrollment",
            "International",
            "nontraditional_age_flag",
        ],
        "software": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (CONFIG_DIR / "task4_model_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    summary = {
        "task": "Task 4: Model Implementation",
        "data": {
            "training_rows": int(len(train)),
            "heldout_test_rows": int(len(test)),
            "model_input_columns": len(model_columns),
            "heldout_used_once_after_selection": True,
        },
        "evaluation_design": {
            "candidate_models": comparison["model"].tolist(),
            "cross_validation_folds": CV_FOLDS,
            "selection_metric": "Dropout F2, subject to cross-validated dropout precision of at least 0.50 when feasible",
            "threshold_tuning": "Out-of-fold training probabilities only",
            "uncertainty": "1,000 bootstrap resamples on held-out test metrics",
        },
        "best_model": {
            "name": best_model_name,
            "parameters": best_parameters,
            "dropout_alert_threshold": best_threshold,
            "threshold_reason": threshold_reason,
            "oof_threshold_metrics": {
                key: float(chosen_threshold_row[key])
                for key in [
                    "dropout_precision",
                    "dropout_recall",
                    "dropout_f1",
                    "dropout_f2",
                    "macro_f1",
                    "alert_share",
                ]
            },
        },
        "heldout_test": {
            "thresholded_metrics": {
                key: float(value)
                for key, value in tuned_metrics.items()
                if key != "prediction_rule"
            },
            "top_30pct_capacity": test_capacity,
            "meets_dropout_recall_target": bool(
                tuned_metrics["dropout_recall"] >= TARGET_DROPOUT_RECALL
            ),
            "meets_dropout_precision_floor": bool(
                tuned_metrics["dropout_precision"] >= MIN_DROPOUT_PRECISION
            ),
            "meets_top_30pct_capture_target": bool(
                test_capacity["dropout_capture_rate"] >= 0.60
            ),
        },
        "bootstrap_intervals": intervals.to_dict(orient="records"),
        "interpretation": [
            "The tuned threshold converts dropout probability into an operational alert while preserving the model's three-class probabilities.",
            "Top-30% capacity metrics evaluate ranking quality independently from the alert threshold.",
            "Held-out results estimate performance for similar students from the same institutional context, not guaranteed performance at other institutions.",
            "Fairness and subgroup error analysis are deferred to Task 5 and must precede deployment.",
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return Task4Artifacts(
        summary=summary,
        comparison=comparison,
        test_metrics=test_metrics,
        threshold_table=threshold_table,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
    )


def main() -> None:
    artifacts = run_task4()
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
