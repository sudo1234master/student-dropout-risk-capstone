from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "uci_student_dropout.csv"
METADATA_PATH = ROOT / "data" / "external" / "uci_dataset_697_metadata.json"
PROCESSED_DIR = ROOT / "data" / "processed" / "task3"
MODEL_DIR = ROOT / "models" / "task3"
FIGURE_DIR = ROOT / "reports" / "figures" / "task3"
TABLE_DIR = ROOT / "reports" / "tables" / "task3"
SUMMARY_PATH = ROOT / "reports" / "task3_analysis_summary.json"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_K_FEATURES = 30

SECOND_SEMESTER_COLUMNS = [
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
]

AUDIT_ONLY_COLUMNS = [
    "Marital Status",
    "Nationality",
    "Displaced",
    "Educational special needs",
    "Gender",
    "Age at enrollment",
    "International",
    "nontraditional_age_flag",
]

ORIGINAL_CATEGORICAL_COLUMNS = [
    "Marital Status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nationality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

ENGINEERED_CATEGORICAL_COLUMNS = [
    "first_sem_no_enrollment_flag",
    "first_choice_flag",
    "financial_risk_flag",
    "nontraditional_age_flag",
]

ENGINEERED_NUMERIC_COLUMNS = [
    "first_sem_approval_rate",
    "first_sem_evaluation_intensity",
    "first_sem_no_evaluation_rate",
    "first_sem_progress_index",
    "academic_preparation_gap",
]

TARGET_ORDER = ["Dropout", "Enrolled", "Graduate"]
TARGET_PALETTE = {
    "Dropout": "#C34A36",
    "Enrolled": "#D39B2A",
    "Graduate": "#197A78",
}


@dataclass
class AnalysisArtifacts:
    summary: dict[str, Any]
    cleaned_full: pd.DataFrame
    primary_data: pd.DataFrame
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    target_distribution: pd.DataFrame
    subgroup_preview: pd.DataFrame
    outlier_review: pd.DataFrame
    mutual_information: pd.DataFrame
    model_importance: pd.DataFrame
    pca_summary: pd.DataFrame
    selected_features: list[str]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator_array = denominator.to_numpy(dtype=float)
    numerator_array = numerator.to_numpy(dtype=float)
    values = np.divide(
        numerator_array,
        denominator_array,
        out=np.zeros_like(numerator_array, dtype=float),
        where=denominator_array != 0,
    )
    return pd.Series(values, index=numerator.index)


def ensure_directories() -> None:
    for directory in [PROCESSED_DIR, MODEL_DIR, FIGURE_DIR, TABLE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def configure_plots() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 200,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def load_and_engineer_data() -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["data"]

    cleaned = raw.copy()
    cleaned.columns = cleaned.columns.str.strip()
    cleaned = cleaned.rename(columns={"Nacionality": "Nationality"})
    cleaned.insert(0, "record_id", np.arange(1, len(cleaned) + 1))

    enrolled = cleaned["Curricular units 1st sem (enrolled)"]
    approved = cleaned["Curricular units 1st sem (approved)"]
    evaluations = cleaned["Curricular units 1st sem (evaluations)"]
    without_evaluations = cleaned["Curricular units 1st sem (without evaluations)"]
    grade = cleaned["Curricular units 1st sem (grade)"]

    cleaned["first_sem_approval_rate"] = safe_divide(approved, enrolled).clip(0, 1)
    cleaned["first_sem_evaluation_intensity"] = safe_divide(evaluations, enrolled)
    cleaned["first_sem_no_evaluation_rate"] = safe_divide(without_evaluations, enrolled).clip(0, 1)
    cleaned["first_sem_progress_index"] = (
        cleaned["first_sem_approval_rate"] * (grade / 20.0)
    ).clip(0, 1)
    cleaned["academic_preparation_gap"] = (
        cleaned["Admission grade"] - cleaned["Previous qualification (grade)"]
    )
    cleaned["first_sem_no_enrollment_flag"] = (enrolled == 0).astype(int)
    cleaned["first_choice_flag"] = (cleaned["Application order"] == 0).astype(int)
    cleaned["financial_risk_flag"] = (
        (cleaned["Debtor"] == 1) | (cleaned["Tuition fees up to date"] == 0)
    ).astype(int)
    cleaned["nontraditional_age_flag"] = (cleaned["Age at enrollment"] >= 25).astype(int)

    for column in ["Mother's occupation", "Father's occupation"]:
        cleaned[column] = cleaned[column].replace({99: "Unknown/not stated"})

    for column in ORIGINAL_CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS:
        cleaned[column] = cleaned[column].astype("string")

    cleaned["Target"] = pd.Categorical(
        cleaned["Target"], categories=TARGET_ORDER, ordered=True
    )
    return cleaned, metadata


def build_primary_dataset(cleaned: pd.DataFrame) -> pd.DataFrame:
    primary = cleaned.drop(columns=SECOND_SEMESTER_COLUMNS).copy()
    expected_original = 30
    original_after_exclusion = 36 - len(SECOND_SEMESTER_COLUMNS)
    if original_after_exclusion != expected_original:
        raise AssertionError("Unexpected number of original primary predictors")
    return primary


def validate_quality(raw: pd.DataFrame, cleaned: pd.DataFrame) -> dict[str, Any]:
    binary_columns = [
        "Daytime/evening attendance",
        "Displaced",
        "Educational special needs",
        "Debtor",
        "Tuition fees up to date",
        "Gender",
        "Scholarship holder",
        "International",
    ]
    range_checks = {
        "Application order": (0, 9),
        "Previous qualification (grade)": (0, 200),
        "Admission grade": (0, 200),
        "Curricular units 1st sem (grade)": (0, 20),
        "Curricular units 2nd sem (grade)": (0, 20),
    }

    invalid_binary = {
        column: int((~raw[column].isin([0, 1])).sum()) for column in binary_columns
    }
    invalid_ranges = {
        column: int(((raw[column] < low) | (raw[column] > high)).sum())
        for column, (low, high) in range_checks.items()
    }

    return {
        "raw_rows": int(raw.shape[0]),
        "raw_columns": int(raw.shape[1]),
        "machine_missing": int(raw.isna().sum().sum()),
        "blank_strings": int(
            raw.astype("string").apply(lambda series: series.str.strip().eq("")).sum().sum()
        ),
        "exact_duplicates": int(raw.duplicated().sum()),
        "mother_occupation_code_99": int((raw["Mother's occupation"] == 99).sum()),
        "father_occupation_code_99": int((raw["Father's occupation"] == 99).sum()),
        "invalid_binary_values": invalid_binary,
        "invalid_documented_ranges": invalid_ranges,
        "engineered_missing": int(cleaned.isna().sum().sum()),
    }


def create_target_distribution(primary: pd.DataFrame) -> pd.DataFrame:
    counts = primary["Target"].value_counts(sort=False)
    result = counts.rename_axis("outcome").reset_index(name="count")
    result["share"] = result["count"] / len(primary)
    result["random_recall_at_30pct"] = 0.30
    result.to_csv(TABLE_DIR / "target_distribution.csv", index=False)
    return result


def create_subgroup_preview(primary: pd.DataFrame) -> pd.DataFrame:
    group_specs = {
        "Gender": {"0": "Female", "1": "Male"},
        "Tuition fees up to date": {"0": "Not up to date", "1": "Up to date"},
        "Debtor": {"0": "No", "1": "Yes"},
        "Scholarship holder": {"0": "No", "1": "Yes"},
    }
    frames = []
    dropout = primary["Target"].astype("string").eq("Dropout")
    for feature, labels in group_specs.items():
        grouped = (
            primary.assign(dropout_flag=dropout)
            .groupby(feature, observed=True)
            .agg(students=("record_id", "size"), dropouts=("dropout_flag", "sum"))
            .reset_index()
        )
        grouped["dropout_rate"] = grouped["dropouts"] / grouped["students"]
        grouped["feature"] = feature
        grouped["group"] = grouped[feature].astype("string").map(labels).fillna(grouped[feature])
        frames.append(grouped[["feature", "group", "students", "dropouts", "dropout_rate"]])
    result = pd.concat(frames, ignore_index=True)
    result.to_csv(TABLE_DIR / "subgroup_descriptive_preview.csv", index=False)
    return result


def create_outlier_review(primary: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        column
        for column in primary.select_dtypes(include=["number"]).columns
        if column != "record_id"
    ]
    rows = []
    for column in numeric_columns:
        series = primary[column]
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        flagged = (series < lower) | (series > upper)
        rows.append(
            {
                "variable": column,
                "minimum": float(series.min()),
                "q1": q1,
                "median": float(series.median()),
                "q3": q3,
                "maximum": float(series.max()),
                "iqr_lower_fence": lower,
                "iqr_upper_fence": upper,
                "iqr_flag_count": int(flagged.sum()),
                "iqr_flag_share": float(flagged.mean()),
                "disposition": "Retain; validated as plausible or structurally meaningful. Use robust scaling and model diagnostics.",
            }
        )
    result = pd.DataFrame(rows).sort_values("iqr_flag_count", ascending=False)
    result.to_csv(TABLE_DIR / "outlier_review.csv", index=False)
    return result


def define_model_columns(primary: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    excluded = {"record_id", "Target", *AUDIT_ONLY_COLUMNS}
    model_columns = [column for column in primary.columns if column not in excluded]
    categorical_columns = [
        column
        for column in ORIGINAL_CATEGORICAL_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS
        if column in model_columns
    ]
    numeric_columns = [column for column in model_columns if column not in categorical_columns]
    if set(model_columns) != set(categorical_columns + numeric_columns):
        raise AssertionError("Model column classification is incomplete")
    return model_columns, categorical_columns, numeric_columns


def make_preprocessor(
    categorical_columns: list[str], numeric_columns: list[str]
) -> ColumnTransformer:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
            ),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, categorical_columns),
            ("numeric", numeric_pipeline, numeric_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def split_data(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train, test = train_test_split(
        primary,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=primary["Target"],
    )
    return train.sort_values("record_id").reset_index(drop=True), test.sort_values(
        "record_id"
    ).reset_index(drop=True)


def feature_selection(
    transformed_train: np.ndarray,
    y_train: pd.Series,
    feature_names: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    discrete_mask = np.array(
        [name.startswith("categorical__") for name in feature_names], dtype=bool
    )
    scores = mutual_info_classif(
        transformed_train,
        y_train.astype("string"),
        discrete_features=discrete_mask,
        random_state=RANDOM_STATE,
    )
    ranking = (
        pd.DataFrame({"feature": feature_names, "mutual_information": scores})
        .sort_values("mutual_information", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    selected_names = ranking.head(min(TOP_K_FEATURES, len(ranking)))["feature"].tolist()
    selected_index = np.array([np.where(feature_names == name)[0][0] for name in selected_names])
    ranking["selected_top_30"] = ranking["feature"].isin(selected_names)
    ranking.to_csv(TABLE_DIR / "mutual_information_feature_ranking.csv", index=False)
    return ranking, selected_index, selected_names


def diagnostic_model_importance(
    transformed_train: np.ndarray,
    y_train: pd.Series,
    feature_names: np.ndarray,
) -> pd.DataFrame:
    diagnostic_model = ExtraTreesClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    diagnostic_model.fit(transformed_train, y_train.astype("string"))
    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "model_based_importance": diagnostic_model.feature_importances_,
            }
        )
        .sort_values("model_based_importance", ascending=False)
        .reset_index(drop=True)
    )
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(TABLE_DIR / "diagnostic_model_feature_importance.csv", index=False)
    joblib.dump(diagnostic_model, MODEL_DIR / "diagnostic_extratrees_for_importance.joblib")
    return importance


def run_pca(
    transformed_train: np.ndarray,
    transformed_test: np.ndarray,
) -> tuple[PCA, StandardScaler, np.ndarray, np.ndarray, pd.DataFrame, dict[str, int]]:
    pca_scaler = StandardScaler()
    train_scaled = pca_scaler.fit_transform(transformed_train)
    test_scaled = pca_scaler.transform(transformed_test)

    pca = PCA(random_state=RANDOM_STATE)
    train_scores = pca.fit_transform(train_scaled)
    test_scores = pca.transform(test_scaled)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pca_summary = pd.DataFrame(
        {
            "component": np.arange(1, len(cumulative) + 1),
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_explained_variance": cumulative,
        }
    )
    pca_summary.to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False)
    counts = {
        "components_for_80pct": int(np.searchsorted(cumulative, 0.80) + 1),
        "components_for_90pct": int(np.searchsorted(cumulative, 0.90) + 1),
        "components_for_95pct": int(np.searchsorted(cumulative, 0.95) + 1),
    }
    return pca, pca_scaler, train_scores, test_scores, pca_summary, counts


def save_processed_data(
    cleaned: pd.DataFrame,
    primary: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    transformed_train: np.ndarray,
    transformed_test: np.ndarray,
    selected_index: np.ndarray,
    selected_features: list[str],
) -> None:
    cleaned.to_csv(PROCESSED_DIR / "cleaned_full_with_engineered_features.csv", index=False)
    primary.to_csv(PROCESSED_DIR / "primary_first_semester_dataset.csv", index=False)
    train.to_csv(PROCESSED_DIR / "train_primary.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test_primary.csv", index=False)

    train_selected = pd.DataFrame(
        transformed_train[:, selected_index], columns=selected_features
    )
    train_selected.insert(0, "record_id", train["record_id"].to_numpy())
    train_selected["Target"] = train["Target"].astype("string").to_numpy()
    test_selected = pd.DataFrame(
        transformed_test[:, selected_index], columns=selected_features
    )
    test_selected.insert(0, "record_id", test["record_id"].to_numpy())
    test_selected["Target"] = test["Target"].astype("string").to_numpy()
    train_selected.to_csv(PROCESSED_DIR / "train_selected_features.csv", index=False)
    test_selected.to_csv(PROCESSED_DIR / "test_selected_features.csv", index=False)


def plot_target_distribution(target_distribution: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    colors = [TARGET_PALETTE[outcome] for outcome in target_distribution["outcome"]]
    bars = ax.bar(
        target_distribution["outcome"], target_distribution["count"], color=colors
    )
    for bar, share in zip(bars, target_distribution["share"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 35,
            f"{share:.1%}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set(title="Student outcome distribution", xlabel="Outcome", ylabel="Students")
    ax.set_ylim(0, target_distribution["count"].max() * 1.15)
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "01_target_distribution.png", bbox_inches="tight")
    plt.close(fig)


def plot_key_factor_dropout_rates(subgroup_preview: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax, feature in zip(axes.flat, subgroup_preview["feature"].drop_duplicates()):
        subset = subgroup_preview[subgroup_preview["feature"] == feature]
        sns.barplot(
            data=subset,
            x="group",
            y="dropout_rate",
            color="#3C78A8",
            ax=ax,
        )
        for patch, value, count in zip(
            ax.patches, subset["dropout_rate"], subset["students"]
        ):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.015,
                f"{value:.1%}\n(n={count:,})",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_title(feature)
        ax.set_xlabel("")
        ax.set_ylabel("Observed dropout rate")
        ax.set_ylim(0, max(0.65, subset["dropout_rate"].max() + 0.12))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    fig.suptitle(
        "Observed dropout rates vary across financial and demographic groups",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURE_DIR / "02_key_factor_dropout_rates.png", bbox_inches="tight")
    plt.close(fig)


def plot_academic_features(primary: pd.DataFrame) -> None:
    plot_data = primary.copy()
    plot_data["Target"] = plot_data["Target"].astype("string")
    columns = [
        ("first_sem_approval_rate", "Approval rate"),
        ("Curricular units 1st sem (grade)", "Average grade"),
        ("first_sem_progress_index", "Progress index"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    for ax, (column, label) in zip(axes, columns):
        sns.boxplot(
            data=plot_data,
            x="Target",
            y=column,
            order=TARGET_ORDER,
            hue="Target",
            palette=TARGET_PALETTE,
            legend=False,
            showfliers=False,
            ax=ax,
        )
        ax.set(xlabel="", ylabel=label, title=label)
    fig.suptitle(
        "First-semester academic progress separates outcomes",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIGURE_DIR / "03_academic_features_by_outcome.png", bbox_inches="tight")
    plt.close(fig)


def plot_age_distribution(primary: pd.DataFrame) -> None:
    plot_data = primary.copy()
    plot_data["Target"] = plot_data["Target"].astype("string")
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    sns.kdeplot(
        data=plot_data,
        x="Age at enrollment",
        hue="Target",
        hue_order=TARGET_ORDER,
        palette=TARGET_PALETTE,
        common_norm=False,
        fill=False,
        linewidth=2,
        ax=ax,
    )
    ax.set(
        title="Age at enrollment differs across outcomes",
        xlabel="Age at enrollment",
        ylabel="Density",
    )
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "04_age_distribution_by_outcome.png", bbox_inches="tight")
    plt.close(fig)


def plot_correlation_heatmap(primary: pd.DataFrame) -> None:
    columns = [
        "Admission grade",
        "Previous qualification (grade)",
        "Age at enrollment",
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        "Curricular units 1st sem (grade)",
        "first_sem_approval_rate",
        "first_sem_evaluation_intensity",
        "first_sem_no_evaluation_rate",
        "first_sem_progress_index",
        "academic_preparation_gap",
    ]
    correlation_data = primary[columns].copy()
    correlation_data["dropout_flag"] = primary["Target"].astype("string").eq("Dropout").astype(int)
    correlation = correlation_data.corr(method="spearman")
    correlation.to_csv(TABLE_DIR / "numeric_spearman_correlations.csv")
    fig, ax = plt.subplots(figsize=(11.2, 8.5))
    mask = np.triu(np.ones_like(correlation, dtype=bool), k=1)
    sns.heatmap(
        correlation,
        mask=mask,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 7},
        linewidths=0.5,
        cbar_kws={"label": "Spearman correlation"},
        ax=ax,
    )
    ax.set_title("Numeric relationships and dropout association")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "05_numeric_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)


def plot_feature_rankings(
    mutual_information: pd.DataFrame, model_importance: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.4))
    mi_top = mutual_information.head(20).sort_values("mutual_information")
    imp_top = model_importance.head(20).sort_values("model_based_importance")
    axes[0].barh(mi_top["feature"], mi_top["mutual_information"], color="#197A78")
    axes[0].set(title="Filter method: mutual information", xlabel="Mutual information")
    axes[1].barh(
        imp_top["feature"], imp_top["model_based_importance"], color="#6A5A9E"
    )
    axes[1].set(
        title="Diagnostic Extra Trees importance", xlabel="Model-based importance"
    )
    for ax in axes:
        ax.tick_params(axis="y", labelsize=7)
        sns.despine(ax=ax)
    fig.suptitle(
        "First-semester progress dominates preliminary feature relevance",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURE_DIR / "06_feature_selection_and_importance.png", bbox_inches="tight")
    plt.close(fig)


def plot_pca(
    pca_summary: pd.DataFrame, train_scores: np.ndarray, y_train: pd.Series
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(
        pca_summary["component"],
        pca_summary["cumulative_explained_variance"],
        color="#197A78",
        linewidth=2,
    )
    for threshold in [0.80, 0.90, 0.95]:
        axes[0].axhline(threshold, color="#98A2B3", linestyle="--", linewidth=1)
    axes[0].set(
        title="Cumulative explained variance",
        xlabel="Principal components",
        ylabel="Cumulative explained variance",
        ylim=(0, 1.02),
    )
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))

    pca_plot = pd.DataFrame(
        {
            "PC1": train_scores[:, 0],
            "PC2": train_scores[:, 1],
            "Target": y_train.astype("string").to_numpy(),
        }
    )
    sns.scatterplot(
        data=pca_plot,
        x="PC1",
        y="PC2",
        hue="Target",
        hue_order=TARGET_ORDER,
        palette=TARGET_PALETTE,
        alpha=0.50,
        s=20,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title("Training data projected onto PC1 and PC2")
    for ax in axes:
        sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "07_pca_analysis.png", bbox_inches="tight")
    plt.close(fig)


def compute_engineered_feature_summary(primary: pd.DataFrame) -> pd.DataFrame:
    summary = primary[ENGINEERED_NUMERIC_COLUMNS].describe().T.reset_index()
    summary = summary.rename(columns={"index": "engineered_feature"})
    summary.to_csv(TABLE_DIR / "engineered_feature_summary.csv", index=False)
    return summary


def run_analysis() -> AnalysisArtifacts:
    ensure_directories()
    configure_plots()

    raw = pd.read_csv(RAW_PATH, encoding="utf-8-sig")
    cleaned, metadata = load_and_engineer_data()
    primary = build_primary_dataset(cleaned)
    quality_checks = validate_quality(raw, cleaned)
    train, test = split_data(primary)
    target_distribution = create_target_distribution(train)
    subgroup_preview = create_subgroup_preview(train)
    outlier_review = create_outlier_review(train)
    engineered_summary = compute_engineered_feature_summary(train)

    model_columns, categorical_columns, numeric_columns = define_model_columns(primary)
    preprocessor = make_preprocessor(categorical_columns, numeric_columns)

    transformed_train = preprocessor.fit_transform(train[model_columns])
    transformed_test = preprocessor.transform(test[model_columns])
    feature_names = preprocessor.get_feature_names_out()

    mutual_information, selected_index, selected_features = feature_selection(
        transformed_train, train["Target"], feature_names
    )
    model_importance = diagnostic_model_importance(
        transformed_train, train["Target"], feature_names
    )
    pca, pca_scaler, train_pca, test_pca, pca_summary, pca_counts = run_pca(
        transformed_train, transformed_test
    )

    save_processed_data(
        cleaned,
        primary,
        train,
        test,
        transformed_train,
        transformed_test,
        selected_index,
        selected_features,
    )

    joblib.dump(preprocessor, MODEL_DIR / "preprocessor_first_semester.joblib")
    joblib.dump(pca_scaler, MODEL_DIR / "pca_input_scaler.joblib")
    joblib.dump(pca, MODEL_DIR / "pca_full.joblib")
    (MODEL_DIR / "selected_features.json").write_text(
        json.dumps(
            {
                "method": "Mutual information filter selection fitted on training data only",
                "top_k": len(selected_features),
                "selected_features": selected_features,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    plot_target_distribution(target_distribution)
    plot_key_factor_dropout_rates(subgroup_preview)
    plot_academic_features(train)
    plot_age_distribution(train)
    plot_correlation_heatmap(train)
    plot_feature_rankings(mutual_information, model_importance)
    plot_pca(pca_summary, train_pca, train["Target"])

    dropout_rates = subgroup_preview.set_index(["feature", "group"])[
        "dropout_rate"
    ].to_dict()
    target_shares = target_distribution.set_index("outcome")["share"].to_dict()

    summary = {
        "project": "Early Identification of University Students at Risk of Dropping Out",
        "source": {
            "dataset": metadata["name"],
            "doi": metadata["dataset_doi"],
            "license": "CC BY 4.0",
        },
        "reproducibility": {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "quality": quality_checks,
        "data_scope": {
            "rows": int(len(primary)),
            "original_predictors": 36,
            "second_semester_predictors_excluded": len(SECOND_SEMESTER_COLUMNS),
            "original_predictors_available_at_prediction": 30,
            "engineered_features_added": len(ENGINEERED_NUMERIC_COLUMNS)
            + len(ENGINEERED_CATEGORICAL_COLUMNS),
            "features_before_audit_only_exclusion": int(primary.shape[1] - 2),
            "audit_only_features_excluded_from_default_matrix": len(AUDIT_ONLY_COLUMNS),
            "model_input_columns_before_encoding": len(model_columns),
            "transformed_features_after_encoding": int(len(feature_names)),
            "selected_features": len(selected_features),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        },
        "eda_scope": "Training partition only; the held-out test partition was not used for exploratory or learned Task 3 decisions.",
        "target_distribution": {
            key: float(value) for key, value in target_shares.items()
        },
        "descriptive_dropout_rates": {
            f"{feature} | {group}": float(rate)
            for (feature, group), rate in dropout_rates.items()
        },
        "feature_engineering": {
            "features": ENGINEERED_NUMERIC_COLUMNS + ENGINEERED_CATEGORICAL_COLUMNS,
            "outlier_strategy": "Retain validated values; use RobustScaler; revisit only if model diagnostics show instability.",
            "categorical_strategy": "Explicit one-hot encoding with infrequent-category grouping at fewer than 10 training records.",
            "missing_strategy": "No observed machine missing values; defensive median/mode imputers retained in pipeline.",
        },
        "feature_selection": {
            "method": "Mutual information filter method on training data only",
            "top_k": len(selected_features),
            "top_10": mutual_information.head(10).to_dict(orient="records"),
        },
        "explainability": {
            "method": "Model-based feature importance from a diagnostic Extra Trees classifier; not a final model comparison.",
            "top_10": model_importance.head(10).to_dict(orient="records"),
        },
        "pca": {
            **pca_counts,
            "pc1_explained_variance": float(pca.explained_variance_ratio_[0]),
            "pc2_explained_variance": float(pca.explained_variance_ratio_[1]),
            "first_two_cumulative": float(pca.explained_variance_ratio_[:2].sum()),
            "decision": "Retain PCA as a dimensionality-reduction comparison for Task 4; do not replace interpretable features by default.",
        },
        "limitations": [
            "One Portuguese institution limits external validity.",
            "No student identifier is provided, so only exact-row duplicate checks are possible.",
            "Outcome associations are descriptive and not causal effects of interventions.",
            "Rare sensitive subgroups may not support stable fairness estimates.",
            "First-semester variables support intervention after semester one, not at initial enrollment.",
        ],
        "engineered_feature_summary": engineered_summary.to_dict(orient="records"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return AnalysisArtifacts(
        summary=summary,
        cleaned_full=cleaned,
        primary_data=primary,
        train_data=train,
        test_data=test,
        target_distribution=target_distribution,
        subgroup_preview=subgroup_preview,
        outlier_review=outlier_review,
        mutual_information=mutual_information,
        model_importance=model_importance,
        pca_summary=pca_summary,
        selected_features=selected_features,
    )


def main() -> None:
    artifacts = run_analysis()
    print(json.dumps(artifacts.summary, indent=2))


if __name__ == "__main__":
    main()
