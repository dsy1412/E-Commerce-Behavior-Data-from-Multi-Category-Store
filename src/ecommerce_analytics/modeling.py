from __future__ import annotations

import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import sparse
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ProjectConfig


TARGET = "purchase_after_window"
NUMERIC_FEATURES = [
    "observed_events",
    "view_count",
    "cart_count",
    "remove_count",
    "unique_products",
    "unique_categories",
    "unique_brands",
    "mean_price",
    "max_price",
    "min_price",
    "price_range",
    "session_hour",
    "day_of_week",
    "is_weekend",
    "cart_per_view",
    "remove_per_cart",
    "prior_sessions",
    "prior_purchase_sessions",
    "prior_conversion_rate",
]
CATEGORICAL_FEATURES = [
    "entry_category",
    "entry_brand",
    "entry_price_band",
    "hour_band",
    "user_activity_tier",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class ModelingResult:
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    segment_metrics: pd.DataFrame
    calibration: pd.DataFrame
    shap_importance: pd.DataFrame
    split_summary: dict[str, object]
    best_model_name: str


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                    sparse_output=True,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def _candidate_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "historical_rate": Pipeline(
            [("preprocess", _preprocessor()), ("model", DummyClassifier(strategy="prior"))]
        ),
        "logistic_regression": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=500,
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "lightgbm": Pipeline(
            [
                ("preprocess", _preprocessor()),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=350,
                        learning_rate=0.05,
                        num_leaves=31,
                        min_child_samples=50,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        random_state=random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
    }


def _safe_auc(y_true: pd.Series, probabilities: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probabilities)) if y_true.nunique() > 1 else np.nan


def _choose_threshold(y_true: pd.Series, probabilities: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.nanargmax(scores))])


def _top_fraction_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    fraction: float = 0.10,
) -> dict[str, float | int]:
    labels = y_true.to_numpy()
    top_n = max(1, int(np.ceil(len(labels) * fraction)))
    top_indices = np.argsort(-probabilities, kind="stable")[:top_n]
    top_positives = int(labels[top_indices].sum())
    total_positives = int(labels.sum())
    prevalence = float(labels.mean())
    if np.unique(probabilities).size == 1:
        return {
            "validation_positive_rate": prevalence,
            "top_10pct_sessions": top_n,
            "precision_at_10pct": prevalence,
            "recall_at_10pct": top_n / len(labels),
            "lift_at_10pct": 1.0,
        }
    precision_at_fraction = top_positives / top_n
    return {
        "validation_positive_rate": prevalence,
        "top_10pct_sessions": top_n,
        "precision_at_10pct": precision_at_fraction,
        "recall_at_10pct": top_positives / max(total_positives, 1),
        "lift_at_10pct": precision_at_fraction / max(prevalence, 1e-12),
    }


def _bootstrap_intervals(
    y_true: pd.Series,
    probabilities: np.ndarray,
    random_state: int,
    iterations: int = 300,
) -> dict[str, float]:
    labels = y_true.to_numpy()
    rng = np.random.default_rng(random_state)
    roc_values: list[float] = []
    ap_values: list[float] = []
    for _ in range(iterations):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        sampled_probabilities = probabilities[indices]
        roc_values.append(float(roc_auc_score(sampled_labels, sampled_probabilities)))
        ap_values.append(float(average_precision_score(sampled_labels, sampled_probabilities)))
    if not roc_values:
        return {
            "roc_auc_ci_low": np.nan,
            "roc_auc_ci_high": np.nan,
            "average_precision_ci_low": np.nan,
            "average_precision_ci_high": np.nan,
        }
    return {
        "roc_auc_ci_low": float(np.quantile(roc_values, 0.025)),
        "roc_auc_ci_high": float(np.quantile(roc_values, 0.975)),
        "average_precision_ci_low": float(np.quantile(ap_values, 0.025)),
        "average_precision_ci_high": float(np.quantile(ap_values, 0.975)),
    }


def _evaluate(
    name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    random_state: int,
) -> dict[str, float | int | str]:
    predictions = probabilities >= threshold
    result: dict[str, float | int | str] = {
        "model": name,
        "roc_auc": _safe_auc(y_true, probabilities),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "threshold": threshold,
    }
    result.update(_top_fraction_metrics(y_true, probabilities))
    result.update(_bootstrap_intervals(y_true, probabilities, random_state))
    return result


def _calibration_table(y_true: pd.Series, probabilities: np.ndarray, bins: int = 10) -> pd.DataFrame:
    frame = pd.DataFrame({"y_true": y_true.to_numpy(), "probability": probabilities})
    frame["probability_bin"] = pd.qcut(
        frame["probability"].rank(method="first"), q=min(bins, len(frame)), duplicates="drop"
    )
    return (
        frame.groupby("probability_bin", observed=True)
        .agg(
            observations=("y_true", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_conversion_rate=("y_true", "mean"),
        )
        .reset_index(drop=True)
    )


def _segment_metrics(predictions: pd.DataFrame, threshold: float) -> pd.DataFrame:
    outputs: list[pd.DataFrame] = []
    for dimension in (
        "user_activity_tier",
        "entry_category",
        "entry_brand",
        "entry_price_band",
        "hour_band",
    ):
        rows: list[dict[str, object]] = []
        for segment, group in predictions.groupby(dimension, dropna=False, observed=True):
            if len(group) < 30:
                continue
            predicted = group["y_probability"].ge(threshold)
            rows.append(
                {
                    "dimension": dimension,
                    "segment": str(segment),
                    "sessions": int(len(group)),
                    "conversion_rate": float(group["y_true"].mean()),
                    "mean_predicted_probability": float(group["y_probability"].mean()),
                    "roc_auc": _safe_auc(group["y_true"], group["y_probability"]),
                    "precision": float(precision_score(group["y_true"], predicted, zero_division=0)),
                    "recall": float(recall_score(group["y_true"], predicted, zero_division=0)),
                    "f1": float(f1_score(group["y_true"], predicted, zero_division=0)),
                }
            )
        if rows:
            outputs.append(pd.DataFrame(rows))
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _lightgbm_shap_importance(
    pipeline: Pipeline,
    X: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    sample = X.sample(n=min(2_000, len(X)), random_state=random_state)
    preprocessor = pipeline.named_steps["preprocess"]
    transformed = preprocessor.transform(sample)
    feature_names = preprocessor.get_feature_names_out()
    contributions = pipeline.named_steps["model"].booster_.predict(
        transformed, pred_contrib=True
    )
    if sparse.issparse(contributions):
        contributions = contributions.toarray()
    values = np.asarray(contributions)[:, :-1]
    return (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": np.abs(values).mean(axis=0),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def train_conversion_models(
    session_features: pd.DataFrame,
    config: ProjectConfig,
) -> ModelingResult:
    frame = session_features.sort_values("session_start").reset_index(drop=True)
    if len(frame) < 200 or frame[TARGET].nunique() < 2:
        raise ValueError("Conversion modeling requires at least 200 sessions and both target classes.")
    split_index = int(len(frame) * (1 - config.validation_fraction))
    train = frame.iloc[:split_index].copy()
    validation = frame.iloc[split_index:].copy()
    internal_index = max(1, int(len(train) * 0.8))
    train_core = train.iloc[:internal_index]
    threshold_set = train.iloc[internal_index:]

    results: list[dict[str, float | str]] = []
    fitted_models: dict[str, Pipeline] = {}
    thresholds: dict[str, float] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    for name, candidate in _candidate_models(config.random_state).items():
        threshold_model = clone(candidate)
        threshold_model.fit(train_core[MODEL_FEATURES], train_core[TARGET])
        threshold_probabilities = threshold_model.predict_proba(threshold_set[MODEL_FEATURES])[:, 1]
        threshold = _choose_threshold(threshold_set[TARGET], threshold_probabilities)

        fitted = clone(candidate)
        fitted.fit(train[MODEL_FEATURES], train[TARGET])
        probabilities = fitted.predict_proba(validation[MODEL_FEATURES])[:, 1]
        results.append(
            _evaluate(
                name,
                validation[TARGET],
                probabilities,
                threshold,
                config.random_state,
            )
        )
        fitted_models[name] = fitted
        thresholds[name] = threshold
        validation_probabilities[name] = probabilities

    metrics = pd.DataFrame(results).sort_values(
        ["average_precision", "roc_auc"], ascending=False
    ).reset_index(drop=True)
    best_model_name = str(metrics.iloc[0]["model"])
    best_model = fitted_models[best_model_name]
    best_threshold = thresholds[best_model_name]
    best_probabilities = validation_probabilities[best_model_name]

    predictions = validation[
        [
            "session_start",
            "user_activity_tier",
            "entry_category",
            "entry_brand",
            "entry_price_band",
            "hour_band",
            TARGET,
        ]
    ].copy()
    predictions = predictions.rename(columns={TARGET: "y_true"})
    predictions["y_probability"] = best_probabilities
    predictions["y_predicted"] = predictions["y_probability"].ge(best_threshold).astype(int)

    calibration = _calibration_table(predictions["y_true"], predictions["y_probability"])
    segment_metrics = _segment_metrics(predictions, best_threshold)
    shap_importance = _lightgbm_shap_importance(
        fitted_models["lightgbm"], validation[MODEL_FEATURES], config.random_state
    )
    split_summary = {
        "training_sessions": int(len(train)),
        "validation_sessions": int(len(validation)),
        "training_start": str(train["session_start"].min()),
        "training_end": str(train["session_start"].max()),
        "validation_start": str(validation["session_start"].min()),
        "validation_end": str(validation["session_start"].max()),
        "training_conversion_rate": float(train[TARGET].mean()),
        "validation_conversion_rate": float(validation[TARGET].mean()),
        "observation_minutes": config.observation_minutes,
        "best_model": best_model_name,
        "best_threshold": best_threshold,
    }

    metrics.to_csv(config.tables_dir / "model_metrics.csv", index=False)
    predictions.to_parquet(config.outputs_dir / "validation_predictions.parquet", index=False)
    segment_metrics.to_csv(config.tables_dir / "model_segment_metrics.csv", index=False)
    calibration.to_csv(config.tables_dir / "calibration.csv", index=False)
    shap_importance.to_csv(config.tables_dir / "shap_importance.csv", index=False)
    (config.outputs_dir / "modeling_summary.json").write_text(
        json.dumps(split_summary, indent=2), encoding="utf-8"
    )
    joblib.dump(best_model, config.outputs_dir / "best_conversion_model.joblib")
    return ModelingResult(
        metrics=metrics,
        predictions=predictions,
        segment_metrics=segment_metrics,
        calibration=calibration,
        shap_importance=shap_importance,
        split_summary=split_summary,
        best_model_name=best_model_name,
    )
