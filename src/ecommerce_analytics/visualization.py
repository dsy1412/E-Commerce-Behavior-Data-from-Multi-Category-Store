from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import ProjectConfig
from .modeling import ModelingResult


COLORS = ["#2563eb", "#0f766e", "#d97706", "#be123c", "#6d28d9"]


def _finish(path, title: str | None = None) -> None:
    if title:
        plt.title(title, loc="left", fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def create_figures(
    daily: pd.DataFrame,
    funnel: pd.DataFrame,
    user_segments: pd.DataFrame,
    segment_funnels: pd.DataFrame,
    retention: pd.DataFrame,
    modeling: ModelingResult,
    config: ProjectConfig,
) -> list[str]:
    sns.set_theme(style="whitegrid", context="notebook")
    created: list[str] = []

    plt.figure(figsize=(9, 4.8))
    for column, color in zip(("view", "cart", "purchase"), COLORS):
        plt.plot(daily["event_date"], daily[column], marker="o", label=column, color=color)
    plt.xlabel("")
    plt.yscale("log")
    plt.ylabel("Events (log scale)")
    plt.legend(frameon=False, ncol=3)
    path = config.figures_dir / "daily_event_trends.png"
    _finish(path, "Daily commerce activity")
    created.append(path.name)

    plt.figure(figsize=(9, 4.8))
    plt.plot(
        daily["event_date"],
        daily["daily_user_purchase_rate"],
        marker="o",
        color=COLORS[1],
    )
    plt.gca().yaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
    plt.xlabel("")
    plt.ylabel("Users with a purchase event / DAU")
    path = config.figures_dir / "daily_conversion.png"
    _finish(path, "Observed daily user purchase rate")
    created.append(path.name)

    funnel_values = funnel.set_index("metric")["value"]
    funnel_plot = pd.Series(
        {
            "View": funnel_values.get("view_sessions", 0),
            "Cart after view": funnel_values.get("cart_after_view_sessions", 0),
            "Purchase after view": funnel_values.get("purchase_after_view_sessions", 0),
        }
    )
    plt.figure(figsize=(7.5, 4.8))
    axis = funnel_plot.plot(kind="bar", color=COLORS[:3])
    axis.bar_label(axis.containers[0], fmt="{:,.0f}")
    plt.xticks(rotation=0)
    plt.ylabel("Sessions")
    path = config.figures_dir / "session_funnel.png"
    _finish(path, "Observed session funnel")
    created.append(path.name)

    segment_counts = user_segments["behavior_segment"].value_counts().sort_values()
    plt.figure(figsize=(7.5, 4.8))
    axis = segment_counts.plot(kind="barh", color=COLORS[0])
    axis.bar_label(axis.containers[0], fmt="{:,.0f}")
    plt.xlabel("Users")
    plt.ylabel("")
    path = config.figures_dir / "user_segments.png"
    _finish(path, "Behavior-based user segments")
    created.append(path.name)

    category = segment_funnels.loc[
        segment_funnels["dimension"].eq("entry_category")
        & segment_funnels["segment"].ne("unknown")
        & segment_funnels["sessions"].ge(100)
    ].nlargest(10, "sessions")
    if not category.empty:
        category = category.sort_values("view_to_purchase_rate")
        plt.figure(figsize=(8.5, 5.5))
        plt.barh(category["segment"], category["view_to_purchase_rate"], color=COLORS[2])
        plt.gca().xaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
        plt.xlabel("View-to-purchase rate")
        plt.ylabel("")
        path = config.figures_dir / "category_conversion.png"
        _finish(path, "Conversion among high-volume entry categories")
        created.append(path.name)

    if not retention.empty:
        weighted = (
            retention.groupby("horizon_days", observed=True)
            .apply(
                lambda frame: pd.Series(
                    {
                        "retention_rate": frame["retained_users"].sum()
                        / frame["eligible_users"].sum()
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        plt.figure(figsize=(6.8, 4.5))
        plt.bar(weighted["horizon_days"].astype(str), weighted["retention_rate"], color=COLORS[4])
        plt.gca().yaxis.set_major_formatter(lambda value, _: f"{value:.1%}")
        plt.xlabel("Days after first observed activity")
        plt.ylabel("Retention")
        path = config.figures_dir / "cohort_retention.png"
        _finish(path, "Observed cohort retention")
        created.append(path.name)

    model_plot = modeling.metrics.sort_values("average_precision")
    plt.figure(figsize=(7.5, 4.5))
    plt.barh(model_plot["model"], model_plot["average_precision"], color=COLORS[1])
    plt.xlabel("Average precision")
    plt.ylabel("")
    path = config.figures_dir / "model_comparison.png"
    _finish(path, "Time-based validation performance")
    created.append(path.name)

    plt.figure(figsize=(5.5, 5.2))
    plt.plot(
        modeling.calibration["mean_predicted_probability"],
        modeling.calibration["observed_conversion_rate"],
        marker="o",
        color=COLORS[0],
        label=modeling.best_model_name,
    )
    plt.plot([0, 1], [0, 1], linestyle="--", color="#64748b", label="perfect calibration")
    calibration_limit = min(
        1.0,
        max(
            0.12,
            float(modeling.calibration["mean_predicted_probability"].max()) * 1.2,
            float(modeling.calibration["observed_conversion_rate"].max()) * 1.2,
        ),
    )
    plt.xlim(0, calibration_limit)
    plt.ylim(0, calibration_limit)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed conversion rate")
    plt.legend(frameon=False)
    path = config.figures_dir / "calibration.png"
    _finish(path, "Probability calibration")
    created.append(path.name)

    shap_plot = modeling.shap_importance.head(15).sort_values("mean_abs_shap")
    plt.figure(figsize=(8.5, 6))
    plt.barh(shap_plot["feature"].str.replace(r"^[^_]+__", "", regex=True), shap_plot["mean_abs_shap"], color=COLORS[3])
    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("")
    path = config.figures_dir / "shap_importance.png"
    _finish(path, "Leading LightGBM conversion drivers")
    created.append(path.name)
    return created
