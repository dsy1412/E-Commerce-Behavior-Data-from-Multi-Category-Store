from __future__ import annotations

from typing import Any

import pandas as pd

from .config import ProjectConfig
from .modeling import ModelingResult


def _funnel_value(funnel: pd.DataFrame, metric: str) -> float:
    values = funnel.loc[funnel["metric"].eq(metric), "value"]
    return float(values.iloc[0]) if len(values) else 0.0


def _weighted_retention(retention: pd.DataFrame, horizon: int) -> float | None:
    subset = retention.loc[retention["horizon_days"].eq(horizon)]
    if subset.empty or subset["eligible_users"].sum() == 0:
        return None
    return float(subset["retained_users"].sum() / subset["eligible_users"].sum())


def _category_signal(segment_funnels: pd.DataFrame, highest: bool) -> pd.Series | None:
    subset = segment_funnels.loc[
        segment_funnels["dimension"].eq("entry_category")
        & segment_funnels["segment"].ne("unknown")
        & segment_funnels["sessions"].ge(100)
    ]
    if subset.empty:
        return None
    index = subset["view_to_purchase_rate"].idxmax() if highest else subset["view_to_purchase_rate"].idxmin()
    return subset.loc[index]


def generate_product_report(
    manifest: dict[str, Any],
    quality: dict[str, Any],
    daily: pd.DataFrame,
    funnel: pd.DataFrame,
    user_segments: pd.DataFrame,
    segment_funnels: pd.DataFrame,
    retention: pd.DataFrame,
    root_causes: pd.DataFrame,
    modeling: ModelingResult,
    experiment: dict[str, float | int],
    config: ProjectConfig,
) -> None:
    best = modeling.metrics.iloc[0]
    baseline_ap = float(
        modeling.metrics.loc[
            modeling.metrics["model"].eq("historical_rate"), "average_precision"
        ].iloc[0]
    )
    ap_multiple = float(best["average_precision"] / max(baseline_ap, 1e-12))
    high_category = _category_signal(segment_funnels, highest=True)
    low_category = _category_signal(segment_funnels, highest=False)
    d1 = _weighted_retention(retention, 1)
    d7 = _weighted_retention(retention, 7)
    repeat_share = float(user_segments["behavior_segment"].eq("repeat purchaser").mean())
    top_driver = modeling.shap_importance.iloc[0]["feature"]
    drops = root_causes.loc[root_causes["period_type"].eq("drop")]
    largest_drop = drops.iloc[0] if not drops.empty else None
    daily_changes = daily.sort_values("event_date").copy()
    daily_changes["cart_change"] = daily_changes["cart"].pct_change()
    daily_changes["view_change"] = daily_changes["view"].pct_change()
    largest_cart_shift = daily_changes.loc[
        daily_changes["cart_change"].replace([float("inf"), float("-inf")], pd.NA).abs().idxmax()
    ]

    category_text = "Insufficient volume for a stable category comparison."
    if high_category is not None and low_category is not None:
        category_text = (
            f"Among entry categories with at least 100 sampled sessions, **{high_category['segment']}** "
            f"had the highest observed view-to-purchase rate ({high_category['view_to_purchase_rate']:.2%}), "
            f"while **{low_category['segment']}** had the lowest ({low_category['view_to_purchase_rate']:.2%}). "
            "These are descriptive signals for follow-up, not causal category effects."
        )
    retention_text = (
        f"D1 retention was **{d1:.2%}**" if d1 is not None else "D1 retention was not estimable"
    )
    retention_text += (
        f" and D7 retention was **{d7:.2%}** among eligible observed cohorts."
        if d7 is not None
        else "; D7 was not estimable in the available window."
    )
    root_text = "No prior-window baseline was available for a daily anomaly comparison."
    if largest_drop is not None:
        root_text = (
            f"The largest observed daily purchase-rate drop occurred on "
            f"**{pd.Timestamp(largest_drop['event_date']).date()}**, at "
            f"**{largest_drop['conversion_delta']:.2%}** below its trailing baseline. "
            "The root-cause table preserves activity mix and price context for investigation."
        )

    report = f"""# Product Analytics Report

## Executive summary

This project analyzes a deterministic user-level sample of a public multi-category e-commerce event stream. It treats the observed data as behavioral telemetry, not as randomized experimental evidence. The pipeline profiles product usage, quantifies the session funnel and retention, predicts later-session purchase conversion, and turns the findings into a testable cart-recovery experiment.

## Data and quality

- Raw rows scanned: **{manifest['rows_scanned']:,}**
- Rows after deterministic user sampling and boundary-date removal: **{manifest['rows_after_boundary_trim']:,}**
- Clean events: **{quality['rows_after_cleaning']:,}**
- Users: **{quality['users']:,}**
- Sessions: **{quality['sessions']:,}**
- Products: **{quality['products']:,}**
- Analysis window: **{quality['start_time']}** through **{quality['end_time']}**
- Sampling rule: retain every event for users where `user_id % {manifest['sample_modulus']} < {manifest['sample_threshold']}` ({manifest['sample_rate']:.1%} of user-ID buckets)

The user-level rule preserves sampled user journeys better than random row sampling. The first and last calendar dates are removed because downloaded Parquet shards may start or stop mid-day.

## Product metrics

- Observed view-to-cart session rate: **{_funnel_value(funnel, 'view_to_cart_rate'):.2%}**
- Observed view-to-purchase session rate: **{_funnel_value(funnel, 'view_to_purchase_rate'):.2%}**
- Observed cart-to-purchase session rate: **{_funnel_value(funnel, 'cart_to_purchase_rate'):.2%}**
- Repeat-purchaser share of sampled users: **{repeat_share:.2%}**
- Mean daily active users in the sample: **{daily['active_users'].mean():,.0f}**
- {retention_text}

## Segment and root-cause findings

- {category_text}
- {root_text}
- Cart events changed by **{largest_cart_shift['cart_change']:+.1%}** on **{pd.Timestamp(largest_cart_shift['event_date']).date()}**, while view events changed by only **{largest_cart_shift['view_change']:+.1%}**. This discontinuity should trigger an instrumentation and product-context review before the cart trend is interpreted behaviorally.
- The highest-importance LightGBM feature by mean absolute SHAP value was **`{top_driver}`**. SHAP describes model behavior and should not be interpreted as a causal effect.

## Product recommendations

1. **Test recovery after demonstrated intent.** Target the proposed reminder only after a cart action and a five-minute no-purchase window. This is closer to a product decision than sending broad, high-volume notifications.
2. **Investigate category friction before changing ranking.** Review item availability, shipping, price distribution, and checkout errors in low-converting high-volume categories. The observed category gap is a diagnostic starting point, not evidence that category placement caused conversion.
3. **Add decision-grade instrumentation.** Capture experiment assignment, product impressions, checkout steps, verified order IDs, margin, notification exposure, and latency so future analysis can separate discovery, intent, checkout friction, and incremental impact.

## Conversion model

The prediction task uses only the first **{config.observation_minutes} minutes** of an eligible session and predicts whether a purchase occurs later in that same session. Sessions with an early purchase are excluded. The final holdout is strictly later in time than training data.

- Best validation model: **{modeling.best_model_name}**
- ROC-AUC: **{best['roc_auc']:.3f}**
- Average precision: **{best['average_precision']:.3f}**
- Precision at the training-selected threshold: **{best['precision']:.3f}**
- Recall at the training-selected threshold: **{best['recall']:.3f}**
- F1: **{best['f1']:.3f}**
- Brier score: **{best['brier_score']:.4f}**

## Experiment recommendation

Run a user-randomized cart-recovery experiment rather than treating historical cart behavior as causal evidence. Using the sample's observed eligible-session baseline and a 10% relative MDE, the simple two-proportion power calculation requires **{experiment['sample_size_per_arm']:,} users per arm** before traffic and clustering adjustments. See [experiment_plan.md](experiment_plan.md) for the full design and guardrails.

## Limitations

1. The analysis uses a deterministic sample and a continuous subset of the public event stream, not the entire seven-month dataset.
2. A purchase event represents an item event; the data does not provide a verified order identifier, margin, promotion exposure, or product impression denominator.
3. Observed funnel differences, SHAP values, and segment comparisons are associative. Only a randomized test can establish incremental impact.
4. Cohort retention is measured from first activity observed inside this window, not true customer acquisition.
5. User IDs are sampled consistently, but behavior before and after the downloaded time window is unavailable.
6. A sharp cart-event discontinuity appears during the window. The analysis flags it rather than assuming it represents an organic behavior change.
"""
    config.report_path.write_text(report, encoding="utf-8")

    bullets = f"""# Resume Bullets

- Built a reproducible product analytics pipeline over **{quality['rows_after_cleaning']:,} e-commerce events** from **{quality['users']:,} users**, preserving user journeys through deterministic sampling and defining DAU, session funnel, conversion, segmentation, and cohort-retention metrics.
- Developed a leakage-safe session conversion model using first-{config.observation_minutes}-minute behavior and chronological validation; compared baseline, logistic regression, and LightGBM models, reaching **{best['roc_auc']:.3f} ROC-AUC** and **{best['average_precision']:.3f} average precision ({ap_multiple:.1f}x the historical-rate baseline)**.
- Diagnosed a **{_funnel_value(funnel, 'view_to_cart_rate'):.2%} view-to-cart rate** and segment-level conversion differences with SHAP and root-cause analysis, then designed a powered user-randomized cart-recovery experiment with explicit metrics and guardrails.

All figures and claims above are generated from the current local run. Re-run the pipeline after changing the data sample before using the numbers externally.
"""
    (config.outputs_dir / "resume_bullets.md").write_text(bullets, encoding="utf-8")

    latex = f"""\\vspace{{-10pt}}
    \\resumeProjectHeading
      {{\\textbf{{E-Commerce Conversion Analytics \\& Experimentation}} $|$ \\emph{{Python, SQL, LightGBM, SHAP}}}}{{Sept. 2026 -- Present}}
      \\resumeItemListStart
      \\vspace{{4pt}}
        \\resumeItem{{Built a product analytics pipeline over \\textbf{{{quality['rows_after_cleaning']:,} e-commerce events}} from \\textbf{{{quality['users']:,} users}}, preserving user journeys through deterministic sampling and defining DAU, session funnel, conversion, user-segment, and D1/D7 retention metrics}}
        \\resumeItem{{Developed a leakage-safe session conversion model using first-{config.observation_minutes}-minute behavior and chronological validation, comparing baseline, logistic regression, and LightGBM models to achieve \\textbf{{{best['roc_auc']:.3f} ROC-AUC}} and \\textbf{{{best['average_precision']:.3f} average precision ({ap_multiple:.1f}$\\times$ baseline)}}}}
        \\resumeItem{{Diagnosed a \\textbf{{{_funnel_value(funnel, 'view_to_cart_rate') * 100:.2f}\\% view-to-cart rate}} and segment-level conversion differences with SHAP and root-cause analysis, then designed a powered user-randomized cart-recovery experiment with explicit success metrics and guardrails}}
      \\resumeItemListEnd
"""
    (config.outputs_dir / "resume_project.tex").write_text(latex, encoding="utf-8")

    github_summary = f"""# GitHub Summary

Product Data Science case study on **{quality['rows_after_cleaning']:,} sampled e-commerce events**. The project defines session funnel and retention metrics, builds a leakage-safe conversion model with chronological validation (**{best['roc_auc']:.3f} ROC-AUC; {best['average_precision']:.3f} AP**), explains segment and model behavior with SHAP, and converts observational findings into a powered cart-recovery experiment plan.
"""
    (config.outputs_dir / "github_summary.md").write_text(github_summary, encoding="utf-8")


def generate_readme(
    manifest: dict[str, Any],
    quality: dict[str, Any],
    funnel: pd.DataFrame,
    modeling: ModelingResult,
    config: ProjectConfig,
) -> None:
    best = modeling.metrics.iloc[0]
    readme = f"""# E-Commerce User Behavior Analytics & Conversion Optimization

A resume-ready Product Data Science project built from the public [Kaggle E-Commerce Behavior Data from Multi Category Store dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store). The project moves beyond prediction to define product metrics, diagnose funnel behavior, evaluate segment differences, and propose a powered experiment.

## Current verified run

| Item | Result |
| --- | ---: |
| Raw rows scanned | {manifest['rows_scanned']:,} |
| Clean sampled events | {quality['rows_after_cleaning']:,} |
| Users | {quality['users']:,} |
| Sessions | {quality['sessions']:,} |
| View-to-cart rate | {_funnel_value(funnel, 'view_to_cart_rate'):.2%} |
| View-to-purchase rate | {_funnel_value(funnel, 'view_to_purchase_rate'):.2%} |
| Best model | {modeling.best_model_name} |
| Validation ROC-AUC | {best['roc_auc']:.3f} |
| Validation average precision | {best['average_precision']:.3f} |

These results are produced from the checked local pipeline run. They are not estimates or placeholders.

## Product questions

1. Where do users drop between product view, cart, and purchase?
2. How do conversion and retention differ across category, brand, price, activity, and time segments?
3. Which early-session signals identify users likely to purchase later in the same session?
4. What observed changes coincide with conversion spikes or drops?
5. Which findings should become randomized product experiments rather than causal claims?

## Methods

- **Journey-preserving sample:** keep all rows for deterministic user-ID buckets; never sample independent rows.
- **Data quality:** validate timestamps, event types, prices, duplicates, missing categories/brands, and missing sessions.
- **Metrics:** DAU, event depth, purchase frequency, observed purchase value, session funnel, user segments, and D1/D7 observed-window retention.
- **Prediction:** historical-rate baseline, logistic regression, and LightGBM.
- **Leakage control:** first-{config.observation_minutes}-minute session features, later-session target, training-only threshold selection, and chronological holdout.
- **Explainability:** segment diagnostics, probability calibration, and LightGBM SHAP contributions.
- **Decision layer:** root-cause review plus a user-randomized cart-recovery experiment with MDE, power, guardrails, and rollout stages.

## Repository structure

```text
.
|-- data/                 # raw files are ignored; placement instructions included
|-- notebooks/            # guided EDA and modeling walkthrough
|-- outputs/              # predictions, serialized model, run summaries, resume bullets
|-- reports/
|   |-- figures/          # generated product and modeling charts
|   `-- tables/           # generated metrics and diagnostics
|-- scripts/              # download and end-to-end entry points
|-- sql/                  # warehouse-ready product metric references
|-- src/ecommerce_analytics/
|-- tests/
|-- README.md
`-- requirements.txt
```

## Reproduce the project

```powershell
cd E:\\master\\upenn\\ecommerce-conversion-analytics
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python scripts\\download_sample_data.py --shards 5
python scripts\\run_pipeline.py --force
```

Alternatively, download the original files from Kaggle or the [REES46 public dataset archive](https://rees46.com/en/datasets) and follow [data/README.md](data/README.md). Raw data is intentionally excluded from version control.

## Outputs

- [Product analytics report](reports/product_report.md)
- [Experiment plan](reports/experiment_plan.md)
- [Resume bullets](outputs/resume_bullets.md)
- Figures in `reports/figures/`
- Model and validation predictions in `outputs/`

## Interpretation and limitations

The pipeline distinguishes descriptive evidence from causal claims. It does not invent promotion exposure, order IDs, experiment groups, or revenue impact. The public event stream supports observed behavior analysis; incremental product impact requires the proposed randomized experiment. See the product report for sampling, cohort, measurement, and generalization limitations.
"""
    (config.project_root / "README.md").write_text(readme, encoding="utf-8")
