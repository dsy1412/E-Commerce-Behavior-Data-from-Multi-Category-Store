from __future__ import annotations

from .analytics import (
    build_cohort_retention,
    build_daily_metrics,
    build_root_cause_table,
    build_segment_funnels,
    build_session_journeys,
    build_user_segments,
    summarize_funnel,
)
from .config import ProjectConfig
from .data import load_or_clean_events
from .experimentation import create_experiment_plan
from .features import load_or_build_session_features
from .modeling import train_conversion_models
from .reporting import generate_product_report, generate_readme
from .sampling import create_user_level_sample
from .visualization import create_figures


def run_pipeline(config: ProjectConfig | None = None, force: bool = False) -> None:
    config = config or ProjectConfig()
    config.ensure_directories()

    sampled, manifest = create_user_level_sample(config, force=force)
    events, quality, _ = load_or_clean_events(sampled, config, force=force)
    daily = build_daily_metrics(events)
    journeys = build_session_journeys(events)
    funnel = summarize_funnel(journeys)
    user_segments = build_user_segments(events)
    segment_funnels = build_segment_funnels(events, journeys, user_segments)
    retention = build_cohort_retention(events)
    root_causes = build_root_cause_table(daily)

    daily.to_csv(config.tables_dir / "daily_metrics.csv", index=False)
    funnel.to_csv(config.tables_dir / "funnel_summary.csv", index=False)
    user_segments.to_parquet(config.outputs_dir / "user_segments.parquet", index=False)
    user_segments["behavior_segment"].value_counts().rename_axis("segment").reset_index(
        name="users"
    ).to_csv(config.tables_dir / "user_segment_summary.csv", index=False)
    segment_funnels.to_csv(config.tables_dir / "segment_funnels.csv", index=False)
    retention.to_csv(config.tables_dir / "cohort_retention.csv", index=False)
    root_causes.to_csv(config.tables_dir / "root_cause_days.csv", index=False)

    session_features = load_or_build_session_features(events, config, force=force)
    modeling = train_conversion_models(session_features, config)
    experiment = create_experiment_plan(session_features, config)
    create_figures(
        daily,
        funnel,
        user_segments,
        segment_funnels,
        retention,
        modeling,
        config,
    )
    generate_product_report(
        manifest,
        quality,
        daily,
        funnel,
        user_segments,
        segment_funnels,
        retention,
        root_causes,
        modeling,
        experiment,
        config,
    )
    generate_readme(manifest, quality, funnel, modeling, config)
    print(f"Pipeline complete. Report: {config.report_path}")
