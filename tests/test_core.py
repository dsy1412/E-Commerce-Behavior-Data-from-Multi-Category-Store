from __future__ import annotations

import pandas as pd

from ecommerce_analytics.analytics import build_session_journeys, summarize_funnel
from ecommerce_analytics.config import ProjectConfig
from ecommerce_analytics.experimentation import create_experiment_plan, two_proportion_sample_size
from ecommerce_analytics.features import build_session_features


def _events() -> pd.DataFrame:
    rows = [
        ("2019-11-02 09:00", "view", 1, "s1", 101, 10, "electronics.phone", "alpha", 200),
        ("2019-11-02 09:02", "cart", 1, "s1", 101, 10, "electronics.phone", "alpha", 200),
        ("2019-11-02 09:10", "purchase", 1, "s1", 101, 10, "electronics.phone", "alpha", 200),
        ("2019-11-02 10:00", "view", 1, "s2", 102, 11, "appliances.kitchen", "beta", 80),
        ("2019-11-02 10:02", "purchase", 1, "s2", 102, 11, "appliances.kitchen", "beta", 80),
        ("2019-11-02 11:00", "view", 2, "s3", 103, 12, "apparel.shoes", "gamma", 60),
        ("2019-11-02 11:08", "view", 2, "s3", 104, 12, "apparel.shoes", "gamma", 70),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "event_time",
            "event_type",
            "user_id",
            "user_session",
            "product_id",
            "category_id",
            "category_code",
            "brand",
            "price",
        ],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True)
    frame["event_date"] = frame["event_time"].dt.tz_convert(None).dt.normalize()
    frame["event_hour"] = frame["event_time"].dt.hour
    frame["day_of_week"] = frame["event_time"].dt.dayofweek
    frame["category_root"] = frame["category_code"].str.split(".").str[0]
    return frame


def test_ordered_funnel_counts() -> None:
    summary = summarize_funnel(build_session_journeys(_events())).set_index("metric")["value"]
    assert summary["view_sessions"] == 3
    assert summary["cart_after_view_sessions"] == 1
    assert summary["purchase_after_cart_sessions"] == 1


def test_session_features_exclude_early_conversions(tmp_path) -> None:
    config = ProjectConfig(project_root=tmp_path, observation_minutes=5)
    config.ensure_directories()
    features = build_session_features(_events(), config)
    assert set(features["user_session"]) == {"s1", "s3"}
    target = features.set_index("user_session")["purchase_after_window"]
    assert target["s1"] == 1
    assert target["s3"] == 0


def test_power_increases_for_smaller_effect() -> None:
    large_effect = two_proportion_sample_size(0.10, relative_lift=0.20)
    small_effect = two_proportion_sample_size(0.10, relative_lift=0.05)
    assert small_effect["sample_size_per_arm"] > large_effect["sample_size_per_arm"]


def test_user_history_excludes_overlapping_unfinished_sessions(tmp_path) -> None:
    rows = [
        ("2019-11-02 09:00", "view", 1, "s1", 101, 10, "electronics.phone", "alpha", 200),
        ("2019-11-02 09:10", "purchase", 1, "s1", 101, 10, "electronics.phone", "alpha", 200),
        ("2019-11-02 09:05", "view", 1, "s2", 102, 10, "electronics.phone", "alpha", 150),
        ("2019-11-02 09:06", "cart", 1, "s2", 102, 10, "electronics.phone", "alpha", 150),
        ("2019-11-02 09:20", "view", 1, "s3", 103, 10, "electronics.phone", "alpha", 100),
        ("2019-11-02 09:26", "view", 1, "s3", 104, 10, "electronics.phone", "alpha", 120),
    ]
    events = pd.DataFrame(rows, columns=_events().columns[:9])
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True)
    events["event_date"] = events["event_time"].dt.tz_convert(None).dt.normalize()
    events["event_hour"] = events["event_time"].dt.hour
    events["day_of_week"] = events["event_time"].dt.dayofweek
    events["category_root"] = events["category_code"].str.split(".").str[0]
    config = ProjectConfig(project_root=tmp_path, observation_minutes=5)
    config.ensure_directories()

    features = build_session_features(events, config).set_index("user_session")

    assert features.loc["s2", "prior_sessions"] == 0
    assert features.loc["s3", "prior_sessions"] == 2
    assert features.loc["s3", "prior_purchase_sessions"] == 1


def test_experiment_uses_first_eligible_session_per_user(tmp_path) -> None:
    sessions = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "user_session": ["u1-first", "u1-later", "u2-first"],
            "session_start": pd.to_datetime(
                ["2019-11-01 09:00", "2019-11-02 09:00", "2019-11-01 10:00"],
                utc=True,
            ),
            "cart_count": [1, 1, 1],
            "purchase_after_window": [0, 1, 1],
        }
    )
    config = ProjectConfig(project_root=tmp_path)
    config.ensure_directories()

    result = create_experiment_plan(sessions, config)

    assert result["observed_eligible_users"] == 2
    assert result["observed_converted_users"] == 1
    assert result["baseline_rate"] == 0.5
