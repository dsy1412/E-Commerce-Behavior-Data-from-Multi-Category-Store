from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ProjectConfig


def _collapse_rare(series: pd.Series, top_n: int) -> pd.Series:
    values = series.fillna("unknown").astype(str)
    keep = set(values.value_counts().head(top_n).index)
    return values.where(values.isin(keep), "other")


def _add_completed_session_history(features: pd.DataFrame) -> pd.DataFrame:
    """Add user history known before each session starts.

    Some users have overlapping session IDs. Counting all earlier-starting
    sessions would expose outcomes from sessions that had not finished yet.
    This implementation only uses sessions whose end timestamp is strictly
    earlier than the current session start.
    """
    outputs: list[pd.DataFrame] = []
    for _, group in features.groupby("user_id", sort=False, observed=True):
        group = group.copy()
        starts = group["session_start"].astype("int64").to_numpy()
        end_order = np.argsort(group["session_end"].astype("int64").to_numpy())
        sorted_ends = group["session_end"].astype("int64").to_numpy()[end_order]
        sorted_targets = group["purchase_after_window"].to_numpy()[end_order]
        completed_counts = np.searchsorted(sorted_ends, starts, side="left")
        cumulative_purchases = np.cumsum(sorted_targets)
        completed_purchases = np.where(
            completed_counts > 0,
            cumulative_purchases[np.maximum(completed_counts - 1, 0)],
            0,
        )
        group["prior_sessions"] = completed_counts
        group["prior_purchase_sessions"] = completed_purchases
        outputs.append(group)
    return pd.concat(outputs, ignore_index=True)


def build_session_features(
    events: pd.DataFrame,
    config: ProjectConfig,
) -> pd.DataFrame:
    events = events.sort_values(["user_session", "event_time"]).copy()
    session_index = (
        events.groupby(["user_session", "user_id"], observed=True)
        .agg(
            session_start=("event_time", "min"),
            session_end=("event_time", "max"),
            session_has_purchase=("event_type", lambda values: values.eq("purchase").any()),
        )
        .reset_index()
    )
    session_index["observation_end"] = session_index["session_start"] + pd.Timedelta(
        minutes=config.observation_minutes
    )
    events = events.merge(
        session_index[
            ["user_session", "user_id", "session_start", "session_end", "observation_end"]
        ],
        on=["user_session", "user_id"],
        how="inner",
    )
    events["in_observation_window"] = events["event_time"].le(events["observation_end"])
    early_purchase_sessions = set(
        events.loc[
            events["in_observation_window"] & events["event_type"].eq("purchase"),
            "user_session",
        ]
    )
    observed = events.loc[
        events["in_observation_window"] & ~events["user_session"].isin(early_purchase_sessions)
    ].copy()
    if observed.empty:
        raise ValueError("No eligible sessions remain after applying the observation window.")

    grouped = observed.groupby(["user_session", "user_id"], observed=True)
    features = grouped.agg(
        session_start=("session_start", "first"),
        session_end=("session_end", "first"),
        observation_end=("observation_end", "first"),
        observed_events=("event_type", "size"),
        view_count=("event_type", lambda values: values.eq("view").sum()),
        cart_count=("event_type", lambda values: values.eq("cart").sum()),
        remove_count=("event_type", lambda values: values.eq("remove_from_cart").sum()),
        unique_products=("product_id", "nunique"),
        unique_categories=("category_id", "nunique"),
        unique_brands=("brand", "nunique"),
        mean_price=("price", "mean"),
        max_price=("price", "max"),
        min_price=("price", "min"),
        entry_category=("category_root", "first"),
        entry_brand=("brand", "first"),
    ).reset_index()

    future_purchases = (
        events.loc[
            events["event_time"].gt(events["observation_end"])
            & events["event_type"].eq("purchase")
            & ~events["user_session"].isin(early_purchase_sessions)
        ]
        .groupby(["user_session", "user_id"], observed=True)
        .size()
        .rename("future_purchase_events")
        .reset_index()
    )
    features = features.merge(future_purchases, on=["user_session", "user_id"], how="left")
    features["future_purchase_events"] = features["future_purchase_events"].fillna(0).astype(int)
    features["purchase_after_window"] = features["future_purchase_events"].gt(0).astype(int)

    max_event_time = events["event_time"].max()
    features = features.loc[features["observation_end"].le(max_event_time)].copy()
    features["session_hour"] = features["session_start"].dt.hour
    features["day_of_week"] = features["session_start"].dt.dayofweek
    features["is_weekend"] = features["day_of_week"].ge(5).astype(int)
    features["cart_per_view"] = features["cart_count"] / features["view_count"].clip(lower=1)
    features["remove_per_cart"] = features["remove_count"] / features["cart_count"].clip(lower=1)
    features["price_range"] = features["max_price"] - features["min_price"]
    features["entry_price_band"] = pd.cut(
        features["mean_price"],
        bins=[0, 10, 25, 50, 100, 250, 500, np.inf],
        labels=["0-10", "10-25", "25-50", "50-100", "100-250", "250-500", "500+"],
        include_lowest=True,
    ).astype(str)
    features["hour_band"] = pd.cut(
        features["session_hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["overnight", "morning", "afternoon", "evening"],
    ).astype(str)
    features["entry_category"] = _collapse_rare(features["entry_category"], top_n=30)
    features["entry_brand"] = _collapse_rare(features["entry_brand"], top_n=50)

    features = features.sort_values(["user_id", "session_start", "user_session"])
    features = _add_completed_session_history(features)
    features["prior_conversion_rate"] = (
        features["prior_purchase_sessions"] / features["prior_sessions"].clip(lower=1)
    )
    features["user_activity_tier"] = pd.cut(
        features["prior_sessions"],
        bins=[-1, 0, 3, np.inf],
        labels=["new", "returning", "engaged"],
    ).astype(str)
    features = features.sort_values("session_start").reset_index(drop=True)
    features.to_parquet(config.session_features_path, index=False)
    return features


def load_or_build_session_features(
    events: pd.DataFrame,
    config: ProjectConfig,
    force: bool = False,
) -> pd.DataFrame:
    if config.session_features_path.exists() and not force:
        return pd.read_parquet(config.session_features_path)
    return build_session_features(events, config)
