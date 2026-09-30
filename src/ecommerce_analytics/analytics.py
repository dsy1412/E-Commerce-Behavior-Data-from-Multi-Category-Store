from __future__ import annotations

import numpy as np
import pandas as pd


FUNNEL_STEPS = ("view", "cart", "purchase")


def build_daily_metrics(events: pd.DataFrame) -> pd.DataFrame:
    daily = (
        events.groupby("event_date", observed=True)
        .agg(
            active_users=("user_id", "nunique"),
            sessions=("user_session", "nunique"),
            events=("event_type", "size"),
            avg_event_price=("price", "mean"),
        )
        .reset_index()
    )
    event_counts = (
        events.pivot_table(
            index="event_date",
            columns="event_type",
            values="user_id",
            aggfunc="size",
            fill_value=0,
        )
        .reindex(columns=["view", "cart", "remove_from_cart", "purchase"], fill_value=0)
        .reset_index()
    )
    purchasers = (
        events.loc[events["event_type"].eq("purchase")]
        .groupby("event_date", observed=True)
        .agg(
            purchasing_users=("user_id", "nunique"),
            purchasing_sessions=("user_session", "nunique"),
            purchase_revenue=("price", "sum"),
        )
        .reset_index()
    )
    daily = daily.merge(event_counts, on="event_date", how="left").merge(
        purchasers, on="event_date", how="left"
    )
    count_columns = [
        "view",
        "cart",
        "remove_from_cart",
        "purchase",
        "purchasing_users",
        "purchasing_sessions",
        "purchase_revenue",
    ]
    daily[count_columns] = daily[count_columns].fillna(0)
    daily["views_per_user"] = daily["view"] / daily["active_users"].clip(lower=1)
    daily["carts_per_user"] = daily["cart"] / daily["active_users"].clip(lower=1)
    daily["purchase_events_per_user"] = daily["purchase"] / daily["active_users"].clip(lower=1)
    daily["daily_user_purchase_rate"] = (
        daily["purchasing_users"] / daily["active_users"].clip(lower=1)
    )
    daily["avg_purchase_item_value"] = (
        daily["purchase_revenue"] / daily["purchase"].clip(lower=1)
    )
    return daily.sort_values("event_date").reset_index(drop=True)


def build_session_journeys(events: pd.DataFrame) -> pd.DataFrame:
    step_times = (
        events.loc[events["event_type"].isin(FUNNEL_STEPS)]
        .pivot_table(
            index=["user_session", "user_id"],
            columns="event_type",
            values="event_time",
            aggfunc="min",
        )
        .reset_index()
    )
    for step in FUNNEL_STEPS:
        if step not in step_times:
            step_times[step] = pd.NaT
    step_times["cart_after_view"] = (
        step_times["view"].notna()
        & step_times["cart"].notna()
        & step_times["cart"].ge(step_times["view"])
    )
    step_times["purchase_after_view"] = (
        step_times["view"].notna()
        & step_times["purchase"].notna()
        & step_times["purchase"].ge(step_times["view"])
    )
    step_times["purchase_after_cart"] = (
        step_times["cart"].notna()
        & step_times["purchase"].notna()
        & step_times["purchase"].ge(step_times["cart"])
    )
    return step_times


def summarize_funnel(journeys: pd.DataFrame) -> pd.DataFrame:
    view_sessions = int(journeys["view"].notna().sum())
    cart_sessions = int(journeys["cart_after_view"].sum())
    purchase_after_view = int(journeys["purchase_after_view"].sum())
    purchase_after_cart = int(journeys["purchase_after_cart"].sum())
    rows = [
        ("view_sessions", view_sessions),
        ("cart_after_view_sessions", cart_sessions),
        ("purchase_after_view_sessions", purchase_after_view),
        ("purchase_after_cart_sessions", purchase_after_cart),
        ("view_to_cart_rate", cart_sessions / max(view_sessions, 1)),
        ("view_to_purchase_rate", purchase_after_view / max(view_sessions, 1)),
        ("cart_to_purchase_rate", purchase_after_cart / max(cart_sessions, 1)),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _entry_context(events: pd.DataFrame) -> pd.DataFrame:
    first_events = (
        events.sort_values("event_time")
        .groupby(["user_session", "user_id"], observed=True)
        .first()
        .reset_index()
    )
    first_events["entry_price_band"] = pd.cut(
        first_events["price"],
        bins=[0, 10, 25, 50, 100, 250, 500, np.inf],
        labels=["0-10", "10-25", "25-50", "50-100", "100-250", "250-500", "500+"],
        include_lowest=True,
    ).astype(str)
    first_events["entry_hour_band"] = pd.cut(
        first_events["event_hour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["overnight", "morning", "afternoon", "evening"],
    ).astype(str)
    return first_events[
        [
            "user_session",
            "user_id",
            "category_root",
            "brand",
            "entry_price_band",
            "entry_hour_band",
        ]
    ].rename(columns={"category_root": "entry_category", "brand": "entry_brand"})


def build_user_segments(events: pd.DataFrame) -> pd.DataFrame:
    user_metrics = (
        events.groupby("user_id", observed=True)
        .agg(
            events=("event_type", "size"),
            sessions=("user_session", "nunique"),
            active_days=("event_date", "nunique"),
            views=("event_type", lambda values: values.eq("view").sum()),
            carts=("event_type", lambda values: values.eq("cart").sum()),
            purchases=("event_type", lambda values: values.eq("purchase").sum()),
        )
        .reset_index()
    )
    purchase_sessions = (
        events.loc[events["event_type"].eq("purchase")]
        .groupby("user_id", observed=True)
        .agg(
            purchase_sessions=("user_session", "nunique"),
            observed_purchase_value=("price", "sum"),
        )
        .reset_index()
    )
    user_metrics = user_metrics.merge(purchase_sessions, on="user_id", how="left")
    user_metrics[["purchase_sessions", "observed_purchase_value"]] = user_metrics[
        ["purchase_sessions", "observed_purchase_value"]
    ].fillna(0)
    conditions = [
        user_metrics["purchase_sessions"].ge(2),
        user_metrics["purchase_sessions"].eq(1),
        user_metrics["carts"].gt(0),
    ]
    user_metrics["behavior_segment"] = np.select(
        conditions,
        ["repeat purchaser", "purchaser", "cart abandoner"],
        default="browser",
    )
    user_metrics["activity_segment"] = pd.cut(
        user_metrics["events"],
        bins=[0, 5, 20, np.inf],
        labels=["low", "medium", "high"],
        include_lowest=True,
    ).astype(str)
    return user_metrics


def build_segment_funnels(
    events: pd.DataFrame,
    journeys: pd.DataFrame,
    user_segments: pd.DataFrame,
) -> pd.DataFrame:
    context = _entry_context(events)
    frame = journeys.merge(context, on=["user_session", "user_id"], how="left").merge(
        user_segments[["user_id", "activity_segment"]], on="user_id", how="left"
    )
    outputs: list[pd.DataFrame] = []
    for column in (
        "entry_category",
        "entry_brand",
        "entry_price_band",
        "entry_hour_band",
        "activity_segment",
    ):
        grouped = (
            frame.groupby(column, dropna=False, observed=True)
            .agg(
                sessions=("user_session", "nunique"),
                view_sessions=("view", lambda values: values.notna().sum()),
                cart_after_view_sessions=("cart_after_view", "sum"),
                purchase_after_view_sessions=("purchase_after_view", "sum"),
                purchase_after_cart_sessions=("purchase_after_cart", "sum"),
            )
            .reset_index()
            .rename(columns={column: "segment"})
        )
        grouped["dimension"] = column
        grouped["view_to_cart_rate"] = (
            grouped["cart_after_view_sessions"] / grouped["view_sessions"].clip(lower=1)
        )
        grouped["view_to_purchase_rate"] = (
            grouped["purchase_after_view_sessions"] / grouped["view_sessions"].clip(lower=1)
        )
        grouped["cart_to_purchase_rate"] = (
            grouped["purchase_after_cart_sessions"]
            / grouped["cart_after_view_sessions"].clip(lower=1)
        )
        outputs.append(grouped)
    return pd.concat(outputs, ignore_index=True)


def build_cohort_retention(
    events: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 7),
) -> pd.DataFrame:
    activity = events[["user_id", "event_date"]].drop_duplicates()
    first_seen = activity.groupby("user_id", observed=True)["event_date"].min().rename("cohort_date")
    activity = activity.join(first_seen, on="user_id")
    activity["day_number"] = (activity["event_date"] - activity["cohort_date"]).dt.days
    max_date = activity["event_date"].max()
    rows: list[dict[str, object]] = []
    cohort_sizes = first_seen.value_counts().sort_index()
    for cohort_date, cohort_size in cohort_sizes.items():
        for horizon in horizons:
            if cohort_date + pd.Timedelta(days=horizon) > max_date:
                continue
            retained = activity.loc[
                activity["cohort_date"].eq(cohort_date) & activity["day_number"].eq(horizon),
                "user_id",
            ].nunique()
            rows.append(
                {
                    "cohort_date": cohort_date,
                    "horizon_days": horizon,
                    "eligible_users": int(cohort_size),
                    "retained_users": int(retained),
                    "retention_rate": retained / max(int(cohort_size), 1),
                }
            )
    return pd.DataFrame(rows)


def build_root_cause_table(daily: pd.DataFrame, baseline_days: int = 3) -> pd.DataFrame:
    frame = daily.copy().sort_values("event_date")
    frame["conversion_baseline"] = (
        frame["daily_user_purchase_rate"].shift(1).rolling(baseline_days, min_periods=2).mean()
    )
    frame["conversion_delta"] = (
        frame["daily_user_purchase_rate"] - frame["conversion_baseline"]
    )
    frame["purchase_event_share"] = frame["purchase"] / frame["events"].clip(lower=1)
    frame["cart_event_share"] = frame["cart"] / frame["events"].clip(lower=1)
    frame["view_event_share"] = frame["view"] / frame["events"].clip(lower=1)
    valid = frame.loc[frame["conversion_baseline"].notna()].copy()
    if valid.empty:
        return valid
    tail_count = min(3, len(valid))
    selected = pd.concat(
        [valid.nsmallest(tail_count, "conversion_delta"), valid.nlargest(tail_count, "conversion_delta")]
    ).drop_duplicates(subset="event_date")
    selected["period_type"] = np.where(selected["conversion_delta"].lt(0), "drop", "spike")
    return selected.sort_values("conversion_delta").reset_index(drop=True)
