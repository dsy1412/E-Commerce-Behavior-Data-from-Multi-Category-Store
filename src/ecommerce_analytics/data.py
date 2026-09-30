from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .config import ProjectConfig


VALID_EVENT_TYPES = {"view", "cart", "remove_from_cart", "purchase"}


def _derive_missing_sessions(events: pd.DataFrame) -> pd.Series:
    missing = events.loc[events["user_session"].isna(), ["user_id", "event_time"]].copy()
    if missing.empty:
        return pd.Series(dtype="object")
    missing = missing.sort_values(["user_id", "event_time"])
    gap = missing.groupby("user_id")["event_time"].diff()
    new_session = gap.isna() | (gap > pd.Timedelta(minutes=30))
    sequence = new_session.groupby(missing["user_id"]).cumsum().astype(int)
    return (
        "derived_"
        + missing["user_id"].astype("Int64").astype(str)
        + "_"
        + sequence.astype(str)
    ).reindex(events.index)


def clean_events(
    events: pd.DataFrame,
    config: ProjectConfig,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    frame = events.copy()
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    frame["event_type"] = frame["event_type"].astype(str).str.lower().str.strip()
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["user_id"] = pd.to_numeric(frame["user_id"], errors="coerce").astype("Int64")

    missing_time = int(frame["event_time"].isna().sum())
    missing_user = int(frame["user_id"].isna().sum())
    invalid_event_type = int((~frame["event_type"].isin(VALID_EVENT_TYPES)).sum())
    invalid_price = int(((frame["price"].isna()) | (frame["price"] <= 0)).sum())
    duplicates = int(frame.duplicated().sum())

    frame = frame.loc[
        frame["event_time"].notna()
        & frame["user_id"].notna()
        & frame["event_type"].isin(VALID_EVENT_TYPES)
        & frame["price"].notna()
        & (frame["price"] > 0)
    ].drop_duplicates()

    derived_sessions = _derive_missing_sessions(frame)
    frame["user_session"] = frame["user_session"].astype("object")
    frame.loc[frame["user_session"].isna(), "user_session"] = derived_sessions.dropna()
    frame = frame.loc[frame["user_session"].notna()].copy()

    frame["category_code"] = frame["category_code"].fillna("unknown").astype(str)
    frame["brand"] = frame["brand"].fillna("unknown").astype(str).str.lower()
    frame["category_root"] = frame["category_code"].str.split(".").str[0].replace("unknown", "unknown")
    frame["event_date"] = frame["event_time"].dt.tz_convert(None).dt.normalize()
    frame["event_hour"] = frame["event_time"].dt.hour.astype("int8")
    frame["day_of_week"] = frame["event_time"].dt.dayofweek.astype("int8")
    frame = frame.sort_values(["event_time", "user_id", "user_session"]).reset_index(drop=True)

    quality = {
        "rows_before_cleaning": int(len(events)),
        "rows_after_cleaning": int(len(frame)),
        "missing_event_time_rows": missing_time,
        "missing_user_rows": missing_user,
        "invalid_event_type_rows": invalid_event_type,
        "invalid_price_rows": invalid_price,
        "exact_duplicate_rows_removed": duplicates,
        "users": int(frame["user_id"].nunique()),
        "sessions": int(frame["user_session"].nunique()),
        "products": int(frame["product_id"].nunique()),
        "categories": int(frame["category_id"].nunique()),
        "brands": int(frame["brand"].nunique()),
        "start_time": str(frame["event_time"].min()),
        "end_time": str(frame["event_time"].max()),
    }
    quality_table = pd.DataFrame(
        {
            "column": frame.columns,
            "dtype": [str(dtype) for dtype in frame.dtypes],
            "missing_count": frame.isna().sum().to_numpy(),
            "missing_pct": frame.isna().mean().mul(100).round(3).to_numpy(),
            "unique_values": [frame[column].nunique(dropna=True) for column in frame],
        }
    ).sort_values(["missing_pct", "column"], ascending=[False, True])

    frame.to_parquet(config.cleaned_events_path, index=False)
    (config.outputs_dir / "data_quality.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    quality_table.to_csv(config.tables_dir / "data_quality_summary.csv", index=False)
    return frame, quality, quality_table


def load_or_clean_events(
    sampled_events: pd.DataFrame,
    config: ProjectConfig,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    quality_path = config.outputs_dir / "data_quality.json"
    table_path = config.tables_dir / "data_quality_summary.csv"
    if config.cleaned_events_path.exists() and quality_path.exists() and table_path.exists() and not force:
        return (
            pd.read_parquet(config.cleaned_events_path),
            json.loads(quality_path.read_text(encoding="utf-8")),
            pd.read_csv(table_path),
        )
    return clean_events(sampled_events, config)
