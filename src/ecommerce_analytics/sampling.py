from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import pandas as pd
import pyarrow.parquet as pq

from .config import ProjectConfig


EXPECTED_COLUMNS = [
    "event_time",
    "event_type",
    "product_id",
    "category_id",
    "category_code",
    "brand",
    "price",
    "user_id",
    "user_session",
]


def discover_source_files(config: ProjectConfig) -> list[Path]:
    parquet_files = sorted((config.raw_data_dir / "hf").glob("*.parquet"))
    if parquet_files:
        return parquet_files
    gzip_files = sorted(config.raw_data_dir.glob("*.csv.gz"))
    if gzip_files:
        return gzip_files
    csv_files = sorted(config.raw_data_dir.glob("*.csv"))
    return csv_files


def _parquet_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    parquet_file = pq.ParquetFile(path)
    missing = set(EXPECTED_COLUMNS) - set(parquet_file.schema.names)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")
    for batch in parquet_file.iter_batches(
        batch_size=batch_size,
        columns=EXPECTED_COLUMNS,
    ):
        yield batch.to_pandas()


def _csv_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    compression = "gzip" if path.suffix == ".gz" else "infer"
    for chunk in pd.read_csv(
        path,
        usecols=EXPECTED_COLUMNS,
        chunksize=batch_size,
        compression=compression,
        low_memory=False,
    ):
        yield chunk


def iter_source_batches(path: Path, batch_size: int) -> Iterator[pd.DataFrame]:
    if path.suffix == ".parquet":
        yield from _parquet_batches(path, batch_size)
    else:
        yield from _csv_batches(path, batch_size)


def create_user_level_sample(
    config: ProjectConfig,
    force: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Scan source files and retain all rows for a deterministic set of users."""

    config.ensure_directories()
    manifest_path = config.processed_data_dir / "sampling_manifest.json"
    if config.sampled_events_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_rule = {
            "sample_modulus": config.sample_modulus,
            "sample_threshold": config.sample_threshold,
        }
        if all(manifest.get(key) == value for key, value in expected_rule.items()):
            return pd.read_parquet(config.sampled_events_path), manifest

    source_files = discover_source_files(config)
    if not source_files:
        raise FileNotFoundError(
            "No source data found. Place 2019-Oct.csv.gz in data/raw or "
            "Parquet shards in data/raw/hf."
        )

    sampled_chunks: list[pd.DataFrame] = []
    rows_scanned = 0
    rows_selected = 0
    for source in source_files:
        print(f"Sampling {source.name}")
        for chunk in iter_source_batches(source, config.batch_size):
            rows_scanned += len(chunk)
            user_ids = pd.to_numeric(chunk["user_id"], errors="coerce")
            selected = user_ids.mod(config.sample_modulus) < config.sample_threshold
            sampled = chunk.loc[selected.fillna(False)].copy()
            if not sampled.empty:
                sampled_chunks.append(sampled)
                rows_selected += len(sampled)

    if not sampled_chunks:
        raise ValueError("The deterministic user sampling rule selected no events.")

    sample = pd.concat(sampled_chunks, ignore_index=True)
    sample["event_time"] = pd.to_datetime(sample["event_time"], utc=True, errors="coerce")
    sample = sample.loc[sample["event_time"].notna()].copy()
    sample["event_date"] = sample["event_time"].dt.date

    # Source shards may begin or end mid-day. Remove boundary dates so daily
    # metrics and sessions are not calculated on incomplete dates.
    min_date = sample["event_date"].min()
    max_date = sample["event_date"].max()
    if min_date < max_date:
        sample = sample.loc[
            (sample["event_date"] > min_date) & (sample["event_date"] < max_date)
        ].copy()

    sample = sample.drop(columns="event_date").sort_values("event_time").reset_index(drop=True)
    sample.to_parquet(config.sampled_events_path, index=False)

    manifest = {
        "source_files": [
            {"name": path.name, "bytes": path.stat().st_size} for path in source_files
        ],
        "rows_scanned": rows_scanned,
        "rows_selected_before_boundary_trim": rows_selected,
        "rows_after_boundary_trim": int(len(sample)),
        "unique_users": int(sample["user_id"].nunique()),
        "sample_modulus": config.sample_modulus,
        "sample_threshold": config.sample_threshold,
        "sample_rate": config.sample_threshold / config.sample_modulus,
        "boundary_dates_removed": [str(min_date), str(max_date)],
        "final_start_time": str(sample["event_time"].min()),
        "final_end_time": str(sample["event_time"].max()),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return sample, manifest
