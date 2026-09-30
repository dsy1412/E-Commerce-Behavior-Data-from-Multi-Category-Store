from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


BASE_URL = (
    "https://huggingface.co/datasets/kevykibbz/"
    "ecommerce-behavior-data-from-multi-category-store_oct-nov_2019/resolve/"
    "refs%2Fconvert%2Fparquet/default/partial-train/{index:04d}.parquet?download=true"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def download(index: int, destination: Path) -> None:
    if destination.exists():
        print(f"Using existing {destination.name}")
        return
    temporary = destination.with_suffix(".parquet.part")
    print(f"Downloading shard {index:04d}...")
    with urllib.request.urlopen(BASE_URL.format(index=index)) as response, temporary.open("wb") as file:
        shutil.copyfileobj(response, file)
    temporary.replace(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download contiguous public Parquet shards.")
    parser.add_argument("--shards", type=int, default=5, choices=range(1, 11))
    args = parser.parse_args()
    destination = PROJECT_ROOT / "data" / "raw" / "hf"
    destination.mkdir(parents=True, exist_ok=True)
    for index in range(args.shards):
        download(index, destination / f"{index:04d}.parquet")


if __name__ == "__main__":
    main()
