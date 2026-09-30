# Data setup

This project uses the public [E-Commerce Behavior Data from Multi Category Store](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store).

## Reproduce the verified sample

The checked run uses five contiguous November 2019 Parquet shards from a [Hugging Face conversion of the Kaggle files](https://huggingface.co/datasets/kevykibbz/ecommerce-behavior-data-from-multi-category-store_oct-nov_2019). Download them with:

```powershell
python scripts\download_sample_data.py --shards 5
```

The shards cover a continuous portion of the event stream. The pipeline removes the first and last calendar dates because shard boundaries can create incomplete days.

## Use an original monthly file

The original provider, REES46, also hosts monthly source files. For example:

```text
https://data.rees46.com/datasets/marketplace/2019-Oct.csv.gz
```

Delete any files in `data/raw/hf/` before using a monthly CSV because the pipeline gives Parquet shards precedence. Then place the compressed file at `data/raw/2019-Oct.csv.gz`.

For either source, the pipeline applies the deterministic user-level rule:

```text
user_id % 1000 < 20
```

This selects approximately 2% of user-ID buckets while preserving every available event for each selected user. Raw and processed datasets are excluded from Git.
