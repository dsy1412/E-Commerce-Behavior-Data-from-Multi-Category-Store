# E-Commerce User Behavior Analytics & Conversion Optimization

A resume-ready Product Data Science project built from the public [Kaggle E-Commerce Behavior Data from Multi Category Store dataset](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store). The project moves beyond prediction to define product metrics, diagnose funnel behavior, evaluate segment differences, and propose a powered experiment.

## Current verified run

| Item | Result |
| --- | ---: |
| Raw rows scanned | 18,360,000 |
| Clean sampled events | 314,953 |
| Users | 31,287 |
| Sessions | 69,466 |
| View-to-cart rate | 6.04% |
| View-to-purchase rate | 5.33% |
| Best model | logistic_regression |
| Validation ROC-AUC | 0.819 |
| Validation average precision | 0.136 |

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
- **Leakage control:** first-5-minute session features, later-session target, training-only threshold selection, and chronological holdout.
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
cd E:\master\upenn\ecommerce-conversion-analytics
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python scripts\download_sample_data.py --shards 5
python scripts\run_pipeline.py --force
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
