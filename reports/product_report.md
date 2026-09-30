# Product Analytics Report

## Executive summary

This project analyzes a deterministic user-level sample of a public multi-category e-commerce event stream. It treats the observed data as behavioral telemetry, not as randomized experimental evidence. The pipeline profiles product usage, quantifies the session funnel and retention, predicts later-session purchase conversion, and turns the findings into a testable cart-recovery experiment.

## Data and quality

- Raw rows scanned: **18,360,000**
- Rows after deterministic user sampling and boundary-date removal: **315,530**
- Clean events: **314,953**
- Users: **31,287**
- Sessions: **69,466**
- Products: **40,839**
- Analysis window: **2019-11-02 00:00:42+00:00** through **2019-11-10 23:59:24+00:00**
- Sampling rule: retain every event for users where `user_id % 1000 < 20` (2.0% of user-ID buckets)

The user-level rule preserves sampled user journeys better than random row sampling. The first and last calendar dates are removed because downloaded Parquet shards may start or stop mid-day.

## Product metrics

- Observed view-to-cart session rate: **6.04%**
- Observed view-to-purchase session rate: **5.33%**
- Observed cart-to-purchase session rate: **48.64%**
- Repeat-purchaser share of sampled users: **1.97%**
- Mean daily active users in the sample: **5,301**
- D1 retention was **13.91%** and D7 retention was **11.95%** among eligible observed cohorts.

## Segment and root-cause findings

- Among entry categories with at least 100 sampled sessions, **electronics** had the highest observed view-to-purchase rate (7.03%), while **furniture** had the lowest (1.48%). These are descriptive signals for follow-up, not causal category effects.
- The largest observed daily purchase-rate drop occurred on **2019-11-09**, at **-0.78%** below its trailing baseline. The root-cause table preserves activity mix and price context for investigation.
- Cart events changed by **+277.7%** on **2019-11-08**, while view events changed by only **-0.1%**. This discontinuity should trigger an instrumentation and product-context review before the cart trend is interpreted behaviorally.
- The highest-importance LightGBM feature by mean absolute SHAP value was **`numeric__observed_events`**. SHAP describes model behavior and should not be interpreted as a causal effect.

## Product recommendations

1. **Test recovery after demonstrated intent.** Target the proposed reminder only after a cart action and a five-minute no-purchase window. This is closer to a product decision than sending broad, high-volume notifications.
2. **Investigate category friction before changing ranking.** Review item availability, shipping, price distribution, and checkout errors in low-converting high-volume categories. The observed category gap is a diagnostic starting point, not evidence that category placement caused conversion.
3. **Add decision-grade instrumentation.** Capture experiment assignment, product impressions, checkout steps, verified order IDs, margin, notification exposure, and latency so future analysis can separate discovery, intent, checkout friction, and incremental impact.

## Conversion model

The prediction task uses only the first **5 minutes** of an eligible session and predicts whether a purchase occurs later in that same session. Sessions with an early purchase are excluded. The final holdout is strictly later in time than training data.

- Best validation model: **logistic_regression**
- ROC-AUC: **0.819**
- Average precision: **0.136**
- Precision at the training-selected threshold: **0.180**
- Recall at the training-selected threshold: **0.266**
- F1: **0.215**
- Brier score: **0.0165**

## Experiment recommendation

Run a user-randomized cart-recovery experiment rather than treating historical cart behavior as causal evidence. Using the sample's observed eligible-session baseline and a 10% relative MDE, the simple two-proportion power calculation requires **6,096 users per arm** before traffic and clustering adjustments. See [experiment_plan.md](experiment_plan.md) for the full design and guardrails.

## Limitations

1. The analysis uses a deterministic sample and a continuous subset of the public event stream, not the entire seven-month dataset.
2. A purchase event represents an item event; the data does not provide a verified order identifier, margin, promotion exposure, or product impression denominator.
3. Observed funnel differences, SHAP values, and segment comparisons are associative. Only a randomized test can establish incremental impact.
4. Cohort retention is measured from first activity observed inside this window, not true customer acquisition.
5. User IDs are sampled consistently, but behavior before and after the downloaded time window is unavailable.
6. A sharp cart-event discontinuity appears during the window. The analysis flags it rather than assuming it represents an organic behavior change.
