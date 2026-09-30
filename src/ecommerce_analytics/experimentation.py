from __future__ import annotations

import json
from statistics import NormalDist

import pandas as pd

from .config import ProjectConfig


def two_proportion_sample_size(
    baseline_rate: float,
    relative_lift: float = 0.10,
    alpha: float = 0.05,
    power: float = 0.80,
) -> dict[str, float | int]:
    p1 = min(max(baseline_rate, 1e-5), 1 - 1e-5)
    p2 = min(p1 * (1 + relative_lift), 1 - 1e-5)
    absolute_mde = p2 - p1
    pooled = (p1 + p2) / 2
    normal = NormalDist()
    z_alpha = normal.inv_cdf(1 - alpha / 2)
    z_power = normal.inv_cdf(power)
    numerator = (
        z_alpha * (2 * pooled * (1 - pooled)) ** 0.5
        + z_power * (p1 * (1 - p1) + p2 * (1 - p2)) ** 0.5
    ) ** 2
    n_per_arm = int((numerator / max(absolute_mde**2, 1e-12)) + 0.9999)
    return {
        "baseline_rate": p1,
        "relative_mde": relative_lift,
        "absolute_mde": absolute_mde,
        "target_rate": p2,
        "alpha": alpha,
        "power": power,
        "sample_size_per_arm": n_per_arm,
        "total_sample_size": n_per_arm * 2,
    }


def create_experiment_plan(
    session_features: pd.DataFrame,
    config: ProjectConfig,
) -> dict[str, float | int]:
    eligible = session_features.loc[
        session_features["cart_count"].gt(0)
        & session_features["purchase_after_window"].isin([0, 1])
    ]
    eligible_users = (
        eligible.groupby("user_id", observed=True)["purchase_after_window"]
        .max()
        .rename("converted")
        .reset_index()
    )
    baseline = float(eligible_users["converted"].mean()) if len(eligible_users) else 0.01
    calculation = two_proportion_sample_size(baseline_rate=baseline)
    calculation["observed_eligible_sessions"] = int(len(eligible))
    calculation["observed_eligible_users"] = int(len(eligible_users))
    calculation["observed_converted_users"] = int(eligible_users["converted"].sum())

    plan = f"""# Cart-Recovery Experiment Plan

## Decision

Test whether a timely, personalized cart reminder increases purchase conversion among sessions that add at least one item to cart but have not purchased during the first {config.observation_minutes} minutes.

This is a proposed randomized experiment. The dataset contains no treatment assignment, so the project does not claim a causal promotion or reminder effect.

## Design

- **Population:** users whose first eligible cart session has no purchase in the initial observation window.
- **Control:** current shopping experience with no additional reminder.
- **Treatment:** one reminder showing the most recently carted item and a direct return-to-cart action.
- **Randomization unit:** user ID, assigned persistently to avoid cross-session contamination.
- **Primary metric:** user-level purchase conversion within a pre-specified follow-up window.
- **Secondary metrics:** observed purchase value per randomized eligible user and time to purchase.
- **Guardrails:** remove-from-cart rate, notification opt-out rate, page latency, and seven-day user retention. The public schema includes cart-removal events, but production instrumentation is required for the remaining guardrails and should verify that removal events are populated reliably.
- **Analysis:** intention-to-treat comparison of proportions with a two-sided z-test; report absolute lift, relative lift, 95% confidence interval, and pre-specified segment interactions.
- **Heterogeneous effects:** examine pre-registered interactions for new versus returning users, price band, and entry category. Treat these as secondary and adjust for multiple comparisons.
- **Rollout:** 5% instrumentation check, 20% powered pilot, then broader rollout only if the primary metric improves without material guardrail harm.

## Power calculation

- Observed eligible sessions in the analysis sample: **{len(eligible):,}**
- Observed eligible users in the analysis sample: **{len(eligible_users):,}**
- Observed user-level baseline conversion: **{baseline:.2%}**
- Minimum detectable effect: **10% relative** ({calculation['absolute_mde']:.2%} absolute)
- Significance level: **{calculation['alpha']:.0%}**
- Power: **{calculation['power']:.0%}**
- Required users per arm: **{calculation['sample_size_per_arm']:,}**
- Required total users: **{calculation['total_sample_size']:,}**

The baseline is descriptive and sample-specific. Production planning should recompute power using current traffic, expected eligibility, cluster effects from user-level randomization, and the actual business-relevant MDE.
"""
    config.experiment_path.write_text(plan, encoding="utf-8")
    (config.outputs_dir / "experiment_power.json").write_text(
        json.dumps(calculation, indent=2), encoding="utf-8"
    )
    return calculation
