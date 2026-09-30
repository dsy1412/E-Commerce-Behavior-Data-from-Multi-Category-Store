# Cart-Recovery Experiment Plan

## Decision

Test whether a timely, personalized cart reminder increases purchase conversion among sessions that add at least one item to cart but have not purchased during the first 5 minutes.

This is a proposed randomized experiment. The dataset contains no treatment assignment, so the project does not claim a causal promotion or reminder effect.

## Design

- **Population:** users whose first eligible cart session has no purchase in the initial observation window.
- **Control:** current shopping experience with no additional reminder.
- **Treatment:** one reminder showing the most recently carted item and a direct return-to-cart action.
- **Randomization unit:** user ID, assigned persistently to avoid cross-session contamination.
- **Primary metric:** user-level purchase conversion within a pre-specified post-randomization window.
- **Secondary metrics:** observed purchase value per randomized eligible user and time to purchase.
- **Guardrails:** remove-from-cart rate, notification opt-out rate, page latency, and seven-day user retention. The public schema includes cart-removal events, but production instrumentation is required for the remaining guardrails and should verify that removal events are populated reliably.
- **Analysis:** intention-to-treat comparison of proportions with a two-sided z-test; report absolute lift, relative lift, 95% confidence interval, and pre-specified segment interactions.
- **Heterogeneous effects:** examine pre-registered interactions for new versus returning users, price band, and entry category. Treat these as secondary and adjust for multiple comparisons.
- **Rollout:** 5% instrumentation check, 20% powered pilot, then broader rollout only if the primary metric improves without material guardrail harm.

## Power calculation

- Observed eligible sessions in the analysis sample: **1,987**
- Users represented by their first eligible session: **1,643**
- Historical later-same-session conversion proxy: **18.20%**
- Minimum detectable effect: **10% relative** (1.82% absolute)
- Significance level: **5%**
- Power: **80%**
- Required users per arm: **7,326**
- Required total users: **14,652**

The historical proxy uses one first eligible session per user and asks whether a purchase occurred later in that same session, so repeat sessions do not inflate the rate. It is not a measured treatment effect or a long-term user conversion rate. Production planning should recompute power using the final outcome window, current traffic, expected eligibility, repeat exposure assumptions, and the actual business-relevant MDE.
