-- BigQuery reference queries for a warehouse table named `project.dataset.events`.
-- They mirror the Python metric definitions but are not used to fabricate local results.

-- Daily active users and event depth.
SELECT
  DATE(event_time) AS event_date,
  COUNT(DISTINCT user_id) AS dau,
  COUNT(*) AS events,
  COUNTIF(event_type = 'view') AS views,
  COUNTIF(event_type = 'cart') AS carts,
  COUNTIF(event_type = 'purchase') AS purchase_events,
  SAFE_DIVIDE(COUNTIF(event_type = 'view'), COUNT(DISTINCT user_id)) AS views_per_user
FROM `project.dataset.events`
GROUP BY event_date
ORDER BY event_date;

-- Chronological session funnel. A later step counts only when it follows the prior step.
WITH session_steps AS (
  SELECT
    user_id,
    user_session,
    MIN(IF(event_type = 'view', event_time, NULL)) AS first_view_at,
    MIN(IF(event_type = 'cart', event_time, NULL)) AS first_cart_at,
    MIN(IF(event_type = 'purchase', event_time, NULL)) AS first_purchase_at
  FROM `project.dataset.events`
  GROUP BY user_id, user_session
)
SELECT
  COUNTIF(first_view_at IS NOT NULL) AS view_sessions,
  COUNTIF(first_cart_at >= first_view_at) AS cart_after_view_sessions,
  COUNTIF(first_purchase_at >= first_view_at) AS purchase_after_view_sessions,
  COUNTIF(first_purchase_at >= first_cart_at) AS purchase_after_cart_sessions,
  SAFE_DIVIDE(
    COUNTIF(first_cart_at >= first_view_at),
    COUNTIF(first_view_at IS NOT NULL)
  ) AS view_to_cart_rate,
  SAFE_DIVIDE(
    COUNTIF(first_purchase_at >= first_view_at),
    COUNTIF(first_view_at IS NOT NULL)
  ) AS view_to_purchase_rate,
  SAFE_DIVIDE(
    COUNTIF(first_purchase_at >= first_cart_at),
    COUNTIF(first_cart_at >= first_view_at)
  ) AS cart_to_purchase_rate
FROM session_steps;

-- Observed-window D1 and D7 retention. First seen is not verified acquisition.
WITH activity AS (
  SELECT DISTINCT user_id, DATE(event_time) AS activity_date
  FROM `project.dataset.events`
),
cohorts AS (
  SELECT user_id, MIN(activity_date) AS cohort_date
  FROM activity
  GROUP BY user_id
),
retention AS (
  SELECT
    cohorts.cohort_date,
    DATE_DIFF(activity.activity_date, cohorts.cohort_date, DAY) AS day_number,
    COUNT(DISTINCT activity.user_id) AS retained_users
  FROM activity
  JOIN cohorts USING (user_id)
  WHERE DATE_DIFF(activity.activity_date, cohorts.cohort_date, DAY) IN (0, 1, 7)
  GROUP BY cohort_date, day_number
)
SELECT
  cohort_date,
  day_number,
  retained_users,
  SAFE_DIVIDE(
    retained_users,
    MAX(IF(day_number = 0, retained_users, NULL)) OVER (PARTITION BY cohort_date)
  ) AS retention_rate
FROM retention
ORDER BY cohort_date, day_number;
