import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
rng = np.random.default_rng(SEED)
ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

N = 30000

customer_id = np.arange(200001, 200001 + N)
age = np.clip(rng.normal(40, 11, N).round(), 18, 75).astype(int)
income = np.clip(rng.lognormal(np.log(72000), 0.55, N), 18000, 300000)
tenure_months = np.clip(rng.gamma(3.2, 22, N).round(), 1, 240).astype(int)
product_count = np.clip(rng.poisson(1.8, N) + 1, 1, 6)
transactions_30d = np.clip(rng.poisson(18, N), 0, 90)
app_sessions_30d = np.clip(rng.poisson(9, N), 0, 50)
avg_balance = np.clip(rng.lognormal(np.log(12000), 0.9, N), 100, 250000)
complaints_90d = rng.binomial(2, 0.08, N)
salary_credit = rng.binomial(1, 0.64, N)
card_utilization = np.clip(rng.beta(2.2, 3.8, N), 0, 1)
recent_inactivity_days = np.clip(rng.gamma(2.0, 9.0, N).round(), 0, 120).astype(int)
balance_change_90d = rng.normal(0, 0.18, N)
region = rng.choice(["South","North","East","West"], N, p=[0.30,0.26,0.22,0.22])

# Baseline churn propensity
logit = (
    -3.2
    + 0.035 * recent_inactivity_days
    + 0.55 * complaints_90d
    + 0.85 * np.maximum(-balance_change_90d, 0)
    + 0.35 * card_utilization
    - 0.035 * app_sessions_30d
    - 0.09 * product_count
    - 0.004 * tenure_months
    - 0.30 * salary_credit
)
p_churn = 1 / (1 + np.exp(-logit))

# Non-random retention campaign targeting:
# high-risk customers are more likely to be contacted.
campaign_logit = (
    -2.0
    + 0.045 * recent_inactivity_days
    + 0.45 * complaints_90d
    - 0.02 * app_sessions_30d
    + 0.30 * card_utilization
)
p_campaign = 1 / (1 + np.exp(-campaign_logit))
retention_campaign = rng.binomial(1, np.clip(p_campaign, 0.03, 0.85))

# Treatment effect reduces churn probability, but targeting creates confounding.
treatment_effect = -0.55 * retention_campaign
p_churn_after = 1 / (1 + np.exp(-(logit + treatment_effect)))
churn_90d = rng.binomial(1, p_churn_after)

# "Leakage" fields intentionally added to show why they must be excluded.
account_closed = ((churn_90d == 1) & (rng.random(N) < 0.82)).astype(int)
post_churn_balance = np.where(
    churn_90d == 1,
    np.maximum(0, avg_balance * rng.uniform(0.0, 0.25, N)),
    avg_balance * rng.uniform(0.8, 1.2, N)
)
churn_reason = np.where(
    churn_90d == 1,
    rng.choice(["fees","service","inactive","competitor"], N),
    "none"
)

# Before/after monthly engagement metric for Difference-in-Differences.
# Treated group has worse baseline because targeting isn't random.
pre_txn_rate = np.clip(
    transactions_30d
    + rng.normal(0, 3, N)
    - 2.5 * retention_campaign,
    0, None
)
post_txn_rate = np.clip(
    pre_txn_rate
    + rng.normal(0, 3, N)
    - 1.0
    + 3.0 * retention_campaign,
    0, None
)

df = pd.DataFrame({
    "customer_id": customer_id,
    "age": age,
    "income": income.round(2),
    "tenure_months": tenure_months,
    "product_count": product_count,
    "transactions_30d": transactions_30d,
    "app_sessions_30d": app_sessions_30d,
    "avg_balance": avg_balance.round(2),
    "complaints_90d": complaints_90d,
    "salary_credit": salary_credit,
    "card_utilization": card_utilization.round(4),
    "recent_inactivity_days": recent_inactivity_days,
    "balance_change_90d": balance_change_90d.round(4),
    "region": region,
    "retention_campaign": retention_campaign,
    "churn_90d": churn_90d,
    "account_closed": account_closed,
    "post_churn_balance": post_churn_balance.round(2),
    "churn_reason": churn_reason,
    "pre_txn_rate": pre_txn_rate.round(2),
    "post_txn_rate": post_txn_rate.round(2)
})

df.to_csv(DATA / "retention_customers.csv", index=False)
print(f"Saved {len(df):,} rows")
print("Observed churn by campaign (confounded):")
print(df.groupby("retention_campaign")["churn_90d"].mean().round(4))
