from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "retention_customers.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# ---------------------------------
# Naive comparison (biased)
# ---------------------------------
naive = df.groupby("retention_campaign")["churn_90d"].mean()
naive_diff = naive.loc[1] - naive.loc[0]

# ---------------------------------
# Propensity score estimation
# ---------------------------------
confounders = [
    "age","income","tenure_months","product_count","transactions_30d",
    "app_sessions_30d","avg_balance","complaints_90d","salary_credit",
    "card_utilization","recent_inactivity_days","balance_change_90d","region"
]
categorical = ["region"]
numeric = [c for c in confounders if c not in categorical]

prep = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

ps_model = Pipeline([
    ("prep", prep),
    ("model", LogisticRegression(max_iter=1000))
])

ps_model.fit(df[confounders], df["retention_campaign"])
df["propensity_score"] = ps_model.predict_proba(df[confounders])[:,1]

# Common support trimming
trimmed = df[(df["propensity_score"] >= 0.05) & (df["propensity_score"] <= 0.95)].copy()

treated = trimmed[trimmed["retention_campaign"] == 1].copy()
control = trimmed[trimmed["retention_campaign"] == 0].copy()

# 1:1 nearest-neighbor matching on propensity score
nn = NearestNeighbors(n_neighbors=1)
nn.fit(control[["propensity_score"]])
dist, idx = nn.kneighbors(treated[["propensity_score"]])

matched_control = control.iloc[idx.flatten()].copy().reset_index(drop=True)
matched_treated = treated.reset_index(drop=True)

matched = pd.DataFrame({
    "treated_churn": matched_treated["churn_90d"].values,
    "control_churn": matched_control["churn_90d"].values,
    "treated_ps": matched_treated["propensity_score"].values,
    "control_ps": matched_control["propensity_score"].values,
    "distance": dist.flatten()
})

matched["treatment_effect"] = matched["treated_churn"] - matched["control_churn"]
att = matched["treatment_effect"].mean()

matched.to_csv(OUT / "propensity_matched_pairs.csv", index=False)

# Balance diagnostics using standardized mean differences for numeric variables.
balance_rows = []
for col in numeric:
    t = matched_treated[col].astype(float)
    c = matched_control[col].astype(float)
    pooled_sd = np.sqrt((t.var(ddof=1) + c.var(ddof=1))/2)
    smd = 0.0 if pooled_sd == 0 else (t.mean() - c.mean()) / pooled_sd
    balance_rows.append({
        "feature": col,
        "treated_mean": t.mean(),
        "matched_control_mean": c.mean(),
        "standardized_mean_difference": smd
    })
balance = pd.DataFrame(balance_rows)
balance.to_csv(OUT / "matching_balance.csv", index=False)

# ---------------------------------
# Difference-in-Differences
# ---------------------------------
# Outcome: monthly transaction rate.
grouped = df.groupby("retention_campaign")[["pre_txn_rate","post_txn_rate"]].mean()
treated_change = grouped.loc[1, "post_txn_rate"] - grouped.loc[1, "pre_txn_rate"]
control_change = grouped.loc[0, "post_txn_rate"] - grouped.loc[0, "pre_txn_rate"]
did = treated_change - control_change

did_table = pd.DataFrame([
    {
        "group": "control",
        "pre_mean": grouped.loc[0, "pre_txn_rate"],
        "post_mean": grouped.loc[0, "post_txn_rate"],
        "change": control_change
    },
    {
        "group": "treated",
        "pre_mean": grouped.loc[1, "pre_txn_rate"],
        "post_mean": grouped.loc[1, "post_txn_rate"],
        "change": treated_change
    }
])
did_table.to_csv(OUT / "difference_in_differences.csv", index=False)

summary = f"""
CAUSAL RETENTION ANALYSIS
=========================

Naive observed churn difference
(treated - untreated): {naive_diff*100:.2f} percentage points

Why naive comparison is biased:
Campaign assignment is not random. Higher-risk customers were intentionally
more likely to receive the retention campaign.

Propensity-score matched ATT estimate
(treated churn - matched-control churn): {att*100:.2f} percentage points

Difference-in-Differences outcome:
Monthly transaction-rate effect: {did:.2f} transactions/customer

Key assumptions / limitations:
1. Propensity methods adjust only for observed confounders.
2. Matching quality must be checked using covariate balance.
3. Difference-in-Differences relies on a plausible parallel-trends assumption.
4. Neither method substitutes for randomization when a well-designed experiment is feasible.
5. These results come from synthetic data and demonstrate methodology, not a real bank intervention.
"""
(OUT / "causal_executive_summary.txt").write_text(summary.strip(), encoding="utf-8")
print(summary)
print("\nMean absolute SMD after matching:", balance["standardized_mean_difference"].abs().mean().round(4))
print("\nDiD table:\n", did_table.round(3).to_string(index=False))
