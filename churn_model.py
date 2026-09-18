from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score,
    f1_score, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "retention_customers.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# -----------------------------
# Leakage controls
# -----------------------------
LEAKAGE_COLUMNS = ["account_closed", "post_churn_balance", "churn_reason"]
TARGET = "churn_90d"
ID_COLS = ["customer_id"]
TREATMENT_COL = "retention_campaign"
DID_COLS = ["pre_txn_rate", "post_txn_rate"]

features = [
    "age","income","tenure_months","product_count","transactions_30d",
    "app_sessions_30d","avg_balance","complaints_90d","salary_credit",
    "card_utilization","recent_inactivity_days","balance_change_90d","region"
]

X = df[features].copy()
y = df[TARGET].copy()

categorical = ["region"]
numeric = [c for c in features if c not in categorical]

preprocess = ColumnTransformer([
    ("num", StandardScaler(), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])

models = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=250, max_depth=8, min_samples_leaf=20,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(random_state=42)
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

rows = []
pred_store = {}

for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    prob = pipe.predict_proba(X_test)[:,1]
    pred = (prob >= 0.5).astype(int)
    pred_store[name] = (pipe, prob)

    rows.append({
        "model": name,
        "roc_auc": roc_auc_score(y_test, prob),
        "pr_auc": average_precision_score(y_test, prob),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "brier_score": brier_score_loss(y_test, prob)
    })

metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
metrics.to_csv(OUT / "model_metrics.csv", index=False)

# Primary interpretable model = logistic regression
logit_pipe, logit_prob = pred_store["logistic_regression"]

# -----------------------------
# Lift / decile table
# -----------------------------
lift = X_test.copy()
lift["actual_churn"] = y_test.values
lift["predicted_risk"] = logit_prob
lift["risk_decile"] = pd.qcut(
    lift["predicted_risk"].rank(method="first", ascending=False),
    10, labels=range(1,11)
)

overall_rate = lift["actual_churn"].mean()
deciles = (
    lift.groupby("risk_decile", observed=True)
    .agg(customers=("actual_churn","size"),
         churn_rate=("actual_churn","mean"),
         avg_predicted_risk=("predicted_risk","mean"))
    .reset_index()
)
deciles["lift_vs_average"] = deciles["churn_rate"] / overall_rate
deciles.to_csv(OUT / "lift_deciles.csv", index=False)

# -----------------------------
# Calibration
# -----------------------------
prob_true, prob_pred = calibration_curve(y_test, logit_prob, n_bins=10, strategy="quantile")
cal = pd.DataFrame({"mean_predicted": prob_pred, "observed_rate": prob_true})
cal.to_csv(OUT / "calibration.csv", index=False)

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(prob_pred, prob_true, marker="o", label="Logistic Regression")
ax.plot([0,1],[0,1], linestyle="--", label="Perfect calibration")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed churn rate")
ax.set_title("Calibration Curve")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "calibration_curve.png", dpi=160)
plt.close(fig)

# -----------------------------
# Coefficient interpretation
# -----------------------------
prep = logit_pipe.named_steps["prep"]
feature_names = list(numeric)
ohe = prep.named_transformers_["cat"]
feature_names += list(ohe.get_feature_names_out(categorical))
coefs = logit_pipe.named_steps["model"].coef_[0]
coef_df = pd.DataFrame({
    "feature": feature_names,
    "coefficient": coefs,
    "odds_ratio": np.exp(coefs)
}).sort_values("coefficient", ascending=False)
coef_df.to_csv(OUT / "logistic_coefficients.csv", index=False)

# -----------------------------
# PSI drift monitoring
# -----------------------------
def psi(expected, actual, bins=10):
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    quantiles = np.unique(np.quantile(expected, np.linspace(0,1,bins+1)))
    if len(quantiles) < 3:
        return 0.0

    quantiles[0] = -np.inf
    quantiles[-1] = np.inf

    e_counts, _ = np.histogram(expected, bins=quantiles)
    a_counts, _ = np.histogram(actual, bins=quantiles)

    e_pct = np.maximum(e_counts / len(expected), 1e-6)
    a_pct = np.maximum(a_counts / len(actual), 1e-6)

    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

# Simulate later-period population shift
rng = np.random.default_rng(123)
later = df.sample(9000, random_state=123).copy()
later["app_sessions_30d"] = np.clip(later["app_sessions_30d"] * 0.72 + rng.normal(0,1.5,len(later)), 0, None)
later["recent_inactivity_days"] = later["recent_inactivity_days"] * 1.35
later["avg_balance"] = later["avg_balance"] * 0.95

drift_features = ["income","transactions_30d","app_sessions_30d","avg_balance","recent_inactivity_days"]
drift_rows = []
for col in drift_features:
    score = psi(df[col].values, later[col].values)
    if score < 0.10:
        level = "stable"
    elif score < 0.25:
        level = "moderate_shift"
    else:
        level = "significant_shift"
    drift_rows.append({"feature": col, "psi": score, "interpretation": level})

drift = pd.DataFrame(drift_rows).sort_values("psi", ascending=False)
drift.to_csv(OUT / "psi_drift.csv", index=False)

# Executive summary
top_decile = deciles.sort_values("risk_decile").iloc[0]
summary = f"""
CHURN MODEL EXECUTIVE SUMMARY
=============================

Primary model: Logistic Regression
Reason: interpretability, probability output, and suitability for decision support.

Test ROC-AUC: {metrics.loc[metrics.model=='logistic_regression','roc_auc'].iloc[0]:.3f}
Test PR-AUC: {metrics.loc[metrics.model=='logistic_regression','pr_auc'].iloc[0]:.3f}
Brier score: {metrics.loc[metrics.model=='logistic_regression','brier_score'].iloc[0]:.3f}

Top-risk decile:
Observed churn rate: {top_decile.churn_rate:.2%}
Lift vs population average: {top_decile.lift_vs_average:.2f}x

Leakage columns intentionally excluded:
{", ".join(LEAKAGE_COLUMNS)}

Monitoring:
PSI generated for key features with stable/moderate/significant shift labels.
"""
(OUT / "model_executive_summary.txt").write_text(summary.strip(), encoding="utf-8")
print(summary)
print("\nModel metrics:\n", metrics.round(4).to_string(index=False))
print("\nTop risk deciles:\n", deciles.head(5).round(4).to_string(index=False))
print("\nDrift:\n", drift.round(4).to_string(index=False))
