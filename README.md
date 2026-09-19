# Customer Retention Decision Engine
## Churn Prediction + Calibration + Lift + Drift + Causal Inference

This is a synthetic banking portfolio PoC designed to demonstrate decision-oriented data science.

## Business questions

1. Which customers are at the highest risk of churning?
2. Can an interpretable predictive model create useful targeting lift?
3. Are predicted probabilities reasonably calibrated?
4. How should model drift be monitored?
5. Did a non-random retention campaign actually reduce churn?
6. What changes when treatment assignment is confounded?

## Part 1 — Churn prediction

Models included:

- Logistic Regression
- Random Forest
- Gradient Boosting

The primary model is Logistic Regression because the goal is not only ranking customers but also producing interpretable risk signals.

### Evaluation

The project measures:

- ROC-AUC
- PR-AUC
- Precision
- Recall
- F1
- Brier Score
- Calibration Curve
- Risk-decile lift

### Leakage controls

The synthetic dataset intentionally includes fields that would create target leakage:

- `account_closed`
- `post_churn_balance`
- `churn_reason`

These are excluded from training.

This demonstrates a key production principle:

> A model can look excellent for the wrong reason if features contain information that becomes available only after the outcome.

## Part 2 — Calibration

The notebook/script compares predicted churn probability with observed churn rate.

For retention decisions, calibration matters because a score of `0.30` should ideally mean approximately a 30% observed event rate among similar cases.

## Part 3 — Lift / targeting

Customers are sorted by predicted churn risk and split into deciles.

The output shows:

- churn rate by decile
- average predicted risk
- lift versus the population average

This is more commercially useful than ROC-AUC alone because it answers:

> If the bank can contact only the highest-risk 10–20% of customers, how concentrated is churn in that group?

## Part 4 — Model drift

Population Stability Index (PSI) is calculated for selected features.

Interpretation used in this PoC:

- `< 0.10` = stable
- `0.10–0.25` = moderate shift
- `> 0.25` = significant shift

PSI is a monitoring heuristic, not a universal statistical law. Thresholds should be governed by the model-risk framework and business context.

## Part 5 — Propensity-score matching

Campaign assignment is intentionally non-random.

Higher-risk customers are more likely to receive a retention intervention.

Therefore:

`Churn(treated) - Churn(untreated)`

is a biased causal estimate.

The project:

1. estimates propensity scores
2. restricts observations to common support
3. performs nearest-neighbor matching
4. checks covariate balance
5. estimates the treatment effect among treated customers

### Limitation

Propensity methods adjust only for **observed confounders**.

Unobserved confounding may remain.

## Part 6 — Difference-in-Differences

The project also demonstrates Difference-in-Differences using pre/post transaction activity.

Conceptually:

`DiD = (Treated_post - Treated_pre) - (Control_post - Control_pre)`

The critical assumption is that, absent treatment, treated and control groups would have followed sufficiently similar trends.

With only one pre-period in this small PoC, the parallel-trends assumption cannot be properly validated. In a real analysis, multiple pre-treatment periods would be examined.

## Run

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt

python generate_data.py
python churn_model.py
python causal_analysis.py
```
