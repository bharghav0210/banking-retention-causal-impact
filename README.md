
# Customer Retention Intelligence & Causal Impact Modeling

## Overview

This project demonstrates an end-to-end customer retention analytics workflow for a retail-banking use case using synthetic customer data.

The objective is to answer two different but related business questions:

1. **Prediction:** Which customers are most likely to churn?
2. **Causal impact:** Does a retention intervention actually reduce churn or improve customer engagement?

The project combines interpretable machine learning, model evaluation, calibration, business lift analysis, leakage controls, model drift monitoring, propensity-score matching, and Difference-in-Differences.

The analysis is built on **30,000 synthetic banking customers** and is intended as a portfolio demonstration of decision-oriented data science rather than a production banking model.

---

## Business Context

Customer churn creates direct commercial impact through:

- lost product revenue
- lower customer lifetime value
- increased reacquisition cost
- reduced cross-sell opportunities
- declining engagement

A bank therefore needs to solve two separate problems.

### Problem 1 — Risk Identification

Identify customers with elevated churn risk so retention resources can be prioritized efficiently.

### Problem 2 — Intervention Measurement

Determine whether a retention campaign actually causes an improvement in customer outcomes.

These problems require different analytical approaches.

> Predictive modeling answers **who is likely to churn**.  
> Causal analysis evaluates **whether an intervention changes customer outcomes**.

---

## Project Objectives

This PoC demonstrates the following capabilities:

- interpretable churn modeling
- model discrimination and ranking
- probability calibration
- risk-decile lift analysis
- target-leakage prevention
- feature-drift monitoring
- propensity-score estimation
- nearest-neighbor matching
- covariate-balance diagnostics
- treatment-effect estimation
- Difference-in-Differences
- analytical limitations and causal assumptions

---

## Dataset

The project generates a synthetic dataset containing **30,000 customers**.

Example customer features include:

- age
- income
- tenure
- product count
- monthly transaction activity
- mobile-app engagement
- average balance
- complaints
- salary-credit indicator
- credit-card utilization
- inactivity
- recent balance movement
- region

The dataset also includes:

- retention-campaign exposure
- 90-day churn outcome
- pre-intervention transaction activity
- post-intervention transaction activity

Synthetic post-outcome fields are deliberately included to demonstrate leakage detection.

---

## Project Workflow


Synthetic Customer Data
        |
        v
Data Validation
        |
        +------------------------------+
        |                              |
        v                              v
Churn Prediction                Campaign Evaluation
        |                              |
        v                              v
Logistic Regression             Propensity Scores
Random Forest                   Common Support
Gradient Boosting               Nearest-Neighbor Matching
        |                              |
        v                              v
ROC-AUC / PR-AUC                Covariate Balance
Calibration                    ATT Estimation
Lift Analysis                         |
Leakage Checks                        v
PSI Drift                       Difference-in-Differences
        |                              |
        +--------------+---------------+
                       |
                       v
              Decision-Oriented Insights


---

# 1. Churn Prediction

Three classification models are evaluated:

* Logistic Regression
* Random Forest
* Gradient Boosting

### Primary model

**Logistic Regression** is used as the main decision-support model.

The choice is intentional.

The project prioritizes:

* interpretability
* probability outputs
* business explainability
* transparent risk drivers

rather than selecting a more complex model solely for marginal predictive gains.

---

## Model Evaluation

The models are evaluated using:

* ROC-AUC
* PR-AUC
* Precision
* Recall
* F1-score
* Brier Score
* Calibration Curve
* Risk-Decile Lift

### Validated Results

| Metric                     | Logistic Regression |
| -------------------------- | ------------------: |
| ROC-AUC                    |           **0.644** |
| PR-AUC                     |           **0.077** |
| Brier Score                |           **0.226** |
| Top-risk decile churn rate |          **10.00%** |
| Lift vs population average |           **2.83x** |

The ROC-AUC indicates moderate discriminatory power.

However, the model provides stronger value from a targeting perspective.

The highest-risk 10% of customers exhibit approximately **2.83 times the average churn rate**, making the model useful for prioritizing limited retention resources.

---

## Why Lift Matters

Traditional model metrics such as ROC-AUC measure overall ranking quality.

The business question is often more practical:

> If the retention team can contact only 10% of customers, how much churn risk is concentrated in that group?

Risk-decile lift answers this directly.

Customers are ranked by predicted churn probability and divided into ten equally sized groups.

The highest-risk decile achieved:

**2.83x lift over the population average**

This means targeted outreach can focus on a substantially higher-risk customer segment.

---

# 2. Leakage Controls

Target leakage can produce misleadingly strong model performance.

The synthetic dataset deliberately contains post-outcome variables such as:

```text
account_closed
post_churn_balance
churn_reason
```

These variables are explicitly excluded from model training.

They would not be available at the point when the bank needs to make a retention decision.

This demonstrates an important production principle:

> Model features must reflect information available at prediction time.

A model with leakage may perform extremely well during development while failing completely in real deployment.

---

# 3. Probability Calibration

For retention decisions, ranking customers is not always enough.

Predicted probabilities should also have meaningful interpretation.

For example:

```text
Predicted churn probability = 0.30
```

should ideally correspond to approximately a 30% observed churn rate among customers with similar predicted risk.

Calibration is evaluated using:

* Brier Score
* Calibration Curve

The output is available at:

```text
outputs/calibration.csv
outputs/calibration_curve.png
```

Calibration should be assessed separately from discrimination.

A model may rank customers effectively while still producing poorly calibrated probabilities.

---

# 4. Model Drift Monitoring

Customer behavior changes over time.

For example:

* mobile-app usage may fall
* transaction frequency may change
* balances may shift
* customer inactivity may increase

The project uses **Population Stability Index (PSI)** as a simple feature-distribution monitoring mechanism.

### PSI interpretation used in this PoC

|       PSI | Interpretation    |
| --------: | ----------------- |
|    < 0.10 | Stable            |
| 0.10–0.25 | Moderate Shift    |
|    > 0.25 | Significant Shift |

### Validated Drift Results

| Feature                |        PSI | Interpretation    |
| ---------------------- | ---------: | ----------------- |
| app_sessions_30d       | **1.1016** | Significant shift |
| recent_inactivity_days | **0.1815** | Moderate shift    |
| avg_balance            |     0.0019 | Stable            |
| income                 |     0.0008 | Stable            |
| transactions_30d       |     0.0007 | Stable            |

The simulated future population shows a significant change in digital engagement.

This would be a trigger for further investigation before relying on the existing model unchanged.

### Important Note

PSI thresholds are monitoring heuristics.

They should not be treated as universal statistical cutoffs. In production environments, thresholds should align with model-risk governance, feature importance, business materiality, and historical behavior.

---

# 5. Retention Campaign Evaluation

The second part of the project asks a different question:

> Did the retention campaign actually reduce churn?

Campaign assignment is intentionally **non-random**.

Higher-risk customers have a greater probability of receiving the retention intervention.

This creates **selection bias and confounding**.

The raw comparison is therefore not sufficient for causal interpretation.

---

## Naive Comparison

Observed churn rates:

```text
Untreated customers: 3.68%
Treated customers:   3.06%
```

Naive difference:

**-0.63 percentage points**

At first glance, this could be interpreted as evidence that the campaign reduced churn.

However, the groups are not directly comparable because intervention assignment depends on customer characteristics.

Therefore:

```text
Churn(treated) - Churn(untreated)
```

cannot automatically be interpreted as a causal treatment effect.

---

# 6. Propensity Score Matching

To reduce observed selection bias, the project estimates each customer's probability of receiving the campaign.

The propensity model uses pre-treatment variables such as:

* tenure
* product ownership
* transaction activity
* app engagement
* complaints
* credit-card utilization
* inactivity
* balance behavior
* demographic attributes

### Matching Process

The analysis:

1. estimates propensity scores
2. restricts customers to common support
3. separates treated and untreated customers
4. performs 1:1 nearest-neighbor matching
5. checks covariate balance
6. estimates the Average Treatment Effect on the Treated

---

## Covariate Balance

Matching quality is evaluated using **Standardized Mean Difference (SMD)**.

Validated result:

**Mean absolute SMD after matching = 0.0094**

This indicates very strong observed covariate balance in the synthetic matched sample.

A common rule of thumb considers absolute SMD values below 0.10 as evidence of acceptable balance.

---

## Treatment Effect

Estimated matched ATT:

**-2.23 percentage points**

Interpretation:

> Among customers who received the synthetic retention campaign, matched analysis estimates churn to be approximately 2.23 percentage points lower than for comparable untreated customers.

This result is a methodological demonstration using synthetic data and should not be interpreted as evidence about any real banking campaign.

---

# 7. Difference-in-Differences

The project also evaluates customer transaction activity before and after the intervention.

Average transaction activity:

| Group   |    Pre |   Post | Change |
| ------- | -----: | -----: | -----: |
| Control | 17.990 | 17.005 | -0.985 |
| Treated | 15.538 | 17.552 | +2.014 |

Difference-in-Differences:

```text
(+2.014) - (-0.985)
≈ +3.00 transactions/customer
```

Estimated relative effect:

**+3.00 transactions per customer**

This suggests that treated customers experienced a larger improvement in transaction activity relative to the untreated group.

---

## Key DiD Assumption

Difference-in-Differences relies on the **parallel-trends assumption**.

In the absence of treatment, treated and untreated groups should have followed sufficiently similar trends.

This PoC contains only one pre-intervention period.

Therefore, parallel trends cannot be properly validated.

A production analysis should include multiple pre-treatment periods and test whether historical trends are sufficiently similar before treatment.

---

# 8. Key Findings

The project demonstrates several practical data-science lessons.

### Predictive modeling

* Logistic Regression provides moderate discrimination.
* The highest-risk decile achieves **2.83x lift**.
* Lift is more directly useful for retention targeting than ROC-AUC alone.

### Model governance

* Leakage variables can artificially inflate performance and must be excluded.
* PSI monitoring identifies meaningful changes in customer behavior.
* Significant drift in app engagement would warrant model review.

### Causal analysis

* Raw treated-vs-control comparisons are insufficient when treatment assignment is non-random.
* Propensity-score matching substantially improves observed covariate balance.
* Matched analysis estimates a **-2.23 pp ATT** on churn.
* Difference-in-Differences estimates approximately **+3 transactions/customer** on customer engagement.

---

# 9. Business Interpretation

The project demonstrates how predictive and causal analytics support different decisions.

### Churn model

Used to determine:

> Which customers should the bank prioritize for retention outreach?

### Lift analysis

Used to determine:

> How much risk can be concentrated within a limited intervention budget?

### Causal analysis

Used to determine:

> Does the retention intervention actually improve customer outcomes?

### Drift monitoring

Used to determine:

> Is the customer population changing enough that the model may require review or retraining?

Together, these components form a practical retention decision-support framework.

---

# 10. Limitations

This project intentionally documents its limitations.

### Synthetic data

All customer data and outcomes are generated synthetically.

The results therefore demonstrate analytical methodology, not real-world banking performance.

### Propensity-score methods

Matching can adjust only for observed confounders.

Unobserved confounding may remain.

### Difference-in-Differences

Only one pre-treatment period is available.

The parallel-trends assumption cannot be rigorously validated.

### Model performance

The churn model demonstrates moderate predictive performance rather than production-grade performance.

The objective is to demonstrate disciplined evaluation, targeting lift, interpretability, governance, and decision framing.

### PSI

PSI is used as a monitoring heuristic rather than a formal statistical guarantee of model degradation.

---

# 11. Repository Structure

```text
.
├── data/
│   └── retention_customers.csv
│
├── outputs/
│   ├── calibration.csv
│   ├── calibration_curve.png
│   ├── causal_executive_summary.txt
│   ├── difference_in_differences.csv
│   ├── lift_deciles.csv
│   ├── logistic_coefficients.csv
│   ├── matching_balance.csv
│   ├── model_executive_summary.txt
│   ├── model_metrics.csv
│   ├── propensity_matched_pairs.csv
│   └── psi_drift.csv
│
├── generate_data.py
├── churn_model.py
├── causal_analysis.py
├── requirements.txt
└── README.md
```

---

# 12. Reproducing the Analysis

## Clone the repository

```bash
git clone https://github.com/bharghav0210/banking-retention-causal-impact.git
cd banking-retention-causal-impact
```

## Create a virtual environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Generate the synthetic dataset

```bash
python generate_data.py
```

## Run churn modeling

```bash
python churn_model.py
```

## Run causal analysis

```bash
python causal_analysis.py
```

Generated results are written to:

```text
outputs/
```

---

# 13. Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib

Analytical techniques include:

* Logistic Regression
* Random Forest
* Gradient Boosting
* ROC-AUC
* PR-AUC
* Brier Score
* Calibration Analysis
* Risk-Decile Lift
* Population Stability Index
* Propensity Score Matching
* Standardized Mean Difference
* Difference-in-Differences

---

## Disclaimer

This project uses entirely synthetic data and is intended solely for portfolio, learning, and technical demonstration purposes.

It does not contain proprietary, confidential, or real customer banking data.

