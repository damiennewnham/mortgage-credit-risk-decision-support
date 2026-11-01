# Mortgage Credit-Risk Decision-Support Prototype

Proof-of-concept Python system for estimating 12-month repayment difficulty from the supplied synthetic mortgage data, then combining the probability with explicit referral rules. It supports a human underwriter and does not approve or reject mortgages automatically.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Train, evaluate and save the improved model

```bash
python -m src.main train
```

Outputs are written to `artifacts/`.

## Assess an example record

```bash
python -m src.main assess example_application.json
```

## Run tests

```bash
pytest -q
```

## Design summary

- Target: `repayment_difficulty_12m`
- Model: regularised logistic regression
- Preprocessing: median imputation and standard scaling within a scikit-learn pipeline
- Improvement: balanced class weighting to reduce the original model's tendency to miss repayment-difficulty cases
- Logic: probability bands plus documented referral rules; all outputs require human review

## Boundaries

The data and thresholds are synthetic and educational. The prototype is not a real affordability assessment, legal decision engine, or automated approval/rejection system.
