from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "mortgage_data.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "mortgage_risk_model.joblib"
TARGET = "repayment_difficulty_12m"
FEATURES = [
    "loan_to_value_pct", "debt_to_income_ratio", "credit_score",
    "employment_years", "savings_gbp", "missed_payments_24m",
    "credit_utilisation_pct", "previous_default_flag", "term_years",
    "interest_rate_pct",
]

LOWER_THRESHOLD = 0.35
UPPER_THRESHOLD = 0.60

VALIDATION_BOUNDS = {
    "loan_to_value_pct": {"min": 0, "max": 100, "description": "Loan-to-value percentage"},
    "debt_to_income_ratio": {"min": 0, "max": 2.0, "description": "Debt-to-income ratio"},
    "credit_score": {"min": 300, "max": 900, "description": "Credit score"},
    "employment_years": {"min": 0, "max": 100, "description": "Years of employment"},
    "savings_gbp": {"min": 0, "max": float('inf'), "description": "Savings in GBP"},
    "missed_payments_24m": {"min": 0, "max": 100, "type": "int", "description": "Missed payments in 24 months"},
    "credit_utilisation_pct": {"min": 0, "max": 100, "description": "Credit utilisation percentage"},
    "previous_default_flag": {"allowed_values": [0, 1], "description": "Previous default flag"},
    "term_years": {"min": 1, "max": 50, "description": "Mortgage term in years"},
    "interest_rate_pct": {"min": 0, "max": 20, "description": "Interest rate percentage"},
}

RANDOM_STATE = 42
MIN_RECALL = 0.60  # proof-of-concept operating assumption for the ML threshold; needs stakeholder validation
CV_FOLDS = 5
