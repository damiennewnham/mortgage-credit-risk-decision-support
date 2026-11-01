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
