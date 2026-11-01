from src.decisions import advisory_decision
BASE = {
 "loan_to_value_pct": 75, "debt_to_income_ratio": .30, "credit_score": 700,
 "employment_years": 5, "savings_gbp": 10000, "missed_payments_24m": 0,
 "credit_utilisation_pct": 30, "previous_default_flag": 0,
 "term_years": 25, "interest_rate_pct": 5.0,
}
def test_low_risk_case():
    assert advisory_decision(.20, BASE)["category"].startswith("Lower risk")
def test_moderate_probability():
    assert advisory_decision(.45, BASE)["category"] == "Further review recommended"
def test_high_probability():
    assert advisory_decision(.75, BASE)["category"].startswith("Higher risk")
def test_policy_referral_overrides_low_probability():
    app = BASE | {"debt_to_income_ratio": .65}
    assert advisory_decision(.20, app)["category"] == "Further review recommended"
def test_adverse_credit_rule():
    app = BASE | {"previous_default_flag": 1, "missed_payments_24m": 2}
    assert advisory_decision(.20, app)["category"].startswith("Higher risk")
def test_missing_input_is_not_scored_normally():
    app = BASE.copy(); del app["credit_score"]
    assert advisory_decision(.20, app)["category"] == "Unable to assess"
