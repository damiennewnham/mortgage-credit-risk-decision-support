from .config import FEATURES, LOWER_THRESHOLD, UPPER_THRESHOLD

def validate_application(application):
    missing = [name for name in FEATURES if name not in application or application[name] is None]
    if missing:
        return False, f"Missing required inputs: {', '.join(missing)}"
    return True, "Valid"

def advisory_decision(probability, application):
    valid, reason = validate_application(application)
    if not valid:
        return {"category": "Unable to assess", "reason": reason,
                "human_review": True}
    hard_high = (application["previous_default_flag"] == 1 and
                 application["missed_payments_24m"] >= 2)
    policy_referral = (application["debt_to_income_ratio"] >= 0.60 or
                       application["loan_to_value_pct"] >= 90)
    if probability >= UPPER_THRESHOLD or hard_high:
        reason = "High modelled risk or combined adverse-credit rule triggered."
        category = "Higher risk: detailed human review required"
    elif probability >= LOWER_THRESHOLD or policy_referral:
        reason = "Moderate modelled risk or affordability/LTV referral rule triggered."
        category = "Further review recommended"
    else:
        reason = "Lower modelled risk; no additional referral rule triggered."
        category = "Lower risk: continue human review"
    return {"category": category, "reason": reason,
            "probability": round(float(probability), 4), "human_review": True}
