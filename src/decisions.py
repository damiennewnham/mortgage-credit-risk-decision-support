from .config import LOWER_THRESHOLD, UPPER_THRESHOLD
from .validation import validate_application

def advisory_decision(probability, application):
    """
    Generate advisory decision based on validated application and model probability.
    
    Always requires human review. Invalid applications are not sent to the model.
    """
    is_valid, errors = validate_application(application)
    if not is_valid:
        return {
            "category": "Unable to assess",
            "reason": "Invalid application input",
            "validation_errors": errors,
            "human_review": True,
        }
    
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
    
    return {
        "category": category,
        "reason": reason,
        "probability": round(float(probability), 4),
        "human_review": True,
    }
