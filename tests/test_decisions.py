import pytest
import math
from unittest.mock import patch, MagicMock
from src.validation import validate_application
from src.decisions import advisory_decision

BASE_VALID = {
    "loan_to_value_pct": 75.0,
    "debt_to_income_ratio": 0.30,
    "credit_score": 700.0,
    "employment_years": 5.0,
    "savings_gbp": 10000.0,
    "missed_payments_24m": 0,
    "credit_utilisation_pct": 30.0,
    "previous_default_flag": 0,
    "term_years": 25,
    "interest_rate_pct": 5.0,
}

class TestValidationBoundary:
    """Test boundary and edge cases for validation."""
    
    def test_valid_complete_application(self):
        is_valid, errors = validate_application(BASE_VALID)
        assert is_valid
        assert len(errors) == 0
    
    def test_valid_boundary_credit_score_minimum(self):
        app = BASE_VALID | {"credit_score": 300}
        is_valid, errors = validate_application(app)
        assert is_valid
    
    def test_valid_boundary_credit_score_maximum(self):
        app = BASE_VALID | {"credit_score": 900}
        is_valid, errors = validate_application(app)
        assert is_valid
    
    def test_valid_boundary_employment_years_zero(self):
        app = BASE_VALID | {"employment_years": 0}
        is_valid, errors = validate_application(app)
        assert is_valid
    
    def test_valid_boundary_savings_zero(self):
        app = BASE_VALID | {"savings_gbp": 0}
        is_valid, errors = validate_application(app)
        assert is_valid
    
    def test_valid_boundary_term_minimum(self):
        app = BASE_VALID | {"term_years": 1}
        is_valid, errors = validate_application(app)
        assert is_valid
    
    def test_valid_boundary_interest_rate_zero(self):
        app = BASE_VALID | {"interest_rate_pct": 0}
        is_valid, errors = validate_application(app)
        assert is_valid

class TestValidationNegativeValues:
    """Test rejection of implausible negative values."""
    
    def test_negative_credit_score(self):
        app = BASE_VALID | {"credit_score": -100}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("credit_score" in e and "must be >=" in e for e in errors)
    
    def test_negative_employment_years(self):
        app = BASE_VALID | {"employment_years": -5}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("employment_years" in e and "must be >=" in e for e in errors)
    
    def test_negative_savings(self):
        app = BASE_VALID | {"savings_gbp": -1000}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("savings_gbp" in e and "must be >=" in e for e in errors)
    
    def test_negative_missed_payments(self):
        app = BASE_VALID | {"missed_payments_24m": -1}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("missed_payments_24m" in e and "must be >=" in e for e in errors)
    
    def test_negative_interest_rate(self):
        app = BASE_VALID | {"interest_rate_pct": -2}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("interest_rate_pct" in e and "must be >=" in e for e in errors)
    
    def test_negative_term_years(self):
        app = BASE_VALID | {"term_years": -10}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("term_years" in e and "must be >=" in e for e in errors)

class TestValidationExceedsMaximum:
    """Test rejection of values exceeding documented maximums."""
    
    def test_credit_score_above_maximum(self):
        app = BASE_VALID | {"credit_score": 950}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("credit_score" in e and "must be <=" in e for e in errors)
    
    def test_employment_years_above_maximum(self):
        app = BASE_VALID | {"employment_years": 150}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("employment_years" in e and "must be <=" in e for e in errors)
    
    def test_loan_to_value_above_maximum(self):
        app = BASE_VALID | {"loan_to_value_pct": 150}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("loan_to_value_pct" in e and "must be <=" in e for e in errors)
    
    def test_debt_to_income_above_maximum(self):
        app = BASE_VALID | {"debt_to_income_ratio": 2.5}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("debt_to_income_ratio" in e and "must be <=" in e for e in errors)
    
    def test_interest_rate_above_maximum(self):
        app = BASE_VALID | {"interest_rate_pct": 25}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("interest_rate_pct" in e and "must be <=" in e for e in errors)
    
    def test_credit_utilisation_above_maximum(self):
        app = BASE_VALID | {"credit_utilisation_pct": 150}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("credit_utilisation_pct" in e and "must be <=" in e for e in errors)

class TestValidationFractionalValues:
    """Test rejection of fractional values for integer fields."""
    
    def test_fractional_missed_payments(self):
        app = BASE_VALID | {"missed_payments_24m": 2.5}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("missed_payments_24m" in e and "must be an integer" in e for e in errors)
    
    def test_fractional_previous_default_flag(self):
        app = BASE_VALID | {"previous_default_flag": 0.5}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("previous_default_flag" in e for e in errors)

class TestValidationPreviousDefaultFlag:
    """Test strict validation of previous default flag."""
    
    def test_previous_default_valid_0(self):
        app = BASE_VALID | {"previous_default_flag": 0}
        is_valid, errors = validate_application(app)
        assert is_valid
    
    def test_previous_default_valid_1(self):
        app = BASE_VALID | {"previous_default_flag": 1}
        is_valid, errors = validate_application(app)
        assert is_valid
    
    def test_previous_default_invalid_2(self):
        app = BASE_VALID | {"previous_default_flag": 2}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("previous_default_flag" in e and "must be one of" in e for e in errors)
    
    def test_previous_default_invalid_negative(self):
        app = BASE_VALID | {"previous_default_flag": -1}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("previous_default_flag" in e for e in errors)

class TestValidationSpecialFloatValues:
    """Test rejection of NaN and infinity."""
    
    def test_nan_credit_score(self):
        app = BASE_VALID | {"credit_score": float('nan')}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("credit_score" in e and "NaN" in e for e in errors)
    
    def test_positive_infinity_savings(self):
        app = BASE_VALID | {"savings_gbp": float('inf')}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("savings_gbp" in e and "infinity" in e for e in errors)
    
    def test_negative_infinity_debt_to_income(self):
        app = BASE_VALID | {"debt_to_income_ratio": float('-inf')}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("debt_to_income_ratio" in e and "infinity" in e for e in errors)

class TestValidationStringValues:
    """Test rejection of string values for numeric fields."""
    
    def test_string_credit_score(self):
        app = BASE_VALID | {"credit_score": "seven hundred"}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("credit_score" in e and ("string" in e or "cannot convert" in e) for e in errors)
    
    def test_numeric_string_savings_is_accepted(self):
        app = BASE_VALID | {"savings_gbp": "10000"}
        is_valid, errors = validate_application(app)
        assert is_valid

class TestValidationMissingFields:
    """Test rejection of missing required fields."""
    
    def test_missing_credit_score(self):
        app = BASE_VALID.copy()
        del app["credit_score"]
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("credit_score" in e and "Missing" in e for e in errors)
    
    def test_missing_multiple_fields(self):
        app = BASE_VALID.copy()
        del app["credit_score"]
        del app["employment_years"]
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert len(errors) >= 2
    
    def test_none_value_for_credit_score(self):
        app = BASE_VALID | {"credit_score": None}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("credit_score" in e and "cannot be None" in e for e in errors)
    
    def test_empty_string_for_employment_years(self):
        app = BASE_VALID | {"employment_years": ""}
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert any("employment_years" in e and "empty string" in e for e in errors)

class TestValidationMultipleErrors:
    """Test that multiple errors are returned together."""
    
    def test_multiple_simultaneous_validation_errors(self):
        app = BASE_VALID | {
            "credit_score": -100,
            "employment_years": -5,
            "savings_gbp": -1000,
            "missed_payments_24m": 2.5,
            "previous_default_flag": 5,
        }
        is_valid, errors = validate_application(app)
        assert not is_valid
        assert len(errors) >= 5

class TestValidationWithMocking:
    """Test that validation prevents predict_proba from being called."""
    
    def test_invalid_application_does_not_call_predict_proba(self):
        app = BASE_VALID | {"credit_score": -100}
        is_valid, errors = validate_application(app)
        
        assert not is_valid
        
        with patch('src.decisions.validate_application', return_value=(is_valid, errors)):
            result = advisory_decision(0.5, app)
            assert result["category"] == "Unable to assess"
            assert "validation_errors" in result

class TestDecisionsIntegration:
    """Test decisions with validation integration."""
    
    def test_low_risk_case_with_valid_application(self):
        result = advisory_decision(0.20, BASE_VALID)
        assert result["category"].startswith("Lower risk")
        assert "validation_errors" not in result
    
    def test_moderate_probability_with_valid_application(self):
        result = advisory_decision(0.45, BASE_VALID)
        assert result["category"] == "Further review recommended"
    
    def test_high_probability_with_valid_application(self):
        result = advisory_decision(0.75, BASE_VALID)
        assert result["category"].startswith("Higher risk")
    
    def test_invalid_application_returns_unable_to_assess(self):
        app = BASE_VALID | {"credit_score": -100}
        result = advisory_decision(0.5, app)
        assert result["category"] == "Unable to assess"
        assert "validation_errors" in result
    
    def test_policy_referral_with_high_dti(self):
        app = BASE_VALID | {"debt_to_income_ratio": 0.65}
        result = advisory_decision(0.20, app)
        assert result["category"] == "Further review recommended"
    
    def test_adverse_credit_rule(self):
        app = BASE_VALID | {
            "previous_default_flag": 1,
            "missed_payments_24m": 2,
        }
        result = advisory_decision(0.20, app)
        assert result["category"].startswith("Higher risk")

