import math
from .config import FEATURES, VALIDATION_BOUNDS

def validate_application(application):
    """
    Validate application input before model prediction.
    
    Returns: (is_valid, errors_list)
    - is_valid: bool
    - errors_list: list of validation error messages (empty if valid)
    """
    errors = []
    
    for field in FEATURES:
        if field not in application:
            errors.append(f"Missing required field: {field}")
            continue
        
        value = application[field]
        
        if value is None:
            errors.append(f"{field}: required field cannot be None")
            continue
        
        if isinstance(value, str):
            if value == "":
                errors.append(f"{field}: required field cannot be empty string")
                continue
            try:
                numeric_value = float(value)
            except (ValueError, TypeError):
                errors.append(f"{field}: cannot convert to numeric; got string '{value}'")
                continue
        elif isinstance(value, bool):
            errors.append(f"{field}: boolean type not allowed; use numeric 0/1")
            continue
        else:
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                errors.append(f"{field}: cannot convert to numeric; got {type(value).__name__}")
                continue
        
        if math.isnan(numeric_value):
            errors.append(f"{field}: cannot be NaN")
            continue
        
        if math.isinf(numeric_value):
            errors.append(f"{field}: cannot be infinity")
            continue
        
        bounds = VALIDATION_BOUNDS[field]
        
        if "allowed_values" in bounds:
            if numeric_value not in bounds["allowed_values"]:
                errors.append(
                    f"{field}: {bounds['description']} must be one of {bounds['allowed_values']}, "
                    f"got {numeric_value}"
                )
            if numeric_value != int(numeric_value):
                errors.append(f"{field}: {bounds['description']} must be an integer, got {numeric_value}")
        else:
            if "type" in bounds and bounds["type"] == "int":
                if numeric_value != int(numeric_value):
                    errors.append(
                        f"{field}: {bounds['description']} must be an integer, got {numeric_value}"
                    )
            
            min_val = bounds.get("min")
            max_val = bounds.get("max")
            
            if min_val is not None and numeric_value < min_val:
                errors.append(
                    f"{field}: {bounds['description']} must be >= {min_val}, got {numeric_value}"
                )
            
            if max_val is not None and not math.isinf(max_val) and numeric_value > max_val:
                errors.append(
                    f"{field}: {bounds['description']} must be <= {max_val}, got {numeric_value}"
                )
    
    return len(errors) == 0, errors

