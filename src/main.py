import argparse
import json
import joblib
import pandas as pd
from .config import DATA_PATH, FEATURES, MODEL_PATH
from .data import load_dataset, split_inputs
from .model import train_and_compare
from .decisions import advisory_decision
from .validation import validate_application

def train():
    df = load_dataset(DATA_PATH)
    X, y = split_inputs(df)
    result = train_and_compare(X, y)
    print(json.dumps(result, indent=2))

def assess(application_path):
    try:
        with open(application_path, encoding="utf-8") as f:
            application = json.load(f)
    except json.JSONDecodeError as e:
        result = {
            "category": "Unable to assess",
            "reason": f"Invalid JSON in application file: {str(e)}",
            "human_review": True,
        }
        print(json.dumps(result, indent=2))
        return
    except FileNotFoundError:
        result = {
            "category": "Unable to assess",
            "reason": f"Application file not found: {application_path}",
            "human_review": True,
        }
        print(json.dumps(result, indent=2))
        return
    except Exception as e:
        result = {
            "category": "Unable to assess",
            "reason": f"Error reading application file: {str(e)}",
            "human_review": True,
        }
        print(json.dumps(result, indent=2))
        return
    
    is_valid, errors = validate_application(application)
    if not is_valid:
        result = {
            "category": "Unable to assess",
            "reason": "Invalid application input",
            "validation_errors": errors,
            "human_review": True,
        }
        print(json.dumps(result, indent=2))
        return
    
    try:
        model = joblib.load(MODEL_PATH)
        row = pd.DataFrame([{name: application.get(name) for name in FEATURES}])
        probability = model.predict_proba(row)[0, 1]
        decision = advisory_decision(probability, application)
        print(json.dumps(decision, indent=2))
    except Exception as e:
        result = {
            "category": "Unable to assess",
            "reason": f"Error during model prediction: {str(e)}",
            "human_review": True,
        }
        print(json.dumps(result, indent=2))

def main():
    parser = argparse.ArgumentParser(description="Mortgage repayment-risk decision support")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    p = sub.add_parser("assess")
    p.add_argument("application")
    args = parser.parse_args()
    train() if args.command == "train" else assess(args.application)

if __name__ == "__main__":
    main()

