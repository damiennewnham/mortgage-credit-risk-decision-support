import argparse
import json
import joblib
import pandas as pd
from .config import DATA_PATH, FEATURES, MODEL_PATH
from .data import load_dataset, split_inputs
from .model import train_and_compare
from .decisions import advisory_decision

def train():
    df = load_dataset(DATA_PATH)
    X, y = split_inputs(df)
    result = train_and_compare(X, y)
    print(json.dumps(result, indent=2))

def assess(application_path):
    model = joblib.load(MODEL_PATH)
    application = json.loads(open(application_path, encoding="utf-8").read())
    row = pd.DataFrame([{name: application.get(name) for name in FEATURES}])
    probability = model.predict_proba(row)[0, 1]
    print(json.dumps(advisory_decision(probability, application), indent=2))

def main():
    parser = argparse.ArgumentParser(description="Mortgage repayment-risk decision support")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    p = sub.add_parser("assess"); p.add_argument("application")
    args = parser.parse_args()
    train() if args.command == "train" else assess(args.application)
if __name__ == "__main__": main()
