import json
import joblib
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import ARTIFACTS, MODEL_PATH

def build_pipeline(class_weight=None):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42,
                                          class_weight=class_weight)),
    ])

def evaluate(y_true, probabilities, threshold=0.50):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }, predictions

def train_and_compare(X, y):
    ARTIFACTS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    baseline = build_pipeline()
    baseline.fit(X_train, y_train)
    baseline_prob = baseline.predict_proba(X_test)[:, 1]
    baseline_metrics, baseline_pred = evaluate(y_test, baseline_prob, 0.50)

    improved = build_pipeline(class_weight="balanced")
    improved.fit(X_train, y_train)
    improved_prob = improved.predict_proba(X_test)[:, 1]
    improved_metrics, improved_pred = evaluate(y_test, improved_prob, 0.50)

    joblib.dump(improved, MODEL_PATH)
    payload = {
        "dataset_rows": int(len(X)), "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)), "positive_cases": int(y.sum()),
        "negative_cases": int((1-y).sum()),
        "baseline": baseline_metrics, "improved": improved_metrics,
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(payload, indent=2))

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, baseline_pred, ax=ax[0], colorbar=False)
    ax[0].set_title("Baseline")
    ConfusionMatrixDisplay.from_predictions(y_test, improved_pred, ax=ax[1], colorbar=False)
    ax[1].set_title("Improved: balanced classes")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "confusion_matrices.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    RocCurveDisplay.from_predictions(y_test, baseline_prob, name="Baseline", ax=ax)
    RocCurveDisplay.from_predictions(y_test, improved_prob, name="Improved", ax=ax)
    fig.tight_layout(); fig.savefig(ARTIFACTS / "roc_curves.png", dpi=180); plt.close(fig)
    return payload
