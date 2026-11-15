import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score,
                             roc_auc_score, ConfusionMatrixDisplay,
                             RocCurveDisplay)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     cross_validate, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .config import (ARTIFACTS, CV_FOLDS, MIN_RECALL, MODEL_PATH,
                     RANDOM_STATE)

THRESHOLD_GRID = np.round(np.arange(5, 96) / 100, 2)
F2_TIE_TOLERANCE = 0.005  # F2 differences below this are treated as equivalent
ISOTONIC_MIN_BRIER_GAIN = 0.005
MAX_AUC_LOSS = 0.02


def build_pipeline(class_weight=None):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE,
                                          class_weight=class_weight)),
    ])


def _cv():
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def build_candidate(name):
    """Candidates share one preprocessing pipeline; calibration is fitted by internal CV."""
    if name == "baseline":
        return build_pipeline()
    if name == "balanced":
        return build_pipeline("balanced")
    method = name.split("_", 1)[1]
    return CalibratedClassifierCV(estimator=build_pipeline("balanced"),
                                  method=method, cv=_cv())


def evaluate_model(y_true, probabilities, threshold=0.50):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "f2": float(fbeta_score(y_true, predictions, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }, predictions


def perform_cross_validation(X_train, y_train):
    scoring = {m: m for m in ["accuracy", "precision", "recall", "f1",
                              "roc_auc", "balanced_accuracy"]}
    results = {}
    for name in ["baseline", "balanced"]:
        scores = cross_validate(build_candidate(name), X_train, y_train,
                                cv=_cv(), scoring=scoring)
        results[name] = {m: {"mean": float(scores[f"test_{m}"].mean()),
                             "std": float(scores[f"test_{m}"].std())}
                         for m in scoring}
    return results


def select_class_weighting(cv_results):
    """Prefer balanced weighting if it raises mean CV recall without materially losing ROC-AUC."""
    base, bal = cv_results["baseline"], cv_results["balanced"]
    auc_loss = base["roc_auc"]["mean"] - bal["roc_auc"]["mean"]
    if bal["recall"]["mean"] > base["recall"]["mean"] and auc_loss <= MAX_AUC_LOSS:
        return "balanced"
    return "baseline"


def oof_probabilities(estimator, X_train, y_train):
    return cross_val_predict(estimator, X_train, y_train, cv=_cv(),
                             method="predict_proba")[:, 1]


def select_calibration(oof_by_candidate, y_train, weighting):
    """Lowest out-of-fold Brier score; isotonic must beat sigmoid by a margin (small sample)."""
    briers = {n: float(brier_score_loss(y_train, p)) for n, p in oof_by_candidate.items()}
    sigmoid, isotonic, plain = briers[f"{weighting}_sigmoid"], briers[f"{weighting}_isotonic"], briers[weighting]
    choice = f"{weighting}_sigmoid"
    if isotonic <= sigmoid - ISOTONIC_MIN_BRIER_GAIN:
        choice = f"{weighting}_isotonic"
    if plain <= briers[choice]:
        choice = weighting
    return choice, briers


def select_threshold(df, min_recall=MIN_RECALL, tolerance=F2_TIE_TOLERANCE):
    """Maximise F2 among thresholds with recall >= min_recall.

    Thresholds within `tolerance` of the best F2 are treated as equivalent and the one
    with the highest precision wins (then the higher threshold). If none meets the recall
    constraint, the highest-recall candidate is returned and the constraint is flagged unmet.
    """
    eligible = df[df["recall"] >= min_recall]
    if eligible.empty:
        best = df.sort_values(["recall", "f2", "threshold"], ascending=False).iloc[0]
        return float(best["threshold"]), False
    near_best = eligible[eligible["f2"] >= eligible["f2"].max() - tolerance]
    best = near_best.sort_values(["precision", "threshold"], ascending=False).iloc[0]
    return float(best["threshold"]), True


def analyze_thresholds(y_true, probabilities):
    rows = [evaluate_model(y_true, probabilities, t)[0] for t in THRESHOLD_GRID]
    df = pd.DataFrame(rows).drop(columns=["confusion_matrix"])
    df.to_csv(ARTIFACTS / "threshold_analysis.csv", index=False)

    threshold, constraint_met = select_threshold(df)
    chosen = df[df["threshold"] == threshold].iloc[0]

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, label in [("precision", "Precision"), ("recall", "Recall"),
                       ("f1", "F1"), ("f2", "F2")]:
        ax.plot(df["threshold"], df[col], label=label, linewidth=2)
    ax.axvline(threshold, color="k", linestyle="--", label=f"Selected {threshold:.2f}")
    ax.axhline(MIN_RECALL, color="grey", linestyle=":", label=f"Min recall {MIN_RECALL}")
    ax.set(xlabel="Probability threshold", ylabel="Score", ylim=(0, 1.05),
           title="Out-of-fold threshold analysis (development partition)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(ARTIFACTS / "threshold_analysis.png", dpi=180); plt.close(fig)
    return threshold, constraint_met, chosen


def plot_calibration(oof_by_candidate, y_train, test_curves, y_test):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for name, p in oof_by_candidate.items():
        frac, mean = calibration_curve(y_train, p, n_bins=8, strategy="quantile")
        axes[0].plot(mean, frac, "o-", label=name)
    axes[0].set_title("Out-of-fold (training partition)")
    for name, p in test_curves.items():
        frac, mean = calibration_curve(y_test, p, n_bins=5, strategy="quantile")
        axes[1].plot(mean, frac, "o-", label=name)
    axes[1].set_title("Untouched test set (original vs selected)")
    for ax in axes:
        ax.plot([0, 1], [0, 1], "k--", label="Perfect")
        ax.set(xlabel="Mean predicted probability", xlim=(0, 1), ylim=(0, 1))
        ax.legend(); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Observed fraction of repayment difficulty")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "calibration_curves.png", dpi=180); plt.close(fig)


def train_and_compare(X, y):
    ARTIFACTS.mkdir(exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    # Development phase: training partition only
    cv_results = perform_cross_validation(X_train, y_train)
    (ARTIFACTS / "cross_validation_results.json").write_text(json.dumps(cv_results, indent=2))
    weighting = select_class_weighting(cv_results)

    names = [weighting, f"{weighting}_sigmoid", f"{weighting}_isotonic"]
    oof = {n: oof_probabilities(build_candidate(n), X_train, y_train) for n in names}
    selected_name, oof_briers = select_calibration(oof, y_train, weighting)
    if selected_name != f"{weighting}_sigmoid":
        raise ValueError(f"Threshold methodology requires sigmoid calibration, but "
                         f"development Brier selection chose {selected_name!r}.")
    calibration_dev = {
        n: {"oof_brier": oof_briers[n], "oof_roc_auc": float(roc_auc_score(y_train, p))}
        for n, p in oof.items()}

    threshold, constraint_met, row = analyze_thresholds(y_train, oof[selected_name])
    threshold_dev = evaluate_model(y_train, oof[selected_name], threshold)[0]
    if not constraint_met:
        print(f"WARNING: no threshold reached recall >= {MIN_RECALL}; "
              f"using the highest-recall candidate ({threshold:.2f}).")

    # Final phase: refit on the full development partition, evaluate the test set once
    final_model = build_candidate(selected_name).fit(X_train, y_train)
    baseline = build_candidate("baseline").fit(X_train, y_train)
    balanced = build_candidate("balanced").fit(X_train, y_train)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    baseline_prob = baseline.predict_proba(X_test)[:, 1]
    balanced_prob = balanced.predict_proba(X_test)[:, 1]

    final_metrics, final_pred = evaluate_model(y_test, test_prob, threshold)
    calibrated_050, _ = evaluate_model(y_test, test_prob, 0.50)
    baseline_metrics, baseline_pred = evaluate_model(y_test, baseline_prob, 0.50)
    balanced_metrics, _ = evaluate_model(y_test, balanced_prob, 0.50)

    joblib.dump(final_model, MODEL_PATH)

    payload = {
        "dataset_rows": int(len(X)), "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)), "positive_cases": int(y.sum()),
        "negative_cases": int((1 - y).sum()),
        "protocol": "Model, calibration and threshold chosen from development-partition "
                    "cross-validation only; the test set is scored once by the final "
                    "pipeline at the validation-selected threshold. The 0.50 rows are "
                    "fixed reference comparisons and were not used for selection.",
        "original_baseline_test_at_0.50": baseline_metrics,
        "class_balanced_candidate_test_at_0.50": balanced_metrics,
        "sigmoid_calibrated_candidate_test_at_0.50": calibrated_050,
        "cross_validation": cv_results,
        "calibration_development": calibration_dev,
        "selected_model": {
            "class_weighting": weighting, "pipeline": selected_name,
            "rationale": "Balanced weighting selected if mean CV recall is higher and mean "
                         f"CV ROC-AUC falls by no more than {MAX_AUC_LOSS}. Calibration "
                         "chosen by lowest out-of-fold Brier score; isotonic must beat "
                         f"sigmoid by {ISOTONIC_MIN_BRIER_GAIN} Brier because 600 development "
                         "rows make isotonic unstable.",
        },
        "validation_selected_threshold": {
            "threshold": threshold,
            "selection_rule": f"On sigmoid-calibrated out-of-fold development probabilities "
                              f"over thresholds 0.05-0.95 (step 0.01): require recall >= {MIN_RECALL}; "
                              f"maximise F2; thresholds within {F2_TIE_TOLERANCE} F2 of the best are "
                              "equivalent and the highest-precision one is chosen; if no threshold "
                              "meets the recall floor, the highest-recall candidate is used and "
                              "the constraint is recorded as unmet.",
            "min_recall": MIN_RECALL,
            "recall_constraint_met": constraint_met,
            "out_of_fold_metrics": threshold_dev,
            "status": "proof-of-concept operating assumption; requires validation by "
                      "underwriting, risk, compliance and model-governance stakeholders",
        },
        "final_test": {
            "applied_threshold": threshold,
            "selected_model_at_validation_threshold": final_metrics,
        },
        "note": "The ML classification threshold only evaluates model classification "
                "performance. The advisory bands (LOWER_THRESHOLD/UPPER_THRESHOLD in "
                "config.py, 0.35/0.60) route the human-review workflow and are unchanged.",
    }
    (ARTIFACTS / "metrics.json").write_text(json.dumps(payload, indent=2))

    plot_calibration(oof, y_train, {"original baseline": baseline_prob,
                                    f"selected ({selected_name})": test_prob}, y_test)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, baseline_pred, ax=ax[0], colorbar=False)
    ax[0].set_title("Original baseline @0.50")
    ConfusionMatrixDisplay.from_predictions(y_test, final_pred, ax=ax[1], colorbar=False)
    ax[1].set_title(f"Selected @{threshold:.2f}")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "confusion_matrices.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(y_test, final_pred, ax=ax, colorbar=False)
    ax.set_title(f"Final test @ threshold {threshold:.2f}")
    fig.tight_layout(); fig.savefig(ARTIFACTS / "confusion_matrix_final.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    RocCurveDisplay.from_predictions(y_test, baseline_prob, name="Original baseline", ax=ax)
    RocCurveDisplay.from_predictions(y_test, test_prob, name="Selected", ax=ax)
    fig.tight_layout(); fig.savefig(ARTIFACTS / "roc_curves.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    RocCurveDisplay.from_predictions(y_test, test_prob, name="Selected (sigmoid)", ax=ax)
    fig.tight_layout(); fig.savefig(ARTIFACTS / "roc_curve_final.png", dpi=180); plt.close(fig)
    return payload
