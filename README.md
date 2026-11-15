# Mortgage Credit-Risk Decision-Support Prototype

Proof-of-concept Python system for estimating 12-month repayment difficulty from the supplied synthetic mortgage data, then combining the probability with explicit referral rules. It supports a human underwriter and does not approve or reject mortgages automatically.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

## Usage

### Train, evaluate and save the improved model

```bash
python -m src.main train
```

Outputs are written to `artifacts/`. The training process includes:
- Stratified train/test split (75% train, 25% test)
- 5-fold cross-validation on training data
- Out-of-fold probability predictions for threshold analysis
- Threshold sensitivity analysis across 0.05 to 0.95
- Calibration comparison (sigmoid vs isotonic)
- Final evaluation on untouched test set

### Assess an example record

```bash
python -m src.main assess example_application.json
```

Invalid JSON or missing/invalid fields produce interpretable errors without calling the model.

### Run tests

```bash
pytest -q
```

## Design summary

- **Target variable**: `repayment_difficulty_12m` (binary: 0 = no difficulty, 1 = repayment difficulty within 12 months)
- **Model**: Regularised logistic regression with class weight balancing
- **Preprocessing**: Median imputation for missing values (scikit-learn SimpleImputer), then standard scaling (StandardScaler)
- **Class handling**: Balanced class weighting to reduce bias towards the majority class
- **Decision logic**: Probability bands plus documented policy referral rules; all outputs require human review

## Input Validation

The system validates all input before model prediction using central bounds defined in `src/config.py`:

### Validation rules by field

| Field | Type | Min | Max | Notes |
|-------|------|-----|-----|-------|
| `loan_to_value_pct` | float | 0 | 100 | Loan-to-value percentage |
| `debt_to_income_ratio` | float | 0 | 2.0 | Debt-to-income ratio |
| `credit_score` | float | 300 | 900 | Credit score |
| `employment_years` | float | 0 | 100 | Years of employment |
| `savings_gbp` | float | 0 | ∞ | Savings in GBP |
| `missed_payments_24m` | int | 0 | 100 | Missed payments in 24 months (integer only) |
| `credit_utilisation_pct` | float | 0 | 100 | Credit utilisation percentage |
| `previous_default_flag` | int | {0, 1} | - | Previous default (0 or 1 only) |
| `term_years` | float | 1 | 50 | Mortgage term in years |
| `interest_rate_pct` | float | 0 | 20 | Interest rate percentage |

### Detected validation errors

- Missing required fields
- `None`, empty strings, or non-numeric values for numeric fields
- NaN (not a number) or infinity values
- Negative values where impossible (credit score, employment years, savings, missed payments, interest rate, term)
- Values exceeding documented maximums
- Fractional values for integer-only fields (`missed_payments_24m`, `previous_default_flag`)
- Boolean values (use numeric 0/1 instead)

Invalid applications return:
```json
{
  "category": "Unable to assess",
  "reason": "Invalid application input",
  "validation_errors": [
    "credit_score: Credit score must be >= 300, got -100.0",
    "employment_years: Years of employment must be >= 0, got -5.0"
  ],
  "human_review": true
}
```

**No invalid application ever reaches the trained model's `predict_proba` method.**

## Model Selection & Training Approach

### 1. Data split strategy

1. **Stratified train/test split**: 75% training (600 records), 25% final test (200 records), stratified by target
2. **Training data only**: Used for model selection, cross-validation, and threshold decisions
3. **Final test set**: Completely untouched until final evaluation

### 2. Cross-validation results

Using 5-fold stratified cross-validation on training data (600 records):

#### Baseline (unweighted logistic regression)
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.8183 | 0.0322 |
| Precision | 0.6661 | 0.1465 |
| Recall | 0.3510 | 0.0882 |
| F1 | 0.4555 | 0.1008 |
| Balanced Accuracy | 0.6499 | 0.0506 |
| ROC-AUC | 0.7793 | 0.0544 |

#### Improved (class weight = "balanced")
| Metric | Mean | Std |
|--------|------|-----|
| Accuracy | 0.6983 | 0.0396 |
| Precision | 0.3921 | 0.0552 |
| Recall | **0.6789** | 0.0910 |
| F1 | 0.4955 | 0.0624 |
| Balanced Accuracy | 0.6913 | 0.0490 |
| ROC-AUC | 0.7789 | 0.0563 |

**Selection rationale**: The improved model significantly increases recall (0.3510 → 0.6789), meaning it identifies 93% more cases of repayment difficulty. This is critical in a decision-support context where missing a difficult case is far more damaging than a false positive requiring additional review. The trade-off is reduced precision and lower overall accuracy, but these are acceptable costs in a human-supervised workflow.

### 3. Threshold analysis

Using out-of-fold predictions from 5-fold CV (not on final test data):

- **Probabilities used**: sigmoid-calibrated out-of-fold probabilities of the final (balanced + sigmoid) pipeline on the 600-row development partition, so the threshold is on the same scale as the saved model. Training raises a clear error if sigmoid is not the calibration chosen.
- **Threshold range**: 0.05 to 0.95 in 0.01 increments (`artifacts/threshold_analysis.csv`, `.png`)
- **Exact rule**: require recall >= `MIN_RECALL` (0.60, `src/config.py`); maximise F2; thresholds within 0.005 F2 of the best (`F2_TIE_TOLERANCE`) are treated as equivalent and the highest-precision one is chosen. If no threshold reaches the recall floor, the highest-recall candidate is used and `recall_constraint_met: false` is recorded in `metrics.json`.
- **Selected threshold**: 0.12. Out-of-fold: precision 0.2965, recall 0.9008, F1 0.4461, F2 0.6399, TP/FP/FN/TN = 118/280/13/189.
- **Applied once** to the untouched test set; the 0.50 results are fixed reference comparisons only.
- **Grid precision**: adjacent thresholds differ in F2 by <= ~0.01, which is below the sampling noise of 131 positives, so 0.01 steps are sufficient; F2 is nearly flat (0.63-0.64) between 0.08 and 0.15.
- **Caveat**: F2 is almost flat across a wide low-threshold plateau, so 0.12 is one of several near-equivalent choices. It flags about 68% of development cases, which implies a heavy review workload.
- **`MIN_RECALL = 0.60` is a proof-of-concept operating assumption** that needs validation by underwriting, risk, compliance and model-governance stakeholders. It is not a universally correct rule.

**ML classification threshold vs advisory bands**:
- The ML classification threshold (0.12) is used only to evaluate classification performance.
- The advisory probability bands (0.35 lower, 0.60 upper) in `src/decisions.py` route the human-review workflow and are unchanged.

### 4. Probability calibration

Calibration is selected from out-of-fold training predictions only (calibrators are fitted inside each fold via `CalibratedClassifierCV`, cv=5):

| Candidate | OOF Brier | OOF ROC-AUC |
|-----------|-----------|-------------|
| Balanced, uncalibrated | 0.1915 | 0.7758 |
| Balanced + sigmoid | **0.1390** | 0.7748 |
| Balanced + isotonic | 0.1410 | 0.7671 |

**Selection rationale**: sigmoid has the lowest Brier score. With 600 training rows isotonic is unstable, so it must beat sigmoid by 0.005 Brier to be chosen; it did not. Sigmoid is therefore the more defensible choice. The balanced class weight inflates raw probabilities, which is why the uncalibrated Brier score is much worse.

**Final model**: balanced logistic regression with sigmoid calibration, refitted on the full training partition.

## Final Test Set Results (Untouched, scored once)

| Metric | Original baseline @0.50 | Balanced uncalibrated @0.50 | Sigmoid-calibrated @0.50 | Selected (sigmoid) @0.12 | Previous rule @0.54 |
|--------|------|------|------|------|------|
| Accuracy | 0.8100 | 0.6900 | 0.8000 | 0.5550 | 0.8000 |
| Balanced accuracy | 0.6416 | 0.6707 | 0.5944 | 0.6984 | 0.5781 |
| Precision | 0.6250 | 0.3784 | 0.6250 | 0.3256 | 0.6667 |
| Recall | 0.3409 | 0.6364 | 0.2273 | 0.9545 | 0.1818 |
| F1 | 0.4412 | 0.4746 | 0.3333 | 0.4855 | 0.2857 |
| F2 | 0.3750 | 0.5600 | 0.2604 | 0.6885 | 0.2128 |
| ROC-AUC | 0.7685 | 0.7692 | 0.7705 | 0.7705 | 0.7705 |
| Brier | 0.1432 | 0.1971 | 0.1416 | 0.1416 | 0.1416 |
| TP / FP / FN / TN | 15 / 9 / 29 / 147 | 28 / 46 / 16 / 110 | 10 / 6 / 34 / 150 | 42 / 87 / 2 / 69 | 8 / 4 / 36 / 152 |

The recall floor was met (development recall 0.90; test recall 0.95). The trade-off is 87 false positives among 156 non-difficulty cases, so precision falls to 0.33 and accuracy to 0.555. ROC-AUC and Brier are threshold-independent and unchanged. Test recall rests on only 44 positives, so it is a noisy estimate.

## Generated Artifacts

After training, the following are saved to `artifacts/`:

1. **`mortgage_risk_model.joblib`** – Complete fitted pipeline (imputation → scaling → sigmoid-calibrated logistic regression)
2. **`metrics.json`** – All performance metrics including cross-validation, threshold analysis, calibration comparison, and final test results
3. **`cross_validation_results.json`** – Detailed 5-fold CV statistics for both baseline and improved models
4. **`threshold_analysis.csv`** – Out-of-fold performance across 91 thresholds (precision, recall, F1, F2, FP, FN, etc.)
5. **`threshold_analysis.png`** – Graph of precision, recall, F1, F2 vs threshold
6. **`calibration_curves.png`** – Out-of-fold calibration curves for the candidates, plus test-set curves for original vs selected
7. **`confusion_matrix_final.png`** / **`roc_curve_final.png`** – Selected model on the test set at the validation-selected threshold
8. **`confusion_matrices.png`** / **`roc_curves.png`** – Original baseline vs selected model on the test set

## Decision Output Format

All decision outputs require human review and include:

```json
{
  "category": "string",
  "reason": "string explaining the decision",
  "probability": 0.xxxx,
  "human_review": true
}
```

### Categories and logic

1. **"Lower risk: continue human review"**
   - Probability < 0.35 AND no policy referral rules triggered
   
2. **"Further review recommended"**
   - Probability 0.35–0.60 OR policy referral rule triggered (debt_to_income ≥ 0.60 or loan_to_value ≥ 90)
   
3. **"Higher risk: detailed human review required"**
   - Probability ≥ 0.60 OR hard adverse-credit rule triggered (previous_default_flag=1 AND missed_payments_24m ≥ 2)

All paths require human review. The model assists decision-making but does not make autonomous approval/rejection decisions.

## Boundaries and Limitations

### This is a proof-of-concept educational system

- **Data**: Synthetic 800-record dataset. Not representative of real mortgages.
- **Model scope**: Binary classification (repayment difficulty vs no difficulty) using only 10 features
- **No legal status**: Not a complete FCA affordability assessment or regulatory compliance framework
- **No automation**: Human underwriter retains final authority on all applications
- **Threshold values**: Advisory bands (0.35/0.60) are proof-of-concept only, not authorised lender policy
- **Predictions**: Reflect training data only; real-world performance would differ significantly

### When to seek expert review

- Applications with missing or inconsistent information
- Cases with unusual feature combinations
- Regulatory changes affecting affordability rules
- Model retraining when using new data

## Reproducing the results

All results are deterministic given the fixed random state (42):

```bash
# Full pipeline
python -m src.main train

# View metrics
cat artifacts/metrics.json

# Validate model on test application
python -m src.main assess example_application.json

# Run all tests (47 total)
pytest -q
```

Expected test results: all 47 tests pass (data, validation/decision and threshold/CLI-guard tests).

## Implementation notes

- **Preprocessing pipeline**: Instantiated fresh for each cross-validation fold; fitted only on training data
- **Stratification**: Used throughout (train/test split, 5-fold CV) to maintain class proportions
- **Validation**: Occurs in `src/validation.py` before model prediction; invalid applications short-circuit at decision stage
- **Calibration**: Fitted by internal cross-validation on training data only; selected by out-of-fold Brier score, never by the test set
- **Cross-validation**: 5-fold CV; results reported as mean ± std
- **JSON error handling**: Graceful degradation for malformed input files

## File structure

```
├── src/
│   ├── config.py          # Configuration, feature names, validation bounds
│   ├── data.py            # Dataset loading and splitting
│   ├── validation.py      # Input validation (NEW)
│   ├── model.py           # Training, cross-validation, calibration
│   ├── decisions.py       # Advisory decision logic
│   └── main.py            # CLI entry point
├── tests/
│   ├── test_data.py       # Dataset tests
│   ├── test_decisions.py  # Validation and decision tests (EXTENDED)
│   └── test_threshold.py  # Threshold rule and CLI predict_proba guard
├── artifacts/             # Generated models and metrics
├── data/
│   └── mortgage_data.csv  # Input dataset
├── example_application.json
├── requirements.txt
└── README.md (this file)
```

## Summary of improvements

| Requirement | Implementation |
|------------|-----------------|
| Input validation | `src/validation.py` with bounds in `config.py`; 42 validation tests |
| Model selection | 5-fold CV comparing baseline vs balanced weighting |
| Threshold analysis | OOF predictions of the final pipeline, 91 thresholds, F2 subject to recall >= 0.60 (ties to highest precision) |
| Probability calibration | Sigmoid vs isotonic vs none; out-of-fold Brier score selection |
| Final test evaluation | Single evaluation on untouched 200-record test set |
| Error handling | Graceful JSON/validation error responses in CLI |
| Comprehensive metrics | JSON output with cross-validation, calibration, and test results |
| Reproducibility | Fixed random state; all commands generate identical results |

