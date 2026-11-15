import json
from unittest.mock import MagicMock, patch

import pandas as pd

from src import main
from src.model import select_threshold


def _table(rows):
    return pd.DataFrame(rows, columns=["threshold", "recall", "precision", "f2"])


def test_threshold_maximises_f2_among_recall_eligible():
    df = _table([(0.2, 0.90, 0.30, 0.65), (0.3, 0.70, 0.40, 0.60),
                 (0.4, 0.55, 0.60, 0.70), (0.5, 0.30, 0.80, 0.40)])
    assert select_threshold(df, min_recall=0.60) == (0.2, True)


def test_threshold_tie_prefers_highest_precision():
    df = _table([(0.2, 0.90, 0.30, 0.640), (0.3, 0.80, 0.38, 0.638),
                 (0.4, 0.62, 0.50, 0.636), (0.5, 0.40, 0.70, 0.50)])
    assert select_threshold(df, min_recall=0.60, tolerance=0.005) == (0.4, True)


def test_unmet_recall_returns_highest_recall_and_flags_it():
    df = _table([(0.6, 0.30, 0.70, 0.35), (0.7, 0.45, 0.60, 0.48), (0.8, 0.10, 0.9, 0.12)])
    assert select_threshold(df, min_recall=0.60) == (0.7, False)


def test_cli_does_not_load_or_call_model_for_invalid_application(tmp_path, capsys):
    application = {"loan_to_value_pct": 75, "debt_to_income_ratio": 0.3, "credit_score": -5,
                   "employment_years": 5, "savings_gbp": 1000, "missed_payments_24m": 0,
                   "credit_utilisation_pct": 30, "previous_default_flag": 0,
                   "term_years": 25, "interest_rate_pct": 5.0}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(application))
    model = MagicMock()
    with patch("src.main.joblib.load", return_value=model) as load:
        main.assess(str(path))
    model.predict_proba.assert_not_called()
    load.assert_not_called()
    output = json.loads(capsys.readouterr().out)
    assert output["category"] == "Unable to assess"
    assert any("credit_score" in e for e in output["validation_errors"])
