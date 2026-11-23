"""Shared training utilities for Member A (Ridge and Random Forest)."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import make_scorer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import read_dataset
from src.evaluation import metrics
from src.protocol import holdout_indices, make_cv

RESULTS_DIR = ROOT / "results" / "member_a"
FIGURES_DIR = ROOT / "report" / "figures"
SEED_ESTIMATOR = 42
SEED_SEARCH = 42
CV_DESC = "KFold(n_splits=5, shuffle=True, random_state=123) over development rows"


def load_dev_test():
    """Load raw Ames predictors and apply only the frozen shared row split."""
    X, y = read_dataset()
    dev_idx, test_idx = holdout_indices(len(y))
    return X.iloc[dev_idx], y.iloc[dev_idx], X.iloc[test_idx], y.iloc[test_idx]


def mape_pct(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100)


USD_SCORERS = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "mape": make_scorer(mape_pct, greater_is_better=False),
}


def wrap_target(regressor, log=True):
    if log:
        return TransformedTargetRegressor(regressor=regressor, func=np.log1p,
                                          inverse_func=np.expm1, check_inverse=False)
    return TransformedTargetRegressor(regressor=regressor)


def cv_folds_table(model, X, y):
    from sklearn.model_selection import cross_validate
    out = cross_validate(model, X, y, cv=make_cv(), scoring=USD_SCORERS)
    result = {}
    for key, vals in out.items():
        if key.startswith("test_"):
            name = key[5:]
            arr = np.asarray(vals, float)
            if name != "r2":
                arr = -arr
            result[name] = {"per_fold": [float(v) for v in arr],
                            "mean": float(arr.mean()), "std": float(arr.std(ddof=1))}
    return result


def holdout_metrics(y_true, y_pred):
    """Shared holdout metric names, all computed on the original USD scale."""
    values = metrics(y_true, y_pred)
    return {"test_mae": values["mae"], "test_rmse": values["rmse"],
            "test_r2": values["r2"], "test_mape_pct": values["mape_pct"],
            "test_rmsle": values["rmsle"]}


def regression_metrics(y_true, y_pred):
    """Return metrics on the original USD scale for reporting and inference."""
    return metrics(y_true, y_pred)


def library_versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__}


def save_json(name, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {path.relative_to(ROOT)}", flush=True)


def save_csv(frame, name):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RESULTS_DIR / name, index=False)
