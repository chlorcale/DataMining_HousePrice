"""Shared utilities for Member B (ElasticNet + LightGBM) — CO3029 Ames Housing.

Contract compliance (docs/TEAM_MODEL_CONTRACT.md):
- Model selection uses ONLY the 2344-row development split through the frozen
  5-fold protocol (KFold shuffle, seed 123).
- Preprocessing factories from src.preprocessing are unfitted and refit inside
  each training fold, so validation rows never influence imputers, scalers,
  encoders or category vocabularies.
- All scorers operate on USD: when the log1p target hypothesis is used, the
  wrapper inverts predictions back to USD before any metric.
- The 586-row holdout is evaluated exactly once, after the config freeze.
"""
import json
import platform
import sys
from pathlib import Path

import lightgbm
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

RESULTS_DIR = ROOT / "results" / "member_b"
FIGURES_DIR = ROOT / "report" / "figures"

CV_DESC = "KFold(n_splits=5, shuffle=True, random_state=123) over 2344 development rows"
SEED_ESTIMATOR = 42
SEED_SEARCH = 42


def load_dev_test():
    """Frozen split per contract: dev 2344 rows, holdout 586 rows."""
    X, y = read_dataset()
    dev_idx, test_idx = holdout_indices(len(y))
    return X.iloc[dev_idx], y.iloc[dev_idx], X.iloc[test_idx], y.iloc[test_idx]


def mape_pct(y_true, y_pred):
    """MAPE in percent, guarded against tiny targets (matches src.evaluation)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1.0))) * 100.0)


# All scorers see predictions and targets in USD.
USD_SCORERS = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2",
    "mape": make_scorer(mape_pct, greater_is_better=False),
}


def wrap_target(regressor, log=True):
    """Target hypothesis wrapper; predictions always come back in USD.

    TransformedTargetRegressor is used for BOTH branches so the hyperparameter
    prefix stays uniform ('regressor__estimator__...') during search:
    log=True  -> fit on log1p(SalePrice), inverse-transform before scoring.
    log=False -> identity target (same estimator family, USD scale).
    """
    if log:
        return TransformedTargetRegressor(
            regressor=regressor, func=np.log1p, inverse_func=np.expm1,
            check_inverse=False)
    return TransformedTargetRegressor(regressor=regressor)


def cv_folds_table(model, X, y):
    """Per-fold USD metrics of one frozen configuration via the shared CV."""
    from sklearn.model_selection import cross_validate

    out = cross_validate(model, X, y, cv=make_cv(), scoring=USD_SCORERS)
    table = {}
    for key, vals in out.items():
        if not key.startswith("test_"):
            continue
        name = key[len("test_"):]
        arr = np.asarray(vals, dtype=float)
        if name != "r2":
            arr = -arr  # scorers were negative-oriented
        table[name] = {
            "per_fold": [float(v) for v in arr],
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)),
        }
    return table


def test_block(y_true, y_pred):
    """Final one-shot holdout metrics via the shared src.evaluation function."""
    m = metrics(y_true, y_pred)
    return {"test_mae": m["mae"], "test_rmse": m["rmse"], "test_r2": m["r2"],
            "test_mape_pct": m["mape_pct"], "test_rmsle": m["rmsle"]}


def library_versions():
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
    }


def save_json(filename, payload):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {path.relative_to(ROOT)}", flush=True)


def save_csv(df, filename):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"saved: {path.relative_to(ROOT)}", flush=True)


def save_fig(fig, name):
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, dpi=150)
    plt.close(fig)
    print(f"saved: report/figures/{name}", flush=True)
