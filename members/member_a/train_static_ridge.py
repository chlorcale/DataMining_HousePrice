"""Train a small pre-sale-feature Ridge model and export browser coefficients.

This is a separate, intentionally restricted experiment for the static Pages UI.
It uses only five house attributes that can normally be known before a sale; it
does not replace the full 86-feature Ridge result used for the group comparison.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from members.member_a import common as ma
from src.protocol import make_cv

FEATURES = ["gr_liv_area", "overall_qual", "bedroom_abvgr", "full_bath", "year_built"]
FEATURE_LABELS = {
    "gr_liv_area": {"label": "Above-ground living area", "unit": "sq ft"},
    "overall_qual": {"label": "Overall material and finish quality", "unit": "1-10"},
    "bedroom_abvgr": {"label": "Bedrooms above ground", "unit": "count"},
    "full_bath": {"label": "Full bathrooms", "unit": "count"},
    "year_built": {"label": "Year built", "unit": "year"},
}
ALPHAS = np.logspace(-8, 4, 25).tolist()


def make_model(alpha=1.0):
    pipe = Pipeline([
        ("preprocess", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])),
        ("ridge", Ridge(alpha=alpha)),
    ])
    return ma.wrap_target(pipe, log=True)


def main():
    started = time.perf_counter()
    X_dev, y_dev, X_test, y_test = ma.load_dev_test()
    X_dev, X_test = X_dev[FEATURES], X_test[FEATURES]
    search = GridSearchCV(
        make_model(), {"regressor__ridge__alpha": ALPHAS}, cv=make_cv(),
        scoring="neg_mean_absolute_error", n_jobs=-1, refit=False,
        return_train_score=False,
    )
    search.fit(X_dev, y_dev)
    search_results = pd.DataFrame(search.cv_results_)
    search_results["cv_mae_usd"] = -search_results["mean_test_score"]
    search_results["cv_mae_std_usd"] = search_results["std_test_score"]
    ma.save_csv(search_results.sort_values("rank_test_score"), "static_ridge_cv_results.csv")
    alpha = float(search.best_params_["regressor__ridge__alpha"])
    cv = ma.cv_folds_table(make_model(alpha), X_dev, y_dev)
    model = make_model(alpha).fit(X_dev, y_dev)
    predictions = np.asarray(model.predict(X_test), dtype=float)
    holdout = ma.holdout_metrics(y_test, predictions)

    fitted = model.regressor_
    prep = fitted.named_steps["preprocess"]
    imputer = prep.named_steps["impute"]
    scaler = prep.named_steps["scale"]
    ridge = fitted.named_steps["ridge"]
    imputed_test = imputer.transform(X_test)
    browser_log_predictions = ridge.intercept_ + (
        (imputed_test - scaler.mean_) / scaler.scale_) @ ridge.coef_
    browser_predictions = np.expm1(browser_log_predictions)
    parity_max_abs_usd = float(np.max(np.abs(browser_predictions - predictions)))
    if not np.allclose(browser_predictions, predictions, rtol=1e-10, atol=1e-6):
        raise RuntimeError(f"Browser formula differs from sklearn pipeline by {parity_max_abs_usd} USD")
    browser_model = {
        "model_name": "Ames pre-sale Ridge (5 numeric features)",
        "model_version": "1.0.0",
        "target_transform": "log1p",
        "target_inverse": "expm1",
        "currency": "USD",
        "dataset": "Ames Housing; historical transactions in Ames, Iowa",
        "dataset_rows": 2930,
        "split": {"development_rows": int(len(y_dev)), "holdout_rows": int(len(y_test)),
                  "holdout_seed": 42, "cv_folds": 5, "cv_seed": 123},
        "features": [{"name": key, **FEATURE_LABELS[key],
                      "median": float(imputer.statistics_[i]),
                      "mean": float(scaler.mean_[i]),
                      "scale": float(scaler.scale_[i])}
                     for i, key in enumerate(FEATURES)],
        "intercept": float(ridge.intercept_),
        "coefficients": [float(v) for v in ridge.coef_],
        "alpha": alpha,
        "cv_mae_usd_mean": cv["mae"]["mean"],
        "cv_mae_usd_std": cv["mae"]["std"],
        **holdout,
        "browser_parity_max_abs_usd": parity_max_abs_usd,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Separate restricted-feature Ridge experiment for the static web; not the full 86-feature member comparison.",
            "Historical Ames, Iowa prices in USD; not a Vietnam market estimator or current market quote.",
            "Only five numeric house features are used; location and many property details are omitted.",
            "Single random holdout estimate; no prediction interval is provided.",
            "Naive expm1 inverse of a log target is not bias-corrected for the original-price mean.",
        ],
    }
    out_dir = ROOT / "docs" / "model"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ridge_model.json").write_text(
        json.dumps(browser_model, indent=2, ensure_ascii=False), encoding="utf-8")
    ma.save_json("static_ridge_metrics.json", {
        "model": browser_model["model_name"],
        "preprocessing": "Five raw numeric pre-sale features; median imputation and StandardScaler fit within each CV fold.",
        "selection_policy": "GridSearchCV minimizes 5-fold development MAE in USD on log1p target.",
        "search_space": {"alpha": ALPHAS},
        "tuning": {"method": "GridSearchCV", "cv": ma.CV_DESC},
        "best_params": {"alpha": alpha, "target_transform": "log1p"},
        "cv_mae": {"mean": cv["mae"]["mean"], "std": cv["mae"]["std"]},
        "cv_table": cv, "n_dev": len(y_dev), "n_test": len(y_test),
        **holdout,
        "feature_names": FEATURES,
        "limitations": browser_model["limitations"],
    })
    print(json.dumps({"alpha": alpha, "cv_mae_usd": browser_model["cv_mae_usd_mean"],
                      "test_mae": holdout["test_mae"], "test_rmse": holdout["test_rmse"],
                      "test_r2": holdout["test_r2"],
                      "browser_artifact": str((out_dir / "ridge_model.json").relative_to(ROOT))}, indent=2))


if __name__ == "__main__":
    main()
