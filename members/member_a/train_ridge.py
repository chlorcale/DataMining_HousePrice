"""Tune Ridge on development data, then score the frozen holdout once."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from members.member_a import common as ma
from src.protocol import make_cv
from src.preprocessing import make_linear_preprocessor

ALPHAS_LOG = np.logspace(-3, 3, 19).tolist()
ALPHAS_ID = np.logspace(0, 6, 19).tolist()
LOG_PREF_TOL = 0.01


def make_model(alpha=1.0, log=True):
    pipe = Pipeline([("preprocess", make_linear_preprocessor()),
                     ("estimator", Ridge(alpha=alpha))])
    return ma.wrap_target(pipe, log=log)


def search(X, y, log, alphas, tag):
    grid = GridSearchCV(make_model(log=log),
                        {"regressor__estimator__alpha": alphas},
                        cv=make_cv(),
                        scoring="neg_mean_absolute_error", n_jobs=-1,
                        refit=False, return_train_score=False)
    start = time.perf_counter()
    grid.fit(X, y)
    elapsed = time.perf_counter() - start
    frame = pd.DataFrame(grid.cv_results_)
    frame["cv_mae"] = -frame.mean_test_score
    frame["cv_mae_std"] = frame.std_test_score
    ma.save_csv(frame.sort_values("rank_test_score"), f"ridge_cv_{tag}.csv")
    best = frame.loc[frame.cv_mae.idxmin()]
    return float(best["param_regressor__estimator__alpha"]), float(best.cv_mae), elapsed


def main():
    start = time.perf_counter()
    Xd, yd, Xt, yt = ma.load_dev_test()
    a_log, mae_log, t_log = search(Xd, yd, True, ALPHAS_LOG, "log")
    a_id, mae_id, t_id = search(Xd, yd, False, ALPHAS_ID, "identity")
    use_log = mae_log <= mae_id * (1 + LOG_PREF_TOL)
    alpha = a_log if use_log else a_id
    model = make_model(alpha=alpha, log=use_log)
    cv = ma.cv_folds_table(model, Xd, yd)
    t0 = time.perf_counter()
    model.fit(Xd, yd)
    fit_seconds = time.perf_counter() - t0
    pred = np.asarray(model.predict(Xt), float)
    test = ma.holdout_metrics(yt, pred)
    fitted_pipe = model.regressor_
    names = fitted_pipe.named_steps["preprocess"].named_steps["pre"].get_feature_names_out()
    coefs = fitted_pipe.named_steps["estimator"].coef_
    top = np.argsort(-np.abs(coefs))[:20]
    payload = {
        "model": "Ridge Regression (scikit-learn)",
        "preprocessing": "FeatureBuilder(86 predictors), numeric median imputation + missing indicators + StandardScaler, categorical Unknown fill + OneHotEncoder; fitted within each CV fold.",
        "selection_policy": f"5-fold development CV MAE in USD; log1p target retained unless identity improves by more than {LOG_PREF_TOL:.0%}.",
        "search_space": {"alpha_log1p": ALPHAS_LOG, "alpha_identity": ALPHAS_ID,
                         "scoring": "negative MAE in USD"},
        "tuning": {"method": "GridSearchCV", "cv": ma.CV_DESC, "seed_cv": 123},
        "best_params": {"alpha": alpha, "target_transform": "log1p" if use_log else "identity"},
        "cv_mae": {"mean": cv["mae"]["mean"], "std": cv["mae"]["std"]},
        "cv_table": cv,
        "target_transform_ablation": {"log1p_cv_mae": mae_log, "identity_cv_mae": mae_id,
                                       "tolerance": LOG_PREF_TOL},
        "top_coefficients_by_absolute_value": [
            {"feature": str(names[i]), "coefficient": float(coefs[i])} for i in top],
        "n_dev": len(yd), "n_test": len(yt),
        **test,
        "runtime_seconds": {"log_grid": t_log, "identity_grid": t_id,
                            "final_fit": fit_seconds, "total": time.perf_counter()-start},
        "library_versions": ma.library_versions(),
        "limitations": ["Linear additive model; interactions limited to shared engineered features.",
                        "One-hot coefficients for rare levels may be unstable.",
                        "Ames transactions are historical and from one city; this is not a current Vietnam valuation model."],
    }
    ma.save_json("ridge_metrics.json", payload)
    ma.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, ma.RESULTS_DIR / "ridge_model.joblib", compress=3)
    print(json.dumps({k: payload[k] for k in ["best_params", "cv_mae", "test_mae", "test_rmse", "test_r2"]}, indent=2))


if __name__ == "__main__":
    main()
