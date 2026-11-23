"""Tune Random Forest on development data, then score holdout once."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from members.member_a import common as ma
from src.protocol import make_cv
from src.preprocessing import make_tree_preprocessor

SPACE = {
    "regressor__estimator__n_estimators": [300, 500, 800],
    "regressor__estimator__max_depth": [None, 12, 20, 30],
    "regressor__estimator__min_samples_split": [2, 5, 10],
    "regressor__estimator__min_samples_leaf": [1, 2, 4],
    "regressor__estimator__max_features": [0.5, 0.8, 1.0],
    "regressor__estimator__max_samples": [0.7, 0.85, None],
}
N_ITER = 30


def load_or_search(Xd, yd):
    """Reuse a complete cache only when it matches this seeded parameter sample."""
    cache = ma.RESULTS_DIR / "random_forest_cv_results.csv"
    expected = list(ParameterSampler(SPACE, n_iter=N_ITER, random_state=ma.SEED_SEARCH))
    if cache.exists():
        frame = pd.read_csv(cache)
        cols = {k: "param_" + k for k in SPACE}
        if len(frame) == N_ITER and all(v in frame for v in cols.values()):
            norm = lambda value: "None" if pd.isna(value) else str(value)
            observed = [tuple(norm(row[cols[k]]) for k in SPACE) for _, row in frame.iterrows()]
            wanted = [tuple(norm(p[k]) for k in SPACE)
                      for p in expected]
            if sorted(observed) == sorted(wanted):
                print(f"Loaded matching cached search: {len(frame)} configurations", flush=True)
                return frame, 0.0

    search = RandomizedSearchCV(make_model(log=True), SPACE, n_iter=N_ITER,
                                cv=make_cv(), scoring="neg_mean_absolute_error",
                                random_state=ma.SEED_SEARCH, n_jobs=-1, refit=False,
                                return_train_score=False, verbose=1)
    t0 = time.perf_counter()
    search.fit(Xd, yd)
    elapsed = time.perf_counter() - t0
    frame = pd.DataFrame(search.cv_results_)
    frame["cv_mae"] = -frame.mean_test_score
    frame["cv_mae_std"] = frame.std_test_score
    ma.save_csv(frame.sort_values("rank_test_score"), "random_forest_cv_results.csv")
    return frame, elapsed


def make_model(log=True, **params):
    pipe = Pipeline([("preprocess", make_tree_preprocessor()),
                     ("estimator", RandomForestRegressor(
                         random_state=ma.SEED_ESTIMATOR, n_jobs=1, **params))])
    return ma.wrap_target(pipe, log=log)


def main():
    started = time.perf_counter()
    Xd, yd, Xt, yt = ma.load_dev_test()
    results, search_seconds = load_or_search(Xd, yd)
    best_row = results.loc[results.cv_mae.idxmin()]
    params = {k.replace("param_regressor__estimator__", ""): v
              for k, v in best_row.items() if k.startswith("param_regressor__estimator__")}
    for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
        if params.get(key) is not None:
            params[key] = int(params[key])
    if params.get("max_samples") is not None:
        params["max_samples"] = float(params["max_samples"])
    else:
        params["max_samples"] = None
    if params.get("max_features") is not None:
        params["max_features"] = float(params["max_features"])

    # Target transforms can still affect leaf averages even though trees are
    # invariant to monotone transforms for split ordering. Compare both at the
    # selected structure using development CV only.
    model_log = make_model(log=True, **params)
    model_id = make_model(log=False, **params)
    cv_log = ma.cv_folds_table(model_log, Xd, yd)
    cv_id = ma.cv_folds_table(model_id, Xd, yd)
    use_log = cv_log["mae"]["mean"] <= cv_id["mae"]["mean"]
    model = model_log if use_log else model_id
    cv = cv_log if use_log else cv_id
    t0 = time.perf_counter()
    model.fit(Xd, yd)
    fit_seconds = time.perf_counter() - t0
    pred = np.asarray(model.predict(Xt), float)
    test = ma.holdout_metrics(yt, pred)
    forest = model.regressor_.named_steps["estimator"]
    prep = model.regressor_.named_steps["preprocess"]
    names = prep.named_steps["pre"].get_feature_names_out()
    imp = forest.feature_importances_
    top = np.argsort(-imp)[:20]
    payload = {
        "model": "Random Forest Regressor (scikit-learn)",
        "preprocessing": "FeatureBuilder(86 predictors), numeric median imputation + missing indicators, categorical Unknown fill + OrdinalEncoder; fitted within each CV fold.",
        "selection_policy": "RandomizedSearchCV selects tree structure using log1p-target development CV MAE in USD; at selected structure compare log1p and identity via development CV MAE; then score holdout once.",
        "search_space": SPACE, "n_iter": N_ITER,
        "tuning": {"method": "RandomizedSearchCV", "cv": ma.CV_DESC,
                   "seed_cv": 123, "seed_search": ma.SEED_SEARCH},
        "best_params": {**params, "target_transform": "log1p" if use_log else "identity"},
        "cv_mae": {"mean": cv["mae"]["mean"], "std": cv["mae"]["std"]},
        "cv_table": cv,
        "target_transform_ablation": {"log1p_cv_mae": cv_log["mae"]["mean"],
                                       "identity_cv_mae": cv_id["mae"]["mean"]},
        "feature_importance_top20": [{"feature": str(names[i]), "importance": float(imp[i])} for i in top],
        "n_dev": len(yd), "n_test": len(yt),
        **test,
        "runtime_seconds": {"randomized_search": search_seconds, "final_fit": fit_seconds,
                            "total": time.perf_counter()-started},
        "library_versions": ma.library_versions(),
        "limitations": ["Ordinal encoding imposes an arbitrary order on nominal categories for tree splits.",
                        "Ensemble can be less interpretable than a linear model; impurity importance can favor high-cardinality features.",
                        "Ames transactions are historical and from one city; this is not a current Vietnam valuation model."],
    }
    ma.save_json("random_forest_metrics.json", payload)
    import joblib
    joblib.dump(model, ma.RESULTS_DIR / "random_forest_model.joblib", compress=3)
    print(json.dumps({k: payload[k] for k in ["best_params", "cv_mae", "test_mae", "test_rmse", "test_r2"]}, indent=2))


if __name__ == "__main__":
    main()
