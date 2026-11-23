"""Member B — LightGBM: tuning, ablations, one-shot holdout evaluation.

Protocol (docs/TEAM_MODEL_CONTRACT.md):
- All model selection on the 2344 development rows via the frozen 5-fold CV
  (KFold shuffle, seed 123). make_lightgbm_preprocessor() is unfitted: the
  pandas category vocabularies are learned on each training fold only, and
  numeric NaNs are kept for LightGBM's native missing handling.
- Target hypothesis log1p(SalePrice) vs identity (scale-invariant trees), both
  predicting back to USD before any metric; selection by CV MAE in USD.
- The 586-row holdout is scored exactly once, after the configuration freeze.
- Post-freeze diagnostics (gain importance, per-neighborhood and per-year
  holdout MAE) are descriptive only; they did not influence selection.
"""
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from members.member_b import common as mb
from src.preprocessing import make_lightgbm_preprocessor

# sklearn-canonical names; LightGBM aliases: colsample_bytree=feature_fraction,
# subsample=bagging_fraction, subsample_freq=bagging_freq, reg_lambda=lambda_l2.
LGB_SPACE = {
    "regressor__estimator__n_estimators": [300, 500, 700, 1000],
    "regressor__estimator__learning_rate": [0.03, 0.05, 0.1],
    "regressor__estimator__num_leaves": [15, 31, 63],
    "regressor__estimator__min_child_samples": [5, 10, 20, 40],
    "regressor__estimator__colsample_bytree": [0.6, 0.8, 1.0],
    "regressor__estimator__subsample": [0.7, 0.9, 1.0],
    "regressor__estimator__subsample_freq": [1],
    "regressor__estimator__reg_lambda": [0.0, 1.0, 5.0, 10.0],
}
N_ITER = 40          # ~40 configs x 5 folds, user-approved budget
N_ITER_FALLBACK = 15  # only if identity target turns out materially better
LOG_PREF_TOL = 0.01   # keep log1p unless identity wins CV MAE by more than 1%


def make_model(log=True, **params):
    est = Pipeline([
        ("preprocess", make_lightgbm_preprocessor()),
        ("estimator", LGBMRegressor(random_state=mb.SEED_ESTIMATOR,
                                    verbosity=-1, n_jobs=-1, **params)),
    ])
    return mb.wrap_target(est, log=log)


def params_of(row):
    out = {}
    for k, v in row.items():
        if k.startswith("param_regressor__estimator__"):
            key = k.replace("param_regressor__estimator__", "")
            out[key] = int(v) if float(v).is_integer() else float(v)
    return out


def run_search(X, y, log, n_iter, tag):
    search = RandomizedSearchCV(
        make_model(log=log), LGB_SPACE, n_iter=n_iter,
        cv=mb.make_cv(), scoring="neg_mean_absolute_error",
        random_state=mb.SEED_SEARCH, n_jobs=1, refit=False,
        return_train_score=False, verbose=2)
    t0 = time.perf_counter()
    search.fit(X, y)
    dt = time.perf_counter() - t0
    res = pd.DataFrame(search.cv_results_)
    res["cv_mae"] = -res["mean_test_score"]
    res["cv_mae_std"] = res["std_test_score"]
    print(f"{tag}: best cv_mae={res.cv_mae.min():.1f} in {dt:.1f}s", flush=True)
    return res, dt


def main():
    t_start = time.perf_counter()
    X_dev, y_dev, X_test, y_test = mb.load_dev_test()
    print(f"dev={X_dev.shape} holdout={X_test.shape} (frozen until final eval)", flush=True)

    # ---- Primary search: log1p target ----
    res, dt1 = run_search(X_dev, y_dev, True, N_ITER, "search(log1p)")
    mb.save_csv(res.sort_values("rank_test_score"), "lightgbm_cv_results.csv")
    best_row = res.loc[res.cv_mae.idxmin()]
    best_params = params_of(best_row)
    print("best params (log1p):", best_params, flush=True)

    # ---- Ablation: identity target at the best structure ----
    id_model = make_model(log=False, **best_params)
    id_table = mb.cv_folds_table(id_model, X_dev, y_dev)
    id_mae = id_table["mae"]["mean"]
    log_mae = float(best_row.cv_mae)
    print(f"ablation identity at best structure: cv_mae={id_mae:.1f} vs log1p {log_mae:.1f}", flush=True)

    # ---- Selection policy ----
    final_log, final_params = True, best_params
    secondary = None
    if id_mae < log_mae * (1 - LOG_PREF_TOL):
        print("identity target materially better -> secondary search on identity", flush=True)
        res2, dt2 = run_search(X_dev, y_dev, False, N_ITER_FALLBACK, "search(identity)")
        row2 = res2.loc[res2.cv_mae.idxmin()]
        if float(row2.cv_mae) < log_mae:
            final_log, final_params = False, params_of(row2)
            secondary = {"n_iter": N_ITER_FALLBACK, "best_cv_mae": float(row2.cv_mae),
                         "best_params": params_of(row2)}
        else:
            secondary = {"n_iter": N_ITER_FALLBACK, "best_cv_mae": float(row2.cv_mae),
                         "best_params": params_of(row2), "not_selected": "log1p still better"}
    selection_note = (f"target={('log1p' if final_log else 'identity')} "
                      f"(identity at best structure: {id_mae:.1f} vs log1p {log_mae:.1f} USD, "
                      f"tolerance {LOG_PREF_TOL:.0%})")

    # ---- Per-fold CV table of the chosen configuration ----
    final_model = make_model(log=final_log, **final_params)
    cv_table = mb.cv_folds_table(final_model, X_dev, y_dev)

    # ---- Final fit on all dev rows; train-vs-CV overfitting gap ----
    t0 = time.perf_counter()
    final_model.fit(X_dev, y_dev)
    t_final_fit = time.perf_counter() - t0
    pred_train = np.asarray(final_model.predict(X_dev), dtype=float)
    train_mae = float(np.mean(np.abs(np.asarray(y_dev, dtype=float) - pred_train)))
    print(f"final fit {t_final_fit:.1f}s | dev-train MAE={train_mae:.1f} "
          f"vs CV MAE={cv_table['mae']['mean']:.1f}", flush=True)

    # ---- One-shot holdout evaluation ----
    pred_test = np.asarray(final_model.predict(X_test), dtype=float)
    test_block = mb.test_block(y_test, pred_test)
    print("test:", {k: round(v, 1) for k, v in test_block.items()}, flush=True)

    # ---- Gain/split importance (post-freeze, descriptive) ----
    fitted = final_model.regressor_
    booster = fitted.named_steps["estimator"].booster_
    names = booster.feature_name()
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    split = np.asarray(booster.feature_importance(importance_type="split"), dtype=float)
    gain_pct = gain / gain.sum() * 100.0
    top_idx = np.argsort(-gain_pct)[:20]
    importance = [{"feature": names[i], "gain_pct": round(float(gain_pct[i]), 3),
                   "splits": int(split[i])} for i in top_idx]

    # ---- Generalization by neighborhood and sale year (post-freeze) ----
    diag = pd.DataFrame({
        "neighborhood": X_test["neighborhood"].to_numpy(),
        "yr_sold": pd.to_numeric(X_test["yr_sold"], errors="coerce").to_numpy(),
        "y": np.asarray(y_test, dtype=float), "p": pred_test})
    diag["abs_err"] = (diag.y - diag.p).abs()
    by_nb = diag.groupby("neighborhood").agg(n=("abs_err", "size"), mae=("abs_err", "mean"))
    by_nb = by_nb[by_nb.n >= 10].sort_values("n", ascending=False)
    by_yr = diag.groupby("yr_sold").agg(n=("abs_err", "size"), mae=("abs_err", "mean")).sort_index()
    nb_json = {str(k): {"n": int(v.n), "mae": float(v.mae)} for k, v in by_nb.iterrows()}
    yr_json = {str(int(k)): {"n": int(v.n), "mae": float(v.mae)} for k, v in by_yr.iterrows()}

    # ---- Figures ----
    fig, ax = plt.subplots(figsize=(7, 4.8))
    top = importance[::-1]
    ax.barh([t["feature"] for t in top], [t["gain_pct"] for t in top], color="tab:green")
    ax.set(xlabel="Gain importance (% of total)", ylabel="",
           title="LightGBM top-20 features by gain")
    fig.tight_layout()
    mb.save_fig(fig, "lgbm_importance.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(pred_test, np.asarray(y_test, dtype=float), s=12, alpha=.5)
    lims = [min(pred_test.min(), float(y_test.min())), max(pred_test.max(), float(y_test.max()))]
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set(xlabel="Predicted (USD)", ylabel="Actual (USD)", title="Holdout: actual vs predicted")
    axes[1].scatter(pred_test, pred_test - np.asarray(y_test, dtype=float), s=12, alpha=.5)
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="Predicted (USD)", ylabel="Residual p - y (USD)", title="Holdout residuals")
    fig.tight_layout()
    mb.save_fig(fig, "lgbm_residuals.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    nb_head = by_nb.head(12)
    axes[0].bar(range(len(nb_head)), nb_head.mae, color="tab:blue")
    axes[0].set_xticks(range(len(nb_head)))
    axes[0].set_xticklabels(nb_head.index, rotation=60, ha="right", fontsize=8)
    axes[0].set(ylabel="MAE (USD)", title="Holdout MAE by neighborhood (n>=10)")
    axes[1].bar([str(int(i)) for i in by_yr.index], by_yr.mae, color="tab:orange")
    axes[1].set(ylabel="MAE (USD)", xlabel="yr_sold", title="Holdout MAE by sale year")
    fig.tight_layout()
    mb.save_fig(fig, "lgbm_generalization.png")

    # ---- Payload ----
    payload = {
        "model": "LightGBM (LGBMRegressor, native categorical)" + (" on log1p(SalePrice)" if final_log else " on raw USD SalePrice"),
        "preprocessing": ("make_lightgbm_preprocessor(): FeatureBuilder(engineered=True) 86 cols; "
                          "categoricals as pandas category with fold-train vocabularies "
                          "(unseen levels -> NaN); numeric NaNs kept for native handling; "
                          "no scaling, no imputation"),
        "selection_policy": selection_note,
        "search_space": {"distributions": {k.replace("regressor__estimator__", ""): v
                                           for k, v in LGB_SPACE.items()},
                         "n_iter": N_ITER, "method": "RandomizedSearchCV"},
        "tuning": {
            "method": "RandomizedSearchCV, per-fold preprocessing refit",
            "cv": mb.CV_DESC, "cv_folds": 5,
            "seed_cv": 123, "seed_search": mb.SEED_SEARCH,
            "seed_estimator": mb.SEED_ESTIMATOR,
            "scoring": "neg mean absolute error in USD (after inverse target transform)",
        },
        "best_params": {**final_params, "target_transform": "log1p" if final_log else "identity"},
        "cv_mae": {"mean": cv_table["mae"]["mean"], "std": cv_table["mae"]["std"]},
        "cv_table": cv_table,
        "train_vs_cv_gap": {"dev_train_mae_after_refit": train_mae,
                            "cv_mae_mean": cv_table["mae"]["mean"],
                            "note": "gap indicates in-sample fit; preprocessing refit on all dev"},
        "ablation_identity_target": {"cv_mae_at_best_structure": id_mae,
                                     "log1p_cv_mae": log_mae,
                                     "secondary_search": secondary},
        "importance_top20_gain": importance,
        "holdout_mae_by_neighborhood": nb_json,
        "holdout_mae_by_year": yr_json,
        "n_dev": int(len(y_dev)), "n_test": int(len(y_test)),
        **test_block,
        "runtime_seconds": {"primary_search": dt1, "final_fit": t_final_fit,
                            "total": time.perf_counter() - t_start},
        "library_versions": mb.library_versions(),
        "limitations": [
            "Historic-transaction framing: sale-time fields (yr_sold, mo_sold, sale_type, sale_condition) are known features; not a pre-sale forecaster.",
            "Categorical vocabularies are learned per training fold; unseen levels at serving time fall to the NaN branch and require a refit on new data.",
            "Small dataset for boosting; heavy regularization selected by CV, yet the train-vs-CV gap stays larger than for linear models (see train_vs_cv_gap).",
            "Gain importance is correlational, not causal.",
            "One city, 2006-2010; year effects may partly memorize the market cycle rather than general price dynamics.",
            "Single holdout estimate is noisy; CV mean +/- std reported alongside.",
        ],
    }
    mb.save_json("lightgbm_metrics.json", payload)
    print("DONE lightgbm", flush=True)


if __name__ == "__main__":
    main()
