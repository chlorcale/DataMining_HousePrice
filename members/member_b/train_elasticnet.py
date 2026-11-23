"""Member B — ElasticNet: tuning, ablations, one-shot holdout evaluation.

Protocol (docs/TEAM_MODEL_CONTRACT.md):
- All model selection on the 2344 development rows via the frozen 5-fold CV
  (KFold shuffle, seed 123). The preprocessing factory is unfitted and refit
  inside each training fold, so validation rows never leak into imputers,
  scalers or one-hot vocabularies.
- Target hypothesis log1p(SalePrice) vs identity, both predicting back to USD
  before any metric; selection by CV MAE in USD.
- Member B mandate is a *mixed* elastic net (0 < l1_ratio < 1); l1_ratio=0 and
  l1_ratio=1 are evaluated as reference points only (they approximate the
  Ridge/Lasso owned by Member A).
- The 586-row holdout is scored exactly once, after the configuration freeze.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from members.member_b import common as mb
from src.preprocessing import make_linear_preprocessor

# Log1p target shrinks the penalty scale; identity target needs a much larger
# alpha range to cover the equivalent regularization strength on USD labels.
ALPHAS_LOG = list(np.logspace(-5, 0, 19))
ALPHAS_ID = list(np.logspace(-1, 4, 15))
L1R_GRID = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 1.0]
L1R_MIXED = [r for r in L1R_GRID if 0.0 < r < 1.0]
LOG_PREF_TOL = 0.01  # keep log1p unless identity wins CV MAE by more than 1%


def make_model(alpha=1.0, l1_ratio=0.5, log=True):
    est = Pipeline([
        ("preprocess", make_linear_preprocessor()),
        ("estimator", ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000,
                                  random_state=mb.SEED_ESTIMATOR)),
    ])
    return mb.wrap_target(est, log=log)


def cache_matches(res, alphas, l1_ratios):
    """True when a cached grid holds exactly the requested (alpha, l1_ratio) pairs.

    Guards against silently reusing a stale CSV after a grid definition change
    (the stage-2 grid also moves whenever the stage-1 optimum moves).
    """
    got = np.array(sorted(zip(res["param_regressor__estimator__alpha"].astype(float),
                              res["param_regressor__estimator__l1_ratio"].astype(float))))
    want = np.array(sorted((float(a), float(r)) for a in alphas for r in l1_ratios))
    return got.shape == want.shape and np.allclose(got, want, rtol=1e-9, atol=0.0)


def grid_search(X, y, log, alphas, l1_ratios, tag, cache_name, n_jobs=-1):
    """Run one grid; results are cached to results/member_b/ for resumability."""
    cache = mb.RESULTS_DIR / cache_name
    if cache.exists():
        res = pd.read_csv(cache)
        if cache_matches(res, alphas, l1_ratios):
            side = cache.with_suffix(".runtime.json")
            dt = None
            if side.exists():
                dt = json.loads(side.read_text(encoding="utf-8"))["seconds"]
            msg = f"{tag}: reloaded {len(res)} cached configs"
            if dt:
                msg += f" (wall {dt:.1f}s)"
            print(msg, flush=True)
            return res, dt
        print(f"{tag}: cached grid differs from the requested grid, recomputing", flush=True)
    grid = GridSearchCV(
        make_model(log=log),
        {"regressor__estimator__alpha": list(alphas),
         "regressor__estimator__l1_ratio": list(l1_ratios)},
        cv=mb.make_cv(), scoring="neg_mean_absolute_error",
        n_jobs=n_jobs, refit=False, return_train_score=False)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        grid.fit(X, y)
    dt = time.perf_counter() - t0
    res = pd.DataFrame(grid.cv_results_)
    res["cv_mae"] = -res["mean_test_score"]
    res["cv_mae_std"] = res["std_test_score"]
    mb.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    res.sort_values("rank_test_score").to_csv(cache, index=False)
    cache.with_suffix(".runtime.json").write_text(
        json.dumps({"seconds": dt}), encoding="utf-8")
    print(f"{tag}: {len(res)} configs x 5 folds in {dt:.1f}s | best cv_mae={res.cv_mae.min():.1f}", flush=True)
    return res, dt


def best_mixed(res):
    """Best row among strictly mixed l1_ratio (0 < r < 1)."""
    l1 = res["param_regressor__estimator__l1_ratio"].astype(float)
    sub = res[l1.between(0.0, 1.0, inclusive="neither")]
    return sub.loc[sub.cv_mae.idxmin()]


def endpoint_best(res, value):
    l1 = res["param_regressor__estimator__l1_ratio"].astype(float)
    sub = res[l1 == value]
    row = sub.loc[sub.cv_mae.idxmin()]
    return {"cv_mae": float(row.cv_mae),
            "alpha": float(row["param_regressor__estimator__alpha"])}


def fold_convergence(X, y, alpha, l1_ratio, log):
    """Iterations used and ConvergenceWarnings when refitting each CV training fold.

    Grid search silences ConvergenceWarning, so the chosen configuration is
    re-checked on the same five training partitions that the CV used.
    """
    out = []
    for k, (train_idx, _) in enumerate(mb.make_cv().split(X), 1):
        model = make_model(alpha=alpha, l1_ratio=l1_ratio, log=log)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(X.iloc[train_idx], y.iloc[train_idx])
        est = model.regressor_.named_steps["estimator"]
        out.append({"fold": k, "n_iter": int(np.max(est.n_iter_)),
                    "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning)
                                                for w in caught)})
    return out


def pretty_name(n):
    n = n.replace("num__", "").replace("cat__", "")
    return n.replace("missingindicator_", "is_missing_")


def main():
    t_start = time.perf_counter()
    X_dev, y_dev, X_test, y_test = mb.load_dev_test()
    print(f"dev={X_dev.shape} holdout={X_test.shape} (frozen until final eval)", flush=True)

    # ---- Stage 1: coarse grid on log1p target, all l1_ratios ----
    res1, dt1 = grid_search(X_dev, y_dev, True, ALPHAS_LOG, L1R_GRID,
                            "stage1(log1p)", "elasticnet_cv_results_stage1.csv")
    best1 = best_mixed(res1)
    a0 = float(best1["param_regressor__estimator__alpha"])
    r0 = float(best1["param_regressor__estimator__l1_ratio"])

    # ---- Stage 2: refine around the stage-1 optimum ----
    alphas2 = [float(a) for a in np.logspace(np.log10(a0) - 0.6, np.log10(a0) + 0.6, 13)]
    l1r2 = sorted({round(min(max(r, 0.01), 0.99), 3) for r in (r0 - 0.05, r0, r0 + 0.05)})
    res2, dt2 = grid_search(X_dev, y_dev, True, alphas2, l1r2,
                            "stage2(log1p, refine)", "elasticnet_cv_results_stage2.csv")
    best_log = best2 = res2.loc[res2.cv_mae.idxmin()]
    if best1.cv_mae < best2.cv_mae:
        best_log = best1
    log_grid_union = pd.concat([res1, res2], ignore_index=True)

    # ---- Ablation: identity target at USD penalty scale (mixed l1 only) ----
    res_id, dt_id = grid_search(X_dev, y_dev, False, ALPHAS_ID, L1R_MIXED,
                                "identity-grid", "elasticnet_cv_results_identity.csv")
    best_id = best_mixed(res_id)

    # ---- Selection policy ----
    if best_id.cv_mae < best_log.cv_mae * (1 - LOG_PREF_TOL):
        final_row, final_log, res_final = best_id, False, res_id
        selection_note = (f"identity target won CV MAE by >{LOG_PREF_TOL:.0%}; "
                          f"identity chosen ({float(best_id.cv_mae):.1f} vs {float(best_log.cv_mae):.1f} USD)")
    else:
        final_row, final_log, res_final = best_log, True, log_grid_union
        selection_note = (f"log1p target kept (identity best {float(best_id.cv_mae):.1f} "
                          f"vs log best {float(best_log.cv_mae):.1f} USD, tolerance {LOG_PREF_TOL:.0%})")
    alpha_star = float(final_row["param_regressor__estimator__alpha"])
    l1_star = float(final_row["param_regressor__estimator__l1_ratio"])
    print(f"chosen: alpha={alpha_star:.4g} l1_ratio={l1_star} log={final_log}", flush=True)

    # ---- Endpoint references on the log1p grid ----
    endpoint_refs = {
        "ridge_like_l1_ratio_0": endpoint_best(res1, 0.0),
        "lasso_like_l1_ratio_1": endpoint_best(res1, 1.0),
        "note": "reference only; final model is a mixed elastic net (Member B mandate)",
    }

    # ---- Per-fold CV table of the chosen configuration ----
    final_model = make_model(alpha=alpha_star, l1_ratio=l1_star, log=final_log)
    cv_table = mb.cv_folds_table(final_model, X_dev, y_dev)
    cv_convergence = fold_convergence(X_dev, y_dev, alpha_star, l1_star, final_log)
    print("cv convergence:", cv_convergence, flush=True)

    # ---- Final fit on all dev rows; convergence checked explicitly ----
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        final_model.fit(X_dev, y_dev)
    conv_warnings = [w for w in caught if issubclass(w.category, ConvergenceWarning)]
    t_final_fit = time.perf_counter() - t0
    print(f"final fit done in {t_final_fit:.1f}s, convergence warnings: {len(conv_warnings)}", flush=True)

    # ---- One-shot holdout evaluation ----
    pred_test = np.asarray(final_model.predict(X_test), dtype=float)
    test_block = mb.test_block(y_test, pred_test)
    pred_train = np.asarray(final_model.predict(X_dev), dtype=float)
    train_mae = float(np.mean(np.abs(np.asarray(y_dev, dtype=float) - pred_train)))
    print("test:", {k: round(v, 1) for k, v in test_block.items()}, flush=True)

    # ---- Coefficients of the frozen model ----
    fitted = final_model.regressor_
    ct = fitted.named_steps["preprocess"].named_steps["pre"]
    names = [pretty_name(n) for n in ct.get_feature_names_out()]
    coefs = np.asarray(fitted.named_steps["estimator"].coef_, dtype=float)
    order = np.argsort(-np.abs(coefs))[:15]
    top_coefs = [{"feature": names[i], "coef": round(float(coefs[i]), 4)} for i in order]
    nonzero = int((np.abs(coefs) > 0).sum())

    # ---- Figures ----
    l1_col = res_final["param_regressor__estimator__l1_ratio"].astype(float)
    a_col = res_final["param_regressor__estimator__alpha"].astype(float)
    curve = res_final[l1_col == l1_star].copy()
    curve = curve.drop_duplicates(subset=["param_regressor__estimator__alpha"]).sort_values(
        by="param_regressor__estimator__alpha")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(curve["param_regressor__estimator__alpha"].astype(float), curve.cv_mae,
                yerr=curve.cv_mae_std, marker="o", capsize=3)
    ax.axvline(alpha_star, color="tab:red", ls="--", lw=1,
               label=f"chosen alpha={alpha_star:.3g}")
    ax.set(xscale="log", xlabel="alpha (log scale)", ylabel="CV MAE (USD)",
           title=f"ElasticNet CV error vs alpha (l1_ratio={l1_star})")
    ax.legend()
    fig.tight_layout()
    mb.save_fig(fig, "enet_alpha_curve.png")

    fig, ax = plt.subplots(figsize=(7, 4.4))
    top = top_coefs[::-1]
    ax.barh([t["feature"] for t in top], [t["coef"] for t in top],
            color=["tab:blue" if t["coef"] > 0 else "tab:red" for t in top])
    ax.set(xlabel="Coefficient (log-USD scale)" if final_log else "Coefficient (USD scale)",
           title=f"Top-15 |coefficient|, nonzero={nonzero}/{len(coefs)}")
    fig.tight_layout()
    mb.save_fig(fig, "enet_coefficients.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(pred_test, np.asarray(y_test, dtype=float), s=12, alpha=.5)
    lims = [min(pred_test.min(), y_test.min()), max(pred_test.max(), y_test.max())]
    axes[0].plot(lims, lims, "k--", lw=1)
    axes[0].set(xlabel="Predicted (USD)", ylabel="Actual (USD)", title="Holdout: actual vs predicted")
    axes[1].scatter(pred_test, pred_test - np.asarray(y_test, dtype=float), s=12, alpha=.5)
    axes[1].axhline(0, color="k", ls="--", lw=1)
    axes[1].set(xlabel="Predicted (USD)", ylabel="Residual p - y (USD)", title="Holdout residuals")
    fig.tight_layout()
    mb.save_fig(fig, "enet_residuals.png")

    # ---- Payload ----
    payload = {
        "model": "ElasticNet (scikit-learn)" + (" on log1p(SalePrice)" if final_log else " on raw USD SalePrice"),
        "preprocessing": ("make_linear_preprocessor(): FeatureBuilder(engineered=True) 86 cols; "
                          "numeric median impute + missing indicators + StandardScaler; "
                          "categorical 'Unknown' fill + OneHot(handle_unknown='ignore'); "
                          "~334 dense numeric output columns; refit inside each training fold"),
        "selection_policy": selection_note,
        "search_space": {
            "stage1_log_grid": {"alpha": ALPHAS_LOG, "l1_ratio": L1R_GRID},
            "stage2_refine": {"alpha": alphas2, "l1_ratio": l1r2,
                              "center": {"alpha": a0, "l1_ratio": r0}},
            "identity_grid": {"alpha": ALPHAS_ID, "l1_ratio": L1R_MIXED,
                              "note": "ablation grid; mixed l1 only, USD-scale penalty range"},
            "scoring": "neg mean absolute error in USD (after inverse target transform)",
        },
        "tuning": {
            "method": "GridSearchCV coarse->refine, per-fold preprocessing refit",
            "cv": mb.CV_DESC, "cv_folds": 5,
            "seed_cv": 123, "seed_estimator": mb.SEED_ESTIMATOR,
        },
        "best_params": {"alpha": alpha_star, "l1_ratio": l1_star,
                        "target_transform": "log1p" if final_log else "identity",
                        "max_iter": 10000},
        "cv_mae": {"mean": cv_table["mae"]["mean"], "std": cv_table["mae"]["std"]},
        "cv_table": cv_table,
        "train_vs_cv_gap": {"dev_train_mae_after_refit": train_mae,
                            "cv_mae_mean": cv_table["mae"]["mean"]},
        "endpoint_references": endpoint_refs,
        "ablation_identity_target": {
            "best_cv_mae": float(best_id.cv_mae),
            "alpha": float(best_id["param_regressor__estimator__alpha"]),
            "l1_ratio": float(best_id["param_regressor__estimator__l1_ratio"]),
        },
        "coefficients": {"nonzero": nonzero, "total": int(len(coefs)),
                         "top15_by_abs": top_coefs},
        "n_dev": int(len(y_dev)), "n_test": int(len(y_test)),
        "convergence": {"max_iter": 10000, "warnings_in_final_fit": len(conv_warnings),
                        "cv_training_folds": cv_convergence},
        **test_block,
        "runtime_seconds": {
            "stage1": dt1, "stage2_refine": dt2, "identity_grid": dt_id,
            "final_fit": t_final_fit,
            "last_invocation": time.perf_counter() - t_start,
            "note": ("grid stage times are read from the cached runs that produced the CSVs; "
                     "last_invocation covers only the latest execution, where cached stages "
                     "are reloaded instead of recomputed"),
        },
        "library_versions": mb.library_versions(),
        "limitations": [
            "Linear additive model; nonlinearity only via the 7 shared engineered features, no learned interactions.",
            "One-hot expansion to ~334 columns; per-level coefficients are noisy for rare categories.",
            "Ames, Iowa 2006-2010, one city; historic-transaction analysis (sale-time fields included), not a pre-sale forecaster.",
            "Single holdout estimate is noisy; CV mean +/- std reported alongside.",
            "Convergence warnings suppressed during grid search; the final frozen model was checked separately.",
        ],
    }
    mb.save_json("elasticnet_metrics.json", payload)
    print("DONE elasticnet", flush=True)


if __name__ == "__main__":
    main()
