"""Member B — post-hoc sensitivity of the ElasticNet vs LightGBM holdout comparison.

Descriptive diagnostic only; nothing here feeds back into model selection.
- Both frozen configurations are refit on the 2344 dev rows from the best_params
  stored in results/member_b/*_metrics.json (no new tuning).
- The holdout comparison is repeated without houses whose gr_liv_area > 4000.
  The rule uses a feature only (not the target or the residuals), in line with
  the shared outlier policy of "no target-derived filtering of the holdout".
- A paired bootstrap over the 586 holdout rows gives a confidence interval for
  the ElasticNet-minus-LightGBM error difference.
- CV per-fold MAE of both models is compared from the stored metrics files.
- Pooled out-of-fold (OOF) predictions of both frozen configs over the shared
  5-fold CV give a row-level CV comparison, with and without the same houses.
- A constant median predictor (fit on dev) is reported as a naive baseline.

The area rule follows the dataset documentation (De Cock, 2011), which flags
houses above 4000 sqft as 5 unusual observations: 3 Partial sales that likely
do not reflect market values and 2 very large houses priced appropriately.

Run from the repo root: python members/member_b/analyze_holdout_sensitivity.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from members.member_b import common as mb
from members.member_b.train_elasticnet import make_model as make_enet
from members.member_b.train_lightgbm import make_model as make_lgbm

AREA_RULE = 4000          # sqft; houses above this are treated as atypical
HIGH_PRICE = 400000       # USD; threshold for the high-price residual check
N_BOOT = 5000
BOOT_SEED = 0


def r2(y, p):
    return float(1.0 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def block(y, p):
    err = np.abs(p - y)
    return {"mae": float(err.mean()), "median_ae": float(np.median(err)),
            "rmse": float(np.sqrt(np.mean(err ** 2))), "r2": r2(y, p),
            "mape_pct": mb.mape_pct(y, p)}


def oof_compare(y, p_en, p_lg, mask):
    """Row-level paired comparison of pooled out-of-fold MAE on the masked rows."""
    e_en, e_lg = np.abs(p_en - y)[mask], np.abs(p_lg - y)[mask]
    diff = e_en - e_lg
    return {"n": int(mask.sum()),
            "mae_elasticnet": float(e_en.mean()), "mae_lightgbm": float(e_lg.mean()),
            "mae_diff_enet_minus_lgbm": float(diff.mean()),
            "mae_diff_se": float(diff.std(ddof=1) / np.sqrt(mask.sum())),
            "lgbm_lower_pct": float(100.0 * diff.mean() / e_en.mean())}


def load_params(name):
    data = json.loads((mb.RESULTS_DIR / name).read_text(encoding="utf-8"))
    return data, dict(data["best_params"])


def main():
    X_dev, y_dev, X_test, y_test = mb.load_dev_test()
    en_json, en_p = load_params("elasticnet_metrics.json")
    lg_json, lg_p = load_params("lightgbm_metrics.json")
    en_p.pop("target_transform"), en_p.pop("max_iter")
    lg_p.pop("target_transform")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        enet = make_enet(alpha=en_p["alpha"], l1_ratio=en_p["l1_ratio"], log=True).fit(X_dev, y_dev)
        lgbm = make_lgbm(log=True, **lg_p).fit(X_dev, y_dev)
    y = y_test.to_numpy(dtype=float)
    p_en = np.asarray(enet.predict(X_test), dtype=float)
    p_lg = np.asarray(lgbm.predict(X_test), dtype=float)

    # Sanity: the refit must reproduce the stored one-shot holdout numbers.
    assert abs(block(y, p_en)["mae"] - en_json["test_mae"]) < 1e-6
    assert abs(block(y, p_lg)["mae"] - lg_json["test_mae"]) < 1e-6

    area = X_test["gr_liv_area"].to_numpy(dtype=float)
    atypical = area > AREA_RULE
    keep = ~atypical
    atypical_rows = [{"gr_liv_area": float(area[i]), "sale_price": float(y[i]),
                      "sale_condition": str(X_test["sale_condition"].iloc[i]),
                      "elasticnet_pred": float(p_en[i]), "lightgbm_pred": float(p_lg[i])}
                     for i in np.flatnonzero(atypical)]

    e_en, e_lg = np.abs(p_en - y), np.abs(p_lg - y)
    rng = np.random.default_rng(BOOT_SEED)
    n = len(y)
    d_mae, d_rmse = [], []
    for _ in range(N_BOOT):
        s = rng.integers(0, n, n)
        d_mae.append(e_en[s].mean() - e_lg[s].mean())
        d_rmse.append(np.sqrt(np.mean(e_en[s] ** 2)) - np.sqrt(np.mean(e_lg[s] ** 2)))
    d_mae, d_rmse = np.array(d_mae), np.array(d_rmse)

    # Naive baseline: constant median of the dev prices.
    baseline = block(y, np.asarray(
        DummyRegressor(strategy="median").fit(X_dev, y_dev).predict(X_test), dtype=float))

    # Pooled out-of-fold predictions on dev with the frozen configurations.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oof_en = np.asarray(cross_val_predict(
            make_enet(alpha=en_p["alpha"], l1_ratio=en_p["l1_ratio"], log=True),
            X_dev, y_dev, cv=mb.make_cv()), dtype=float)
        oof_lg = np.asarray(cross_val_predict(
            make_lgbm(log=True, **lg_p), X_dev, y_dev, cv=mb.make_cv()), dtype=float)
    y_d = y_dev.to_numpy(dtype=float)
    area_d = X_dev["gr_liv_area"].to_numpy(dtype=float)
    atyp_d = area_d > AREA_RULE
    dev_excluded = [{"gr_liv_area": float(area_d[i]), "sale_price": float(y_d[i]),
                     "sale_condition": str(X_dev["sale_condition"].iloc[i])}
                    for i in np.flatnonzero(atyp_d)]

    # Residual direction on expensive homes (pred - actual, negative = under-prediction).
    hi = y > HIGH_PRICE
    high_price = {"threshold_usd": HIGH_PRICE, "n": int(hi.sum()),
                  "mean_pred_minus_actual": {"elasticnet": float((p_en - y)[hi].mean()),
                                             "lightgbm": float((p_lg - y)[hi].mean())},
                  "share_underpredicted": {"elasticnet": float(np.mean((p_en < y)[hi])),
                                           "lightgbm": float(np.mean((p_lg < y)[hi]))}}

    f_en = np.asarray(en_json["cv_table"]["mae"]["per_fold"], dtype=float)
    f_lg = np.asarray(lg_json["cv_table"]["mae"]["per_fold"], dtype=float)
    worst = int(np.argmax(f_en))

    out = {
        "note": "post-hoc descriptive diagnostic; frozen configs, no tuning",
        "area_rule_sqft": AREA_RULE,
        "n_test": int(n), "n_excluded": int(atypical.sum()),
        "excluded_rows": atypical_rows,
        "baseline_median_predictor_holdout": baseline,
        "holdout_all": {"elasticnet": block(y, p_en), "lightgbm": block(y, p_lg)},
        "high_price_holdout_residuals": high_price,
        "oof_cv": {
            "note": "pooled out-of-fold predictions of the frozen configs over the shared 5-fold CV",
            "excluded_dev_rows": dev_excluded,
            "all_rows": oof_compare(y_d, oof_en, oof_lg, np.ones(len(y_d), dtype=bool)),
            "without_atypical": oof_compare(y_d, oof_en, oof_lg, ~atyp_d),
        },
        "holdout_without_atypical": {"elasticnet": block(y[keep], p_en[keep]),
                                     "lightgbm": block(y[keep], p_lg[keep])},
        "paired_bootstrap": {
            "n_boot": N_BOOT, "seed": BOOT_SEED,
            "mae_diff_enet_minus_lgbm_ci95": [float(v) for v in np.percentile(d_mae, [2.5, 97.5])],
            "rmse_diff_enet_minus_lgbm_ci95": [float(v) for v in np.percentile(d_rmse, [2.5, 97.5])],
            "p_lgbm_lower_mae": float(np.mean(d_mae > 0)),
            "share_rows_lgbm_better": float(np.mean(e_lg < e_en)),
        },
        "cv_folds": {
            "mae_enet": f_en.tolist(), "mae_lgbm": f_lg.tolist(),
            "folds_lgbm_better": int((f_lg < f_en).sum()),
            "worst_enet_fold": worst + 1,
            "mean_mae_without_worst_fold": {
                "elasticnet": float(np.delete(f_en, worst).mean()),
                "lightgbm": float(np.delete(f_lg, worst).mean())},
        },
    }
    mb.save_json("holdout_sensitivity.json", out)
    print(json.dumps({k: out[k] for k in ("n_excluded", "baseline_median_predictor_holdout",
                                          "high_price_holdout_residuals", "oof_cv")}, indent=1))


if __name__ == "__main__":
    main()
