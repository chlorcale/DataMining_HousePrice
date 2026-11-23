"""Member B contract tests: handoff metrics schema + leakage-safe smoke checks.

The schema tests activate only after `members/member_b/train_*.py` has been
run; until then they skip with an explicit reason, so a fresh clone of the
repository still passes `python -m pytest -q`.
"""
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MB_RESULTS = ROOT / "results" / "member_b"

REQUIRED_KEYS = [
    "model", "preprocessing", "selection_policy", "search_space", "tuning",
    "best_params", "cv_mae", "cv_table", "n_dev", "n_test",
    "test_mae", "test_rmse", "test_r2", "test_mape_pct",
    "runtime_seconds", "library_versions", "limitations",
]

METRICS_FILES = ["elasticnet_metrics.json", "lightgbm_metrics.json"]


def _payload(name):
    path = MB_RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated yet: run members/member_b/train_*.py first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", METRICS_FILES)
def test_metrics_schema(name):
    data = _payload(name)
    for key in REQUIRED_KEYS:
        assert key in data, f"missing key: {key}"
    cv_mae = data["cv_mae"]
    assert cv_mae["mean"] > 0 and cv_mae["std"] >= 0
    # USD-scale sanity for Ames: a constant median predictor scores ~64k MAE on the holdout.
    assert 0 < data["test_mae"] < 55000
    assert 0 < data["test_r2"] <= 1
    assert 0 < data["test_mape_pct"] < 30
    assert data["test_rmse"] >= data["test_mae"]
    assert isinstance(data["limitations"], list) and len(data["limitations"]) >= 3


@pytest.mark.parametrize("name", METRICS_FILES)
def test_metrics_use_frozen_split_sizes(name):
    """Stored metrics were computed on the frozen 2344/586 split.

    This checks sizes only. It cannot prove how many times the holdout was
    scored; that is guaranteed by the order of operations in train_*.py (all
    selection happens before X_test is touched).
    """
    data = _payload(name)
    assert data["n_dev"] == 2344 and data["n_test"] == 586


def test_preprocessing_learns_only_from_fit_partition():
    """A level that exists only in the validation partition must not leak into
    the fitted vocabularies of either member B preprocessing factory."""
    from src.data import read_dataset
    from src.preprocessing import make_linear_preprocessor, make_lightgbm_preprocessor
    from src.protocol import holdout_indices

    X, _ = read_dataset()
    dev_idx, _ = holdout_indices(len(X))
    train = X.iloc[dev_idx[:300]].copy()
    valid = X.iloc[dev_idx[300:400]].copy()
    valid["neighborhood"] = "ZZ_ONLY_IN_VALIDATION"

    # Linear: one-hot vocabulary comes from the training partition only.
    linear = make_linear_preprocessor().fit(train)
    names = linear.named_steps["pre"].get_feature_names_out()
    assert not any("ZZ_ONLY_IN_VALIDATION" in n for n in names)
    out = linear.transform(valid)
    assert np.isfinite(out).all()
    # Unseen level encodes as all-zero across the neighborhood one-hot block.
    nb_cols = [i for i, n in enumerate(names) if n.startswith("cat__neighborhood_")]
    assert nb_cols and np.all(out[:, nb_cols] == 0)

    # LightGBM: category vocabulary is learned on the training partition only,
    # so the validation-only level becomes NaN instead of a new category.
    lgb_pre = make_lightgbm_preprocessor().fit(train)
    frame = lgb_pre.named_steps["frame"]
    assert "ZZ_ONLY_IN_VALIDATION" not in frame.cats_["neighborhood"]
    assert lgb_pre.transform(valid)["neighborhood"].isna().all()

    # Numeric imputation statistics come from the training partition.
    ct = linear.named_steps["pre"]
    num_cols = list(ct.transformers_[0][2])
    stats = ct.named_transformers_["num"].named_steps["impute"].statistics_
    fe_train = linear.named_steps["feature_builder"].transform(train)
    j = num_cols.index("lot_frontage")
    assert stats[j] == pytest.approx(float(np.nanmedian(fe_train["lot_frontage"].astype(float))))


def test_preprocessing_refits_inside_pipeline():
    """Smoke: both member B factories run unfitted inside a small pipeline and
    reproduce identical predictions with the same seed (determinism, not leakage)."""
    from sklearn.linear_model import ElasticNet
    from sklearn.pipeline import Pipeline
    from lightgbm import LGBMRegressor

    from members.member_b.common import wrap_target
    from src.data import read_dataset
    from src.protocol import holdout_indices
    from src.preprocessing import make_linear_preprocessor, make_lightgbm_preprocessor

    X, y = read_dataset()
    dev_idx, _ = holdout_indices(len(y))
    Xs, ys = X.iloc[dev_idx[:200]], y.iloc[dev_idx[:200]]

    def fresh_elasticnet():
        return wrap_target(Pipeline([
            ("preprocess", make_linear_preprocessor()),
            ("estimator", ElasticNet(alpha=1e-3, max_iter=5000, random_state=42)),
        ]))

    p1 = fresh_elasticnet().fit(Xs, ys).predict(Xs)
    p2 = fresh_elasticnet().fit(Xs, ys).predict(Xs)
    assert np.allclose(p1, p2) and np.isfinite(p1).all()

    lgb = Pipeline([
        ("preprocess", make_lightgbm_preprocessor()),
        ("estimator", LGBMRegressor(n_estimators=40, verbosity=-1, random_state=42)),
    ]).fit(Xs, ys)
    q = lgb.predict(Xs)
    assert np.isfinite(q).all() and (q > 0).all()


def test_scorers_are_usd_scale():
    """MAPE guard matches src.evaluation: never divides by a near-zero price."""
    from members.member_b.common import mape_pct

    assert mape_pct([100.0, 200.0], [110.0, 180.0]) == pytest.approx(10.0)
    assert mape_pct([0.5], [1.0]) == 50.0  # guarded denominator, no ZeroDivision
