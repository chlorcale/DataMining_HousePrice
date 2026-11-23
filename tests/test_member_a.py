"""Member A result schema and leakage-safe model construction checks."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "member_a"
FILES = ["ridge_metrics.json", "random_forest_metrics.json"]


@pytest.mark.parametrize("name", FILES)
def test_metrics_schema_when_generated(name):
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} not generated; run the Member A training script")
    data = json.loads(path.read_text(encoding="utf-8"))
    for key in ["model", "preprocessing", "best_params", "cv_mae", "cv_table",
                "n_dev", "n_test", "test_mae", "test_rmse", "test_r2",
                "test_mape_pct", "runtime_seconds", "library_versions", "limitations"]:
        assert key in data
    for key in ["selection_policy", "search_space", "tuning"]:
        assert key in data
    assert data["n_dev"] == 2344 and data["n_test"] == 586
    assert data["cv_mae"]["mean"] > 0
    assert data["test_mae"] > 0 and data["test_rmse"] >= data["test_mae"]
    assert -1 < data["test_r2"] <= 1
    assert 0 < data["test_mape_pct"] < 60
    assert isinstance(data["limitations"], list) and len(data["limitations"]) >= 3


def test_member_a_models_build_and_predict():
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from members.member_a import common as ma
    from src.preprocessing import make_linear_preprocessor, make_tree_preprocessor

    X, y, _, _ = ma.load_dev_test()
    Xs, ys = X.iloc[:100], y.iloc[:100]
    models = [
        ma.wrap_target(Pipeline([("preprocess", make_linear_preprocessor()),
                                 ("estimator", Ridge(alpha=10))])),
        ma.wrap_target(Pipeline([("preprocess", make_tree_preprocessor()),
                                 ("estimator", RandomForestRegressor(n_estimators=10,
                                                                    random_state=42,
                                                                    n_jobs=1))]))]
    for model in models:
        model.fit(Xs, ys)
        pred = model.predict(Xs.iloc[:5])
        assert np.isfinite(pred).all() and (pred > 0).all()
