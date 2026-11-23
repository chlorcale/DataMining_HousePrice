"""Publish verified model comparison metrics as static, read-only JSON."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ("Ridge", "member_a/ridge_metrics.json"),
    ("Random Forest", "member_a/random_forest_metrics.json"),
    ("ElasticNet", "member_b/elasticnet_metrics.json"),
    ("LightGBM", "member_b/lightgbm_metrics.json"),
]


def main():
    rows = []
    for name, relative in SOURCES:
        path = ROOT / "results" / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing verified model result: {path.relative_to(ROOT)}")
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "model": name,
            "cv_mae": data["cv_mae"]["mean"],
            "cv_mae_std": data["cv_mae"]["std"],
            "test_mae": data["test_mae"],
            "test_rmse": data["test_rmse"],
            "test_r2": data["test_r2"],
            "test_mape_pct": data["test_mape_pct"],
            "best_params": data.get("best_params", {}),
        })
    output = ROOT / "docs" / "model" / "research_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "scope": "Ames Housing historical transactions, Ames Iowa, USD",
        "holdout_rows": 586,
        "models": rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
