"""Interactive, reproducible modeling lifecycle for the Ames regression study."""
from __future__ import annotations

import re
import time

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from app.shared import (
    MODELS, ROOT, currency, explainability_data, lesson, load_data,
    load_results, metric_table, page_hero, read_csv, scope_notice,
)

TUNING_FILES = {
    "Random Forest": "results/member_a/random_forest_cv_results.csv",
    "ElasticNet": "results/member_b/elasticnet_cv_results_stage2.csv",
    "LightGBM": "results/member_b/lightgbm_cv_results.csv",
}
ARTIFACT_DIR = ROOT / "models"


def artifact_path(name: str):
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return ARTIFACT_DIR / f"learnt_{slug}.pkl"


def make_frozen_model(name: str, payload: dict):
    """Rebuild the selected pipeline with the hyperparameters already chosen by CV."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Ridge

    from src.preprocessing import (
        make_lightgbm_preprocessor,
        make_linear_preprocessor,
        make_tree_preprocessor,
    )

    params = payload["best_params"].copy()
    target_transform = params.pop("target_transform", "identity")
    if name == "Ridge":
        estimator = Ridge(alpha=float(params["alpha"]))
        preprocessor = make_linear_preprocessor()
    elif name == "Random Forest":
        estimator = RandomForestRegressor(**params, random_state=42, n_jobs=1)
        preprocessor = make_tree_preprocessor()
    elif name == "ElasticNet":
        estimator = ElasticNet(**params, random_state=42)
        preprocessor = make_linear_preprocessor()
    elif name == "LightGBM":
        from lightgbm import LGBMRegressor

        estimator = LGBMRegressor(**params, random_state=42, verbosity=-1, n_jobs=-1)
        preprocessor = make_lightgbm_preprocessor()
    else:
        raise ValueError(f"Unknown model: {name}")

    pipeline = Pipeline([("preprocess", preprocessor), ("estimator", estimator)])
    if target_transform == "log1p":
        return TransformedTargetRegressor(
            regressor=pipeline, func=np.log1p, inverse_func=np.expm1, check_inverse=False
        )
    return TransformedTargetRegressor(regressor=pipeline, check_inverse=False)


def load_tuning_frame(name: str):
    if name == "Ridge":
        log = read_csv("results/member_a/ridge_cv_log.csv").assign(Target="log1p")
        identity = read_csv("results/member_a/ridge_cv_identity.csv").assign(Target="identity")
        return pd.concat([log, identity], ignore_index=True)
    return read_csv(TUNING_FILES[name])


def render():
    X, y, dev_idx, test_idx = load_data()
    results = load_results()
    page_hero(
        "05 · HANDS-ON MODELING",
        "Từ mẫu dữ liệu đến mô hình đã lưu",
        "Theo vòng đời huấn luyện: chọn mẫu quan sát, đọc CV và tuning, fit pipeline trên development, lưu rồi nạp model để dự đoán.",
        "FIT → LOAD",
    )
    scope_notice()
    if not results:
        st.error("Chưa có metrics. Hãy hoàn tất các script training của nhóm trước.")
        return

    model_name = st.selectbox("Chọn thuật toán", list(results), key="workflow_model")
    payload = results[model_name]
    path = artifact_path(model_name)
    st.markdown(
        "**Cách dùng:** đi lần lượt năm bước. Kết quả CV/tuning đã được lưu trong CSV; nút ở bước 3 fit lại cấu hình đã chọn "
        "trên development. Ứng dụng không chạy lại GridSearchCV/RandomizedSearchCV mỗi lần mở trang."
    )
    steps = st.tabs([
        "1 · Chọn mẫu", "2 · CV / Grid search", "3 · Fit & lưu .pkl",
        "4 · Nạp & dự đoán", "5 · Đánh giá / giải thích",
    ])

    with steps[0]:
        st.subheader("Lấy mẫu để quan sát, không tạo lớp giả")
        st.write(
            "SalePrice là số liên tục nên đây là bài toán hồi quy, không có lớp mất cân bằng kiểu phân loại. "
            "Để minh họa các mức giá, development được chia thành ba phân vị thấp, giữa và cao. "
            "Các nhóm này chỉ giúp chọn record để xem; model vẫn học từ toàn bộ development set."
        )
        X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
        groups = pd.qcut(y_dev, q=3, labels=["Giá thấp · 1/3", "Giá giữa · 1/3", "Giá cao · 1/3"])
        counts = groups.value_counts().reindex(groups.cat.categories).rename("Số giao dịch")
        st.bar_chart(counts, color="#365a49", height=220)
        group_name = st.selectbox("Chọn nhóm để xem record", list(groups.cat.categories), key="price_group")
        n_rows = st.slider("Số record hiển thị", min_value=5, max_value=30, value=12)
        mask = groups.eq(group_name)
        sample = X_dev.loc[mask].copy()
        sample.insert(0, "SalePrice (USD)", y_dev.loc[mask])
        sample = sample.sample(n=min(n_rows, len(sample)), random_state=42)
        visible = [c for c in ["SalePrice (USD)", "neighborhood", "overall_qual", "gr_liv_area", "lot_area", "year_built", "garage_cars"] if c in sample]
        st.dataframe(sample[visible].sort_values("SalePrice (USD)"), width="stretch", hide_index=True)
        st.caption("Record được lấy từ development để xem cấu trúc. Holdout vẫn được giữ riêng để đánh giá cuối.")

    with steps[1]:
        st.subheader("Đọc bằng chứng từ cross-validation")
        tuning_method = payload.get("tuning", {}).get("method", "Cross-validation")
        st.write(f"**Phương pháp:** {tuning_method}. **Giao thức:** 5-fold CV, seed 123; chọn cấu hình theo MAE (USD).")
        st.info(
            "Ridge và ElasticNet dùng GridSearchCV. Random Forest và LightGBM dùng RandomizedSearchCV. "
            "Mỗi fold fit preprocessing trên phần training của fold đó để tránh leakage."
        )
        try:
            tuning = load_tuning_frame(model_name)
            if model_name == "Ridge":
                best_by_target = tuning.loc[
                    tuning.groupby("Target")["cv_mae"].idxmin(),
                    ["Target", "cv_mae", "param_regressor__estimator__alpha"],
                ].copy()
                st.bar_chart(best_by_target.set_index("Target")[["cv_mae"]], color="#c27b4c", height=220)
                st.dataframe(
                    best_by_target.rename(columns={"cv_mae": "CV MAE (USD)", "param_regressor__estimator__alpha": "alpha"}),
                    width="stretch", hide_index=True,
                )
                st.caption("So sánh Ridge với target gốc và log1p; cấu hình cuối tuân theo selection policy trong JSON.")
            else:
                parameter_cols = [c for c in tuning if c.startswith("param_regressor__estimator__")]
                top = tuning.sort_values("cv_mae").head(10).copy()
                top["Cấu hình"] = [f"#{i + 1}" for i in range(len(top))]
                st.bar_chart(top.set_index("Cấu hình")[["cv_mae"]], color="#365a49", height=280)
                st.dataframe(top[parameter_cols + ["cv_mae", "cv_mae_std"]], width="stretch", hide_index=True)
                st.caption(f"CSV nguồn: `{TUNING_FILES[model_name]}` · mười cấu hình tốt nhất theo CV MAE.")
        except (FileNotFoundError, KeyError, ValueError) as exc:
            st.warning(f"Không đọc được CSV tuning: {exc}")
        st.markdown("**Cấu hình được chọn để refit**")
        st.json(payload["best_params"], expanded=False)

    with steps[2]:
        st.subheader("Fit cuối và lưu artifact")
        st.write(f"Pipeline sẽ fit trên {len(dev_idx):,} development rows. Holdout không tham gia fit hoặc preprocessing.")
        st.code(
            "model = make_frozen_model(selected_model, saved_metrics)\n"
            "model.fit(X_dev, y_dev)\n"
            "joblib.dump(model, 'models/learnt_model.pkl')",
            language="python",
        )
        st.caption("Artifact chứa preprocessing đã fit và estimator. File `*.pkl` bị loại khỏi Git; mỗi model được lưu riêng trên máy đang chạy app.")
        if st.button(f"Fit {model_name} và lưu model .pkl", type="primary", key="fit_save_model"):
            try:
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                model = make_frozen_model(model_name, payload)
                started = time.perf_counter()
                with st.spinner(f"Đang fit pipeline trên {len(dev_idx):,} development rows…"):
                    model.fit(X.iloc[dev_idx], y.iloc[dev_idx])
                    joblib.dump(model, path, compress=3)
                elapsed = time.perf_counter() - started
                st.session_state["workflow_last_model"] = model_name
                st.success(f"Đã fit và lưu `{path.relative_to(ROOT)}` trong {elapsed:.1f} giây.")
            except Exception as exc:
                st.error(f"Fit/save thất bại: {exc}")
        if path.exists():
            st.caption(f"Artifact hiện có: `{path.relative_to(ROOT)}` · {path.stat().st_size / (1024 * 1024):.2f} MiB")
        else:
            st.caption("Chưa có artifact cho model này. Bấm nút Fit & lưu để tạo.")

    with steps[3]:
        st.subheader("Nạp model và dự đoán một dòng holdout")
        if path.exists():
            st.success(f"Đã tìm thấy `{path.relative_to(ROOT)}`.")
            position = st.number_input("Vị trí record holdout", min_value=0, max_value=max(0, len(test_idx) - 1), value=0, step=1)
            if st.button("Nạp artifact và dự đoán", key="load_predict_model"):
                try:
                    loaded = joblib.load(path)
                    test_row = X.iloc[[test_idx[int(position)]]]
                    prediction = float(loaded.predict(test_row)[0])
                    st.session_state["workflow_prediction"] = {"model": model_name, "prediction": prediction, "record": int(position)}
                except Exception as exc:
                    st.error(f"Không nạp/dự đoán được artifact: {exc}")
            prediction = st.session_state.get("workflow_prediction")
            if prediction and prediction["model"] == model_name:
                st.metric("Dự đoán holdout · USD", currency(prediction["prediction"]))
                st.caption(f"Record holdout #{prediction['record']}; giá thật không hiển thị ở bước inference.")
        else:
            st.info("Trước tiên tạo artifact ở bước 3.")
        st.markdown("**Đoạn code tương đương**")
        st.code("model = joblib.load('models/learnt_model.pkl')\nprediction_usd = model.predict(X_new)[0]", language="python")

    with steps[4]:
        st.subheader("Đánh giá độ tin cậy và giải thích")
        col1, col2, col3 = st.columns(3)
        col1.metric("CV MAE ± SD", f"{currency(payload['cv_mae']['mean'])} ± {currency(payload['cv_mae']['std'])}")
        col2.metric("Holdout MAE", currency(payload["test_mae"]))
        col3.metric("Holdout R²", f"{payload['test_r2']:.3f}")
        st.caption("CV SD đo dao động giữa fold, không phải CI 95%. Holdout được dùng một lần để đánh giá theo protocol của nghiên cứu.")
        st.markdown("**So sánh cả bốn thuật toán**")
        st.dataframe(metric_table(results), width="stretch", hide_index=True)
        importance = explainability_data(model_name, payload)
        if not importance.empty:
            st.markdown(MODELS[model_name]["explain_label"])
            st.bar_chart(importance.sort_values(importance.columns[0]), color="#c27b4c", height=330)
            st.caption("Hệ số/importance mô tả tín hiệu mà model sử dụng; không chứng minh quan hệ nhân quả.")
        st.warning("Độ tin cậy chỉ được đo trên Ames, Iowa và thời kỳ lịch sử của bộ dữ liệu; không suy rộng trực tiếp sang địa phương hay thời gian khác.")

    with st.expander("Mở bài Markdown: lý thuyết và code Python"):
        lesson("05_model_workflow.md")


render()
