import pandas as pd
import streamlit as st

from app.shared import (
    MODELS,
    explainability_data,
    figure,
    lesson,
    load_results,
    metric_table,
    page_hero,
    read_csv,
    scope_notice,
    ROOT,
)

MODEL_CODE = {
    "Ridge": "members/member_a/train_ridge.py",
    "Random Forest": "members/member_a/train_random_forest.py",
    "ElasticNet": "members/member_b/train_elasticnet.py",
    "LightGBM": "members/member_b/train_lightgbm.py",
}

TUNING_CSV = {
    "Random Forest": "results/member_a/random_forest_cv_results.csv",
    "ElasticNet": "results/member_b/elasticnet_cv_results_stage2.csv",
    "LightGBM": "results/member_b/lightgbm_cv_results.csv",
}


def render():
    results = load_results()
    page_hero(
        "03 · MODELING & EVALUATION",
        "Bốn thuật toán, cùng cách đánh giá",
        "Đọc metric trên cùng giao thức trước khi kết luận. Mở từng thuật toán để hiểu cơ chế, siêu tham số và cách diễn giải.",
        "5-FOLD",
    )
    scope_notice()
    if not results:
        st.error("Chưa có kết quả JSON trong results/. Hãy hoàn tất các script training trước.")
        return

    st.subheader("Bảng kết quả thực nghiệm")
    st.dataframe(metric_table(results), width="stretch", hide_index=True)
    st.caption("CV MAE ± SD trên development (2.344 dòng); holdout metrics trên 586 dòng, USD gốc. Không dùng SD làm khoảng tin cậy 95%.")

    left, right = st.columns([.9, 1.1])
    with left:
        model_name = st.selectbox("Khám phá thuật toán", list(results.keys()), key="modeling_selected")
    payload = results[model_name]
    details = MODELS[model_name]
    with right:
        st.markdown(f"**{details['type']}**  \nPhụ trách: {details['owner']}")

    a, b, c, d = st.columns(4)
    a.metric("CV MAE", f"${payload['cv_mae']['mean']:,.0f}")
    b.metric("CV độ lệch chuẩn", f"${payload['cv_mae']['std']:,.0f}")
    c.metric("Holdout MAE", f"${payload['test_mae']:,.0f}")
    d.metric("Holdout R²", f"{payload['test_r2']:.3f}")

    st.subheader(f"{details['title']}: cơ chế và tiền xử lý")
    st.write(details["intuition"])
    st.info(details["preprocessing"])
    st.markdown("**Cấu hình đã chọn trên development CV**")
    st.json(payload.get("best_params", {}), expanded=False)

    fold_scores = payload.get("cv_table", {}).get("mae", {}).get("per_fold", [])
    if fold_scores:
        st.subheader("Dao động MAE giữa các fold")
        fold_frame = pd.DataFrame({"Fold": [f"Fold {i + 1}" for i in range(len(fold_scores))], "MAE (USD)": fold_scores}).set_index("Fold")
        st.bar_chart(fold_frame, color="#365a49", height=250)

    importance = explainability_data(model_name, payload)
    if not importance.empty:
        st.subheader("Tín hiệu mô hình sử dụng")
        st.bar_chart(importance.sort_values(importance.columns[0]), color="#c27b4c", height=390)
        st.caption(details["explain_label"] + ". Đây là giải thích toàn cục của model, không phải quan hệ nhân quả.")

    st.subheader("Dấu vết tuning trong CSV")
    if model_name == "Ridge":
        log_frame = read_csv("results/member_a/ridge_cv_log.csv").assign(Target="log1p")
        identity_frame = read_csv("results/member_a/ridge_cv_identity.csv").assign(Target="identity")
        tuning = pd.concat([log_frame, identity_frame], ignore_index=True)
        alpha_column = "param_regressor__estimator__alpha"
        best_by_target = tuning.loc[tuning.groupby("Target")["cv_mae"].idxmin(), ["Target", "cv_mae", alpha_column]]
        best_by_target = best_by_target.rename(columns={alpha_column: "alpha"}).set_index("Target")
        st.bar_chart(best_by_target[["cv_mae"]], color="#c27b4c", height=220)
        st.dataframe(best_by_target.reset_index().rename(columns={"cv_mae": "CV MAE (USD)"}), hide_index=True, width="stretch")
        st.caption("Ridge đã thử riêng target identity và log1p; chính sách chọn target được ghi trong JSON kết quả.")
    else:
        tuning = read_csv(TUNING_CSV[model_name]).sort_values("cv_mae").reset_index(drop=True)
        prefix = "param_regressor__estimator__"
        parameter_cols = [c for c in tuning.columns if c.startswith(prefix)]
        best_candidates = tuning.head(12).copy()
        best_candidates["candidate"] = [f"Thử nghiệm {i + 1}" for i in range(len(best_candidates))]
        chart = best_candidates.set_index("candidate")[["cv_mae"]]
        st.bar_chart(chart, color="#365a49", height=320)
        show_cols = parameter_cols + ["cv_mae", "cv_mae_std", "rank_test_score"]
        st.dataframe(best_candidates[show_cols], hide_index=True, width="stretch")
        st.caption(f"Đọc trực tiếp `{TUNING_CSV[model_name]}`; bảng hiển thị 12 cấu hình CV tốt nhất.")

    source_csv = (ROOT / TUNING_CSV[model_name]) if model_name != "Ridge" else ROOT / "results/member_a/ridge_cv_log.csv"
    st.caption(f"CSV tuning hiện tại: {source_csv.relative_to(ROOT)} · kết quả được tạo bởi GridSearchCV/RandomizedSearchCV trong script thành viên.")

    residual_figures = {
        "ElasticNet": "enet_residuals.png",
        "LightGBM": "lgbm_residuals.png",
    }
    if model_name in residual_figures:
        with st.expander("Residual plot từ lần thực nghiệm"):
            figure(residual_figures[model_name], f"Residual analysis · {model_name}")

    lesson("03_modeling_and_evaluation.md")
    with st.expander("Mở script training đang tạo kết quả này"):
        source = ROOT / MODEL_CODE[model_name]
        st.code(source.read_text(encoding="utf-8"), language="python", line_numbers=True)


render()
