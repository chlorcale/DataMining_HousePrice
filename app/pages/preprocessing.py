from pathlib import Path

import streamlit as st

from app.shared import ROOT, lesson, page_hero, read_csv, scope_notice


def render():
    page_hero(
        "02 · DATA PREPARATION",
        "Xây pipeline để tránh leakage",
        "Tạo đặc trưng theo từng record; học imputer, encoder và scaler chỉ trong training fold; áp dụng cách tiền xử lý phù hợp từng họ model.",
        "79 → 86",
    )
    scope_notice()

    st.subheader("Quy trình tiền xử lý")
    stages = [
        ("Raw X", "79 predictor gốc"),
        ("FeatureBuilder", "+7 đặc trưng dòng"),
        ("Fold preprocessing", "impute · encode · scale"),
        ("Estimator", "fit trên training fold"),
    ]
    columns = st.columns(4)
    for col, (title, description) in zip(columns, stages):
        with col, st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)
    st.info("FeatureBuilder chỉ tạo các phép biến đổi theo dòng. Thống kê học từ dữ liệu (median, mean, scale, vocab) nằm trong sklearn Pipeline nên được fit lại ở từng fold.")

    st.subheader("Dữ liệu CSV tại các mốc preprocessing")
    csv_sources = {
        "Loại dữ liệu theo cột": "data/eda/dev_feature_types.csv",
        "Tỷ lệ missing theo cột": "data/eda/dev_missing_fraction.csv",
        "Training features sau row-wise FeatureBuilder": "data/processed/train_features.csv",
    }
    selected_csv = st.selectbox("Chọn artifact CSV", list(csv_sources), key="preprocess_csv")
    relative_path = csv_sources[selected_csv]
    csv_data = read_csv(relative_path)
    st.caption(f"`{relative_path}` · {len(csv_data):,} dòng. Đây là bảng dữ liệu thật của project.")
    if selected_csv == "Tỷ lệ missing theo cột":
        csv_data = csv_data.rename(columns={csv_data.columns[0]: "Feature", csv_data.columns[1]: "Missing fraction"})
        csv_data["Missing %"] = (100 * csv_data["Missing fraction"]).round(2)
        st.dataframe(csv_data.head(20), width="stretch", hide_index=True)
        st.bar_chart(csv_data.head(15).set_index("Feature")[["Missing %"]], color="#c27b4c", height=370)
    elif selected_csv == "Loại dữ liệu theo cột":
        csv_data = csv_data.rename(columns={csv_data.columns[0]: "Feature", csv_data.columns[1]: "Data type"})
        st.dataframe(csv_data.head(30), width="stretch", hide_index=True)
    else:
        targets = read_csv("data/processed/train_targets.csv")
        preview = csv_data.head(12).copy()
        preview.insert(0, "sale_price_USD", targets["sale_price"].head(12))
        visible = [c for c in ["sale_price_USD", "neighborhood", "overall_qual", "gr_liv_area", "lot_area", "quality_x_living_area", "age_at_sale", "total_bathrooms", "garage_presence"] if c in preview]
        st.dataframe(preview[visible], width="stretch", hide_index=True)
        st.caption("Features CSV đã có 7 đặc trưng mới nhưng vẫn giữ category và NaN; nó chưa phải ma trận đã impute/encode/scale.")

    st.subheader("Mỗi loại dữ liệu cần xử lý khác nhau")
    cols = st.columns(3)
    cards = [
        ("Ridge · ElasticNet", "Median + missing indicators", "OneHotEncoder", "StandardScaler"),
        ("Random Forest", "Median + missing indicators", "OrdinalEncoder", "Không scale"),
        ("LightGBM", "Giữ numeric NaN", "Pandas category", "Không scale"),
    ]
    for col, (title, numeric, categorical, scale) in zip(cols, cards):
        with col, st.container(border=True):
            st.markdown(f"### {title}")
            st.write(f"**Numeric**  \n{numeric}")
            st.write(f"**Categorical**  \n{categorical}")
            st.write(f"**Scale**  \n{scale}")

    lesson("02_preprocessing.md")

    st.subheader("Source code của dự án")
    for title, filename in [
        ("FeatureBuilder · 7 đặc trưng mới", "src/features.py"),
        ("Preprocessor factories · 3 nhánh model", "src/preprocessing.py"),
        ("Frozen split và K-fold protocol", "src/protocol.py"),
    ]:
        path = ROOT / filename
        with st.expander(title):
            if path.exists():
                st.code(path.read_text(encoding="utf-8"), language="python", line_numbers=True)
            else:
                st.warning(f"Không tìm thấy {filename}")

    with st.container(border=True):
        st.subheader("Kiểm tra chống leakage")
        st.markdown(
            "1. Chia development/holdout bằng seed 42.\n"
            "2. Đưa pipeline chưa fit vào cross-validation 5 fold, seed 123.\n"
            "3. Chọn siêu tham số chỉ từ development CV.\n"
            "4. Fit cấu hình đã chốt trên toàn development.\n"
            "5. Dùng holdout đúng giao thức đánh giá cuối; không dùng để chọn feature hoặc tuning."
        )


render()

