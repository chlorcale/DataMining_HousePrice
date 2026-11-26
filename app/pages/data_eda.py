import numpy as np
import pandas as pd
import streamlit as st

from app.shared import lesson, load_data, page_hero, read_csv, scope_notice


def render():
    X, y, dev_idx, _ = load_data()
    X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
    page_hero(
        "01 · DATA UNDERSTANDING",
        "Nhìn dữ liệu trước khi chọn mô hình",
        "Mọi quyết định làm sạch và biến đổi đều bắt đầu từ việc hiểu phân phối, độ thiếu, đơn vị và bối cảnh của từng thuộc tính.",
        "2,344",
    )
    scope_notice()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Development rows", f"{len(y_dev):,}")
    c2.metric("Predictors gốc", f"{X_dev.shape[1]}")
    c3.metric("Median SalePrice", f"${y_dev.median():,.0f}")
    c4.metric("Giá thiếu", f"{int(y_dev.isna().sum())}")

    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Phân phối giá")
        counts, edges = np.histogram(y_dev, bins=24)
        histogram = pd.DataFrame(
            {"Số căn nhà": counts},
            index=[f"{edges[i] / 1000:.0f}–{edges[i + 1] / 1000:.0f}k" for i in range(len(counts))],
        )
        st.bar_chart(histogram, color="#365a49", height=285)
        st.markdown("Đuôi phải kéo dài; `log1p(y)` được thử như giả thuyết và so sánh với target gốc bằng CV.")
    with right, st.container(border=True):
        st.subheader("Dữ liệu thiếu")
        missing_view = read_csv("data/eda/dev_missing_fraction.csv")
        missing_view = missing_view.rename(columns={missing_view.columns[0]: "Feature", missing_view.columns[1]: "Fraction"})
        missing_view["Missing (%)"] = missing_view["Fraction"].mul(100)
        missing_view = missing_view.head(12).sort_values("Missing (%)")
        st.bar_chart(missing_view.set_index("Feature")[["Missing (%)"]], color="#c27b4c", height=285)
        st.markdown("Thiếu có thể mang ý nghĩa (ví dụ không có tiện ích). Cách xử lý phụ thuộc model.")

    st.subheader("Quan hệ cần đặt giả thuyết")
    left, right = st.columns([1.1, 1])
    with left, st.container(border=True):
        scatter = pd.DataFrame({"Diện tích sinh hoạt (ft²)": X_dev["gr_liv_area"], "SalePrice (USD)": y_dev}).dropna()
        scatter = scatter.sample(min(900, len(scatter)), random_state=42)
        st.scatter_chart(scatter, x="Diện tích sinh hoạt (ft²)", y="SalePrice (USD)", color="#365a49", height=330)
        st.caption("Mẫu tối đa 900 điểm từ development. Quan hệ quan sát không chứng minh nhân quả.")
    with right, st.container(border=True):
        quality = read_csv("data/eda/dev_price_by_quality.csv")
        quality["quality"] = quality["quality"].astype(str)
        st.markdown("**Giá trung vị theo chất lượng tổng thể**")
        st.bar_chart(quality.set_index("quality")[["median"]], color="#365a49", height=300)
        st.caption("Các bậc chất lượng cao thường đi kèm giá trung vị cao hơn trong mẫu Ames.")

    neighborhood = read_csv("data/eda/dev_price_by_neighborhood.csv")
    neighborhood = neighborhood.sort_values("median", ascending=False).head(12).sort_values("median")
    st.subheader("Khác biệt giữa các khu vực")
    st.bar_chart(neighborhood.set_index("neighborhood")[["median"]], color="#c27b4c", height=420)
    st.caption("Chỉ xem các khu vực có median cao trong development. Khu vực có số lượng mẫu khác nhau; đây không phải bảng giá phổ quát.")

    correlation = read_csv("data/eda/dev_numeric_price_correlation.csv")
    correlation = correlation.rename(columns={correlation.columns[0]: "feature"})
    correlation = correlation.sort_values("correlation", ascending=False).head(12).sort_values("correlation")
    with st.expander("Tương quan tuyến tính với SalePrice"):
        st.bar_chart(correlation.set_index("feature")[["correlation"]], color="#365a49", height=350)
        st.caption("Pearson correlation chỉ tóm tắt quan hệ tuyến tính từng cặp, không cho biết feature importance hay tác động nhân quả.")

    st.subheader("Bảng CSV của giai đoạn khám phá")
    csv_tables = {
        "Tóm tắt SalePrice": "data/eda/dev_price_summary.csv",
        "Giá theo khu vực": "data/eda/dev_price_by_neighborhood.csv",
        "Giá theo quality": "data/eda/dev_price_by_quality.csv",
        "Tương quan số": "data/eda/dev_numeric_price_correlation.csv",
        "Danh sách outlier để review": "data/eda/dev_outlier_review.csv",
    }
    csv_label = st.selectbox("Chọn bảng EDA", list(csv_tables), key="eda_csv")
    csv_path = csv_tables[csv_label]
    csv_frame = read_csv(csv_path)
    st.caption(f"Nguồn: `{csv_path}` · {len(csv_frame):,} dòng")
    st.dataframe(csv_frame.head(30), width="stretch", hide_index=True)

    q1, q3 = y_dev.quantile([0.25, 0.75])
    iqr = q3 - q1
    flags = (y_dev.lt(q1 - 1.5 * iqr) | y_dev.gt(q3 + 1.5 * iqr))
    st.subheader("Ngoại biên: đánh dấu để xem xét, không xóa tự động")
    st.write(f"Quy tắc IQR gắn cờ {int(flags.sum())} / {len(y_dev):,} giá bán. Một giá cao có thể là giao dịch hợp lệ; cần đối chiếu thuộc tính và nguồn trước khi loại.")
    samples = X_dev.copy()
    samples.insert(0, "SalePrice USD", y_dev)
    columns = [c for c in ["SalePrice USD", "neighborhood", "overall_qual", "gr_liv_area", "lot_area", "year_built", "yr_sold"] if c in samples]
    st.dataframe(samples[columns].head(15), width="stretch", hide_index=True)

    with st.expander("Bài học và Python walkthrough", expanded=False):
        lesson("01_business_and_data.md")


render()
