import pandas as pd
import streamlit as st

from app.shared import MODELS, currency, load_data, load_results, page_hero, scope_notice


def render():
    X, y, dev_idx, test_idx = load_data()
    results = load_results()
    page_hero(
        "CO3029 · KHAI PHÁ DỮ LIỆU",
        "Từ giao dịch thật đến mô hình có thể giải thích",
        "Một hướng dẫn tương tác qua toàn bộ quy trình: hiểu dữ liệu, chuẩn bị đặc trưng, so sánh bốn thuật toán và thử form dự đoán.",
        "AMES",
    )
    scope_notice()

    a, b, c, d = st.columns(4)
    a.metric("Giao dịch", f"{len(y):,}")
    b.metric("Thuộc tính gốc", f"{X.shape[1]}")
    c.metric("Đặc trưng sau FE", "86")
    d.metric("Thuật toán", f"{len(results)}")

    st.subheader("Lộ trình khám phá")
    steps = [
        ("01", "Hiểu bài toán", "Chốt mục tiêu, đơn vị và thời điểm thông tin được biết."),
        ("02", "Hiểu dữ liệu", "Quan sát phân phối giá, thiếu dữ liệu, tương quan và ngoại biên."),
        ("03", "Tiền xử lý", "Tạo đặc trưng và fit biến đổi trong từng fold để tránh leakage."),
        ("04", "Mô hình & đánh giá", "Dùng cùng CV và holdout để so Ridge, RF, ElasticNet, LightGBM."),
        ("05", "Demo", "Nhập năm biến vào Ridge gọn nhẹ và nhận giá dự đoán tức thì."),
    ]
    columns = st.columns(len(steps))
    for col, (number, title, body) in zip(columns, steps):
        with col, st.container(border=True):
            st.caption(number)
            st.markdown(f"**{title}**")
            st.markdown(f"<span class='section-caption'>{body}</span>", unsafe_allow_html=True)

    st.subheader("Kết quả cross-validation")
    if results:
        rows = [
            {"Thuật toán": name, "CV MAE (USD)": item["cv_mae"]["mean"]}
            for name, item in results.items()
        ]
        chart = pd.DataFrame(rows).set_index("Thuật toán").sort_values("CV MAE (USD)")
        st.bar_chart(chart, height=260, color="#365a49")
        best = min(results, key=lambda name: results[name]["cv_mae"]["mean"])
        st.caption(
            f"CV MAE thấp nhất hiện tại: {best} — {currency(results[best]['cv_mae']['mean'])} trung bình trên 5 folds. "
            "Đây là kết quả trên Ames, không phải bảo đảm cho dữ liệu khác."
        )
    else:
        st.warning("Chưa có file metrics; kiểm tra thư mục results/.")

    st.info(
        f"Giao thức cố định: {len(dev_idx):,} dòng development để tuning/CV; "
        f"{len(test_idx):,} dòng holdout để đánh giá cuối. Holdout không tham gia fit preprocessing."
    )

    with st.expander("Tuyên bố phạm vi và giới hạn nghiên cứu"):
        st.markdown(
            "Dataset là Ames Housing, các giao dịch lịch sử ở Iowa, Hoa Kỳ. "
            "Một số thuộc tính như `yr_sold`, `sale_type`, `sale_condition` có thể chỉ biết sau khi bán. "
            "Vì vậy bốn model 86 feature nghiên cứu giá giao dịch lịch sử, không chứng minh khả năng dự báo trước giao dịch. "
            "Form demo dùng một Ridge riêng với năm feature pre-sale và có MAE cao hơn; hai thí nghiệm không được nhập chung."
        )


render()
