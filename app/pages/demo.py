import numpy as np
import pandas as pd
import streamlit as st

from app.shared import currency, lesson, load_data, load_results, load_static_model, page_hero, scope_notice


def predict(values: dict, model: dict) -> float:
    score = float(model["intercept"])
    for feature, coefficient in zip(model["features"], model["coefficients"]):
        x = float(values[feature["name"]])
        score += float(coefficient) * (x - feature["mean"]) / feature["scale"]
    return max(0.0, float(np.expm1(score)))


def render():
    page_hero(
        "04 · APPLICATION",
        "Thử một kịch bản căn nhà",
        "Nhập các thuộc tính người dùng có thể cung cấp. Model Ridge đã được fit sẵn; sau khi bấm nút, ước lượng được trả về ngay trong phiên.",
        "LIVE",
    )
    scope_notice()
    try:
        model = load_static_model()
        X, _, dev_idx, _ = load_data()
        results = load_results()
    except Exception as exc:
        st.error(f"Không tải được model hoặc dữ liệu: {exc}")
        return

    st.warning("Form này dùng Ridge riêng với 5 feature; không phải Ridge 86 feature hay các model ở bảng nghiên cứu. Nó không đưa ra lời khuyên mua/bán.")
    with st.container(border=True):
        st.subheader("Thông tin bất động sản")
        with st.form("house-price-form", clear_on_submit=False):
            c1, c2, c3 = st.columns(3)
            area = c1.number_input("Diện tích sinh hoạt trên mặt đất", min_value=300, max_value=6000, value=1500, step=50, help="ft² · GrLivArea trong dữ liệu Ames")
            quality = c2.slider("Chất lượng vật liệu và hoàn thiện", 1, 10, 6, help="OverallQual · thang 1–10")
            bedrooms = c3.number_input("Phòng ngủ trên mặt đất", min_value=0, max_value=8, value=3, step=1)
            c4, c5 = st.columns(2)
            baths = c4.number_input("Phòng tắm đầy đủ", min_value=0, max_value=5, value=2, step=1)
            year = c5.number_input("Năm xây dựng", min_value=1872, max_value=2010, value=2000, step=1)
            submitted = st.form_submit_button("Tính giá dự đoán", type="primary", width="stretch")

    info, metric = st.columns([1.3, 1])
    with info:
        st.markdown("**Mô hình**  \nRidge Regression · 5 biến numeric · target `log1p(SalePrice)`")
        st.caption("Không fit lại khi dự đoán: hệ số và scaler được nạp từ model JSON đã kiểm tra đối chiếu với bản static.")
    with metric:
        demo_results = results.get("Ridge", {})
        demo_mae = model.get("test_mae", demo_results.get("test_mae"))
        if demo_mae is not None:
            st.metric("MAE holdout của model demo", currency(demo_mae))

    if submitted:
        values = {
            "gr_liv_area": area,
            "overall_qual": quality,
            "bedroom_abvgr": bedrooms,
            "full_bath": baths,
            "year_built": year,
        }
        estimated = predict(values, model)
        st.markdown(
            f'<section class="prediction-card"><div class="hero-eyebrow">ƯỚC LƯỢNG LỊCH SỬ · AMES, IOWA</div>'
            f'<div class="prediction-value">{currency(estimated)}</div>'
            f'<p>Giá dự đoán theo Ridge 5 feature · USD</p></section>',
            unsafe_allow_html=True,
        )

        dev_X = X.iloc[dev_idx]
        contributions = []
        out_of_range = []
        for feature, coefficient in zip(model["features"], model["coefficients"]):
            name = feature["name"]
            value = float(values[name])
            standardized = (value - feature["mean"]) / feature["scale"]
            contribution = float(coefficient) * standardized
            low, high = dev_X[name].quantile([.01, .99])
            if value < low or value > high:
                out_of_range.append(f"{name}: phạm vi thường gặp khoảng {low:.0f}–{high:.0f}")
            contributions.append({"Feature": name, "Giá trị nhập": value, "Đóng góp log-score": contribution})

        if out_of_range:
            st.warning("Đầu vào ngoài vùng 1–99% của development; dự đoán có thể là ngoại suy. " + " · ".join(out_of_range))
        with st.expander("Vì sao model đưa ra ước lượng này?"):
            frame = pd.DataFrame(contributions).sort_values("Đóng góp log-score", key=lambda s: s.abs(), ascending=False)
            st.dataframe(frame.round(4), hide_index=True, width="stretch")
            st.caption("Đóng góp được tính trên log-score sau chuẩn hóa; không phải tác động nhân quả hoặc khuyến nghị giao dịch.")
        st.info("Đây là dự đoán điểm, không có khoảng tin cậy 95%. Giá tham chiếu lịch sử ở Ames, Iowa; không dùng để thẩm định nhà ở Việt Nam.")

    st.subheader("Phạm vi của form")
    st.markdown(
        "Form yêu cầu năm thuộc tính để giữ trải nghiệm đơn giản. Nó không chứa vị trí và phần lớn mô tả tiện ích; "
        "vì thế độ chính xác thấp hơn pipeline nghiên cứu 86 feature. Các model 86 feature vẫn được so sánh riêng trên trang **Thuật toán & đánh giá**."
    )
    lesson("04_interactive_demo.md")


render()
