"""Group integration UI skeleton: models intentionally NOT supplied by teammates yet."""
import streamlit as st
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Ames CO3029",layout="centered")
st.title("Ames Housing — CO3029")
st.info("Dữ liệu & preprocessing đã sẵn sàng. Bốn model đang chờ hai thành viên train, đánh giá và bàn giao.")
st.write("Ứng dụng tích hợp sẽ dùng các mô hình được nhóm kiểm chứng, không giả lập dự báo bằng kết quả nghiên cứu cũ.")
manifest=ROOT/'data'/'processed'/'manifest.json'
if manifest.exists():
    m=json.loads(manifest.read_text(encoding="utf8"))
    st.metric("Development samples",m['train_rows'])
    st.metric("Held-out samples",m['test_rows'])
    st.metric("Engineered predictors",m['features_after_ultra_feature_builder'])
st.subheader("Trạng thái thuật toán")
for member,name,models in [('A','member_a',['ridge','random_forest']),('B','member_b',['elasticnet','lightgbm'])]:
    for model in models:
        metrics=ROOT/'results'/name/f'{model}_metrics.json'
        st.write(f"{member} — {model}: {'Đã có bảng metric' if metrics.exists() else 'Chưa bàn giao'}")
st.caption("Không được dự đoán khi chưa có model fitted phù hợp schema và kiểm thử tích hợp.")
