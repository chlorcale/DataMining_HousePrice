"""Integration UI skeleton for the historical Ames transaction study."""
import streamlit as st
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Ames CO3029",layout="centered")
st.title("Ames Housing — CO3029")
st.info("Phạm vi dữ liệu: Ames Housing, giao dịch lịch sử ở Iowa, giá USD. Ứng dụng chưa phải công cụ định giá trước giao dịch.")
st.write("Các trường thời điểm và điều kiện bán có thể không biết trước giao dịch. Form demo cần schema đầy đủ hoặc một tập đặc trưng pre-sale đã kiểm chứng; cho tới khi tích hợp đúng model thì ứng dụng không tạo dự báo giả lập.")
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
st.caption("Chỉ hiển thị dự báo khi model fitted, schema đầu vào và quy trình tích hợp đã được kiểm chứng.")
