"""Ames Housing CO3029 learning app."""

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.shared import inject_design

st.set_page_config(
    page_title="Ames Housing · CO3029",
    page_icon=":material/home:",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_design()

with st.sidebar:
    st.markdown(
        '<div class="app-brand"><span class="app-brand-mark">A</span><span>Ames Housing<small>CO3029 · DATA MINING</small></span></div>',
        unsafe_allow_html=True,
    )
    st.caption("HỌC KỲ HK261")
    st.markdown("---")
    st.markdown("Nghiên cứu học thuật về giá giao dịch lịch sử tại Ames, Iowa.")
    st.markdown("---")
    st.caption("Nguồn dữ liệu: Ames Housing · 2.930 giao dịch")

PAGES_DIR = Path(__file__).resolve().parent / "pages"
navigation = st.navigation(
    {
        "NGHIÊN CỨU": [
            st.Page(str(PAGES_DIR / "overview.py"), title="Tổng quan", icon=":material/home:", url_path="overview", default=True),
            st.Page(str(PAGES_DIR / "data_eda.py"), title="Dữ liệu & EDA", icon=":material/database:", url_path="data-eda"),
            st.Page(str(PAGES_DIR / "preprocessing.py"), title="Tiền xử lý & Python", icon=":material/cleaning_services:", url_path="preprocessing"),
            st.Page(str(PAGES_DIR / "modeling.py"), title="Thuật toán & đánh giá", icon=":material/query_stats:", url_path="models-evaluation"),
        ],
        "MÔ HÌNH HÓA": [
            st.Page(str(PAGES_DIR / "model_workflow.py"), title="Huấn luyện & lưu model", icon=":material/model_training:", url_path="model-workflow"),
        ],
        "ỨNG DỤNG": [
            st.Page(str(PAGES_DIR / "demo.py"), title="Demo nhập liệu", icon=":material/house:", url_path="demo"),
        ],
    },
    position="sidebar",
)
navigation.run()
