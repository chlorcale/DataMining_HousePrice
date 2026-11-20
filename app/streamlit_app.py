"""UI integration preview. No hidden or inherited trained model."""
from pathlib import Path
import streamlit as st
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
st.set_page_config(page_title='Ames | CO3029',page_icon='🏡',layout='wide')
st.markdown("""<style>
[data-testid="stAppViewContainer"] {background:#f5f8f6;}
.block-container{max-width:1150px;padding-top:2rem;}
h1,h2,h3{color:#173e33;}
[data-testid="stMetric"]{background:white;border:1px solid #e2ebe5;border-radius:14px;padding:17px;}
</style>""",unsafe_allow_html=True)
st.title('🏡 Ames Housing · Data Mining')
st.caption('CO3029 — Không hiển thị dự báo khi chưa tích hợp model thật của nhóm')
col1,col2,col3=st.columns(3)
col1.metric('Quan sát nguồn','2.930')
col2.metric('Thuật toán dự kiến','4')
rows=[]
path=ROOT/'outputs/integration/model_comparison.csv'
if path.exists():
    rows=pd.read_csv(path)
    col3.metric('Đã có kết quả',str(len(rows)))
else: col3.metric('Đã có kết quả','0')
tab1,tab2,tab3=st.tabs(['Tổng quan','So sánh thuật toán','Dự báo'])
with tab1:
    st.subheader('Bộ dữ liệu')
    st.write('Ames Housing, Iowa (2006–2010). Nguồn CSV được lưu kèm project. Không phải dữ liệu nhà đất Việt Nam.')
    img=ROOT/'outputs/eda/price_hist.png'
    if img.exists():st.image(str(img),caption='Phân phối giá trên tập phát triển')
    else:st.info('Chạy `python scripts/eda.py` để tạo biểu đồ EDA.')
with tab2:
    st.subheader('Bảng đánh giá dùng chung')
    if path.exists() and len(rows):st.dataframe(rows,use_container_width=True,hide_index=True)
    else:st.warning('Chưa nhận đủ kết quả từ hai thành viên. Không có metric giả.')
with tab3:
    st.subheader('Mô hình dự đoán — chờ tích hợp')
    st.info('Sau khi cả nhóm chọn model bằng CV, người tích hợp sẽ nối đúng các input thực tế với model đã export. Hiện chưa bật dự đoán để tránh dùng nhầm model hoặc giả số.')
