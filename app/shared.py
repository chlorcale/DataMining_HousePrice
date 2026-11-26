"""Shared data loaders, study metadata, and presentation helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
LESSONS = ROOT / "docs" / "tutorials"

MODELS = {
    "Ridge": {
        "owner": "Member A", "metrics": "results/member_a/ridge_metrics.json",
        "title": "Ridge Regression", "type": "Tuyến tính · regularization L2",
        "preprocessing": "Numeric: median + missing indicators + StandardScaler. Categorical: Unknown + OneHotEncoder.",
        "intuition": "Tìm một mặt phẳng dự đoán và phạt hệ số lớn. Hệ số được co nhỏ, giúp ổn định khi predictors tương quan.",
        "explain_key": "top_coefficients_by_absolute_value", "explain_value": "coefficient",
        "explain_label": "Hệ số đã chuẩn hóa (target log1p)",
    },
    "Random Forest": {
        "owner": "Member A", "metrics": "results/member_a/random_forest_metrics.json",
        "title": "Random Forest Regressor", "type": "Ensemble cây · bagging",
        "preprocessing": "Numeric: median + missing indicators. Categorical: Unknown + OrdinalEncoder. Không scale.",
        "intuition": "Huấn luyện nhiều cây trên bootstrap samples và trung bình các dự đoán. Tập biến ngẫu nhiên giúp giảm tương quan giữa các cây.",
        "explain_key": "feature_importance_top20", "explain_value": "importance",
        "explain_label": "Importance theo impurity",
    },
    "ElasticNet": {
        "owner": "Member B", "metrics": "results/member_b/elasticnet_metrics.json",
        "title": "ElasticNet", "type": "Tuyến tính · regularization L1 + L2",
        "preprocessing": "Numeric: median + missing indicators + StandardScaler. Categorical: Unknown + OneHotEncoder.",
        "intuition": "Kết hợp Ridge và Lasso. L1 có thể đưa hệ số về 0; L2 giúp các biến tương quan chia sẻ tín hiệu ổn định hơn.",
        "explain_key": "coefficients.top15_by_abs", "explain_value": "coef",
        "explain_label": "Hệ số đã chuẩn hóa (target log1p)",
    },
    "LightGBM": {
        "owner": "Member B", "metrics": "results/member_b/lightgbm_metrics.json",
        "title": "LightGBM Regressor", "type": "Ensemble cây · gradient boosting",
        "preprocessing": "Numeric NaN giữ nguyên. Categorical dùng pandas category. Không scale.",
        "intuition": "Thêm cây tuần tự để sửa sai số còn lại. Histogram tăng tốc chia nhánh; leaf-wise cần giới hạn số lá và regularization.",
        "explain_key": "importance_top20_gain", "explain_value": "gain_pct",
        "explain_label": "Gain importance (%)",
    },
}


@st.cache_data(show_spinner=False)
def load_data():
    from src.data import read_dataset
    from src.protocol import holdout_indices

    X, y = read_dataset()
    dev_idx, test_idx = holdout_indices(len(y))
    return X, y, dev_idx, test_idx


@st.cache_data(show_spinner=False)
def load_results():
    output = {}
    for name, info in MODELS.items():
        path = ROOT / info["metrics"]
        if path.exists():
            output[name] = json.loads(path.read_text(encoding="utf-8"))
    return output


@st.cache_data(show_spinner=False)
def load_static_model():
    path = ROOT / "docs" / "model" / "ridge_model.json"
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_csv(relative_path: str):
    return pd.read_csv(ROOT / relative_path)


def currency(value: float) -> str:
    return f"${float(value):,.0f}"


def page_hero(eyebrow: str, title: str, description: str, number: str | None = None):
    number_html = f'<span class="hero-number">{number}</span>' if number else ""
    st.markdown(
        f"""
        <section class="hero-card">
          <div class="hero-copy">
            <div class="hero-eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
          </div>
          {number_html}
        </section>
        """,
        unsafe_allow_html=True,
    )


def scope_notice():
    st.markdown(
        '<div class="scope-note"><b>Phạm vi:</b> dữ liệu giao dịch lịch sử tại Ames, Iowa; giá bằng USD. '
        'Nghiên cứu này không định giá hiện tại và không đại diện thị trường Việt Nam.</div>',
        unsafe_allow_html=True,
    )


def lesson(filename: str):
    path = LESSONS / filename
    if path.exists():
        st.markdown(path.read_text(encoding="utf-8"))
    else:
        st.warning(f"Chưa có tài liệu hướng dẫn: {filename}")


def figure(filename: str, caption: str | None = None):
    path = ROOT / "report" / "figures" / filename
    if path.exists():
        st.image(str(path), caption=caption, width="stretch")
    else:
        st.info(f"Chưa tìm thấy biểu đồ {filename} trong report/figures.")


def metric_table(results: dict) -> pd.DataFrame:
    rows = []
    for name in MODELS:
        if name not in results:
            continue
        item = results[name]
        cv = item.get("cv_mae", {})
        rows.append({
            "Thuật toán": name,
            "CV MAE ± SD (USD)": f"{currency(cv.get('mean', float('nan')))} ± {currency(cv.get('std', float('nan')))}",
            "Holdout MAE": currency(item["test_mae"]),
            "Holdout RMSE": currency(item["test_rmse"]),
            "R²": f"{item['test_r2']:.3f}",
            "MAPE": f"{item.get('test_mape_pct', float('nan')):.2f}%",
        })
    return pd.DataFrame(rows)


def explainability_data(name: str, payload: dict) -> pd.DataFrame:
    key = MODELS[name]["explain_key"]
    entries = payload.get("coefficients", {}).get("top15_by_abs", []) if key == "coefficients.top15_by_abs" else payload.get(key, [])
    if not entries:
        return pd.DataFrame()
    return pd.DataFrame(entries).head(15).set_index("feature")[[MODELS[name]["explain_value"]]]


def inject_design():
    st.markdown(
        """
        <style>
          @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
          :root {--ink:#263b32;--green:#365a49;--muted:#68766e;--paper:#faf8f3;--cream:#f0ede4;--orange:#c27b4c;--line:#dedbd2;}
          .stApp {background:var(--paper);color:var(--ink);font-family:'DM Sans',Arial,sans-serif;}
          [data-testid="stHeader"] {background:rgba(250,248,243,.96);}
          .block-container {max-width:1200px;padding:1.7rem 2.4rem 3.5rem;}
          [data-testid="stSidebar"] {border-right:1px solid var(--line);}
          [data-testid="stSidebar"] > div:first-child {background:var(--cream);}
          [data-testid="stSidebarNav"] a {border-radius:0!important;color:#45554c;}
          [data-testid="stSidebarNav"] a[aria-current="page"] {background:#e5e0d4!important;border-left:3px solid var(--orange);color:var(--ink)!important;font-weight:700;}
          .app-brand {display:flex;align-items:center;gap:11px;color:var(--ink);font-size:17px;font-weight:700;line-height:1.2;margin:.25rem 0 .2rem;}
          .app-brand-mark {width:38px;height:38px;display:grid;place-items:center;background:var(--green);color:white;font:24px Georgia,serif;}
          .app-brand small {display:block;margin-top:4px;color:#65746b;font:700 8px 'DM Sans',Arial,sans-serif;letter-spacing:1.5px;}
          .hero-card {display:flex;justify-content:space-between;align-items:center;gap:24px;padding:27px 30px;margin:0 0 22px;border:1px solid var(--line);border-top:3px solid var(--orange);border-radius:0;background:var(--cream);color:var(--ink);box-shadow:none;}
          .hero-copy {max-width:880px;}
          .hero-eyebrow {font-size:.68rem;letter-spacing:.16em;font-weight:700;color:#934e27;text-transform:uppercase;margin-bottom:9px;}
          .hero-card h1 {color:var(--ink)!important;font:600 clamp(2rem,4vw,3.15rem)/1.12 'Playfair Display',Georgia,serif;letter-spacing:-.025em;margin:0 0 10px;}
          .hero-card p {color:#46564d;margin:0;font-size:.98rem;line-height:1.7;max-width:78ch;}
          .hero-number {font:500 4.8rem/1 'Playfair Display',Georgia,serif;color:#78877d;}
          .prediction-card {margin:18px 0;padding:25px 28px;border:1px solid #29473a;border-left:4px solid var(--orange);border-radius:0;background:var(--green);color:#fff;box-shadow:none;}
          .prediction-card .hero-eyebrow {color:#f4cfad;margin-bottom:5px;}
          .prediction-value {font:600 clamp(2.3rem,5vw,3.8rem)/1.2 'Playfair Display',Georgia,serif;color:#fff;}
          .prediction-card p {color:#edf2ed;margin:.4rem 0 0;}
          .scope-note {background:#fff;border:1px solid var(--line);border-left:3px solid var(--orange);border-radius:0;padding:12px 16px;color:var(--ink);margin:0 0 22px;}
          [data-testid="stMetric"] {background:#fff;border:1px solid var(--line);padding:14px 16px;border-radius:0;box-shadow:none;}
          [data-testid="stMetric"], [data-testid="stMetricLabel"], [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {color:#263b32 !important;}
          [data-testid="stMetricLabel"] p {color:var(--muted)!important;font-size:.76rem;letter-spacing:.03em;}
          [data-testid="stMetricValue"] {font:600 1.75rem 'Playfair Display',Georgia,serif;}
          .section-caption {color:#56655d;font-size:.88rem;line-height:1.55;}
          h1,h2,h3 {font-family:'Playfair Display',Georgia,serif!important;color:var(--ink);}
          h2 {font-size:1.65rem!important;margin-top:1.8rem!important;}
          h3 {font-size:1.2rem!important;}
          [data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"], [data-testid="stDataFrame"], [data-testid="stImage"] {border-radius:0!important;}
          [data-testid="stVerticalBlockBorderWrapper"] {border-color:var(--line)!important;background:rgba(255,255,255,.72);}
          [data-testid="stExpander"] {border-color:var(--line)!important;border-radius:0!important;background:rgba(255,255,255,.72);}
          [data-testid="stDataFrame"] {border:1px solid var(--line);}
          [data-testid="stAlert"], [data-testid="stAlert"] > div {border-radius:0!important;}
          [data-testid="stCode"] {border-left:3px solid var(--orange);border-radius:0!important;}
          .stButton>button,.stFormSubmitButton>button {border-radius:1px!important;transition:background .15s ease,color .15s ease;}
          .stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"] {background:var(--green);border:1px solid var(--green);}
          .stButton>button[kind="primary"]:hover,.stFormSubmitButton>button[kind="primary"]:hover {background:#29473a;border-color:#29473a;}
          input,textarea,[data-baseweb="select"]>div {border-radius:1px!important;}
          div[data-testid="stTabs"] button[role="tab"] {font-weight:650;border-radius:0!important;}
          @media(max-width:700px) {.block-container{padding:1rem 1rem 2rem}.hero-card{padding:22px 19px}.hero-number{display:none}}
        </style>
        """,
        unsafe_allow_html=True,
    )
