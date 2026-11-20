# CO3029 Ames Housing

## Phân công

- **Người tích hợp (bạn):** dataset, EDA, Ultra FeatureBuilder, train/test split, fold protocol, hoàn thiện 3 họ preprocessing (Ridge/ElasticNet, RandomForest, native LightGBM), tài liệu kỹ thuật, khung báo cáo/demo, tổng hợp.
- **Member A:** Ridge và RandomForest: estimator, hyperparameter tuning, CV/test evaluation, phân tích kết quả, viết `report/member_a.tex`.
- **Member B:** ElasticNet và LightGBM: tương tự, viết `report/member_b.tex`.

## Đã chạy trước khi phát hành

- `data/processed/train_features.csv`, `test_features.csv`, nhãn và row IDs = **đúng Ultra 86 features**; chưa fit thống kê vì tránh leakage.
- `data/fitted_previews/linear_*` + `tree_*`: **đã impute, encode và (linear) scale**, toàn numeric, không missing. Đã fit *chỉ trên dev*, transform test. **Không dùng bản này cho 5-fold CV**; phải fit lại preprocessing trong mỗi fold.
- `data/fitted_previews/lightgbm_*`: numeric diagnostic view + category levels; native LightGBM sử dụng pandas `category` và cho phép numeric NaN. Với LightGBM hãy dùng factory trực tiếp.
- EDA từ dev-only: `data/eda/`, biểu đồ `report/figures/`.

## Bắt đầu (Python 3.11+, Windows/Linux)

```bash
python -m pip install -r requirements.txt
python scripts/check_project.py
python -m pytest -q
python scripts/collect_results.py
# Báo cáo PDF (nếu cài XeLaTeX): cd report && xelatex -interaction=nonstopmode -halt-on-error main.tex
```

Tái tạo dữ liệu (chỉ khi cần, kết quả đã nằm trong ZIP):

```bash
python scripts/export_shared_data.py
python scripts/export_fitted_previews.py
python scripts/eda.py
```

**Đọc trước khi viết model:** `docs/TEAM_MODEL_CONTRACT.md`.

Các factory dùng **đầu vào từ `src.data.read_dataset()`** hoặc từ `data/processed/train_features.csv` khi truyền `engineered=False`. Tránh chạy feature engineering hai lần. Dùng `sklearn.pipeline.Pipeline([('preprocess', make_preprocessor('ridge')), ('estimator', YourEstimator(...))])` để tất cả các bước được fit trong training fold. Ridge, ElasticNet, RandomForest, LightGBM và tuning/metric đều **CHƯA chạy** trong bản bàn giao nhóm.

**Hạn chế:** Đây là một bản dùng cho nghiên cứu *giá giao dịch lịch sử* như Ultra gốc; có `yr_sold`, `mo_sold`, `sale_type`, `sale_condition`. Muốn dự báo trước khi bán cần xác định và loại các trường không biết trước, sau đó thí nghiệm lại. Thầy yêu cầu hiểu và phản biện khi dùng AI; kết quả cần do từng người đọc, chạy và giải thích lại trung thực.
