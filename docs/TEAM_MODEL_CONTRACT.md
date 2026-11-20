# Hợp đồng triển khai thuật toán — CO3029

## Dữ liệu & bất biến

- Gốc: 2930 dòng, 79 predictors. Split theo `train_test_split(test_size=.2,random_state=42)`; 2344 dev + 586 test.
- Đã chạy `FeatureBuilder(engineered=True)`: **86 cột** trong `data/processed/*.csv`. CSV có thể còn NaN và text *vì đây là lớp biến đổi xác định theo từng dòng*, chưa phải input đã fit thống kê.
- `data/fitted_previews/` có các bản **đã fit trên cả dev** để trực quan kiểm tra preprocessing: linear/tree là ma trận số không missing. **KHÔNG sử dụng các bản này để đo 5-fold CV**. Nếu sử dụng chúng để CV thì validation đã tham gia học imputer, encoder, scaler.
- Với mỗi fold, dùng `src.preprocessing.make_*_preprocessor()` *chưa fit* làm bước `preprocess` trong estimator Pipeline; scikit-learn sẽ fit lại riêng trên training partition của fold.
- Hai người không sửa các tệp dữ liệu chung. Nếu đề xuất thay đổi, thông báo trước, lập ablation riêng.

## Nhóm A: Ridge và RandomForest

```
from sklearn.pipeline import Pipeline
from src.preprocessing import make_linear_preprocessor, make_tree_preprocessor
# Lắp estimator Ridge/RandomForest và tự xây search space, CV, metrics, tests.
```

## Nhóm B: ElasticNet và LightGBM

```
from sklearn.pipeline import Pipeline
from src.preprocessing import make_linear_preprocessor, make_lightgbm_preprocessor
# Lắp estimator ElasticNet/LGBMRegressor và tự xây search space, CV, metrics, tests.
```

**Lưu ý quan trọng:** `src.preprocessing` bao gồm `FeatureBuilder` gốc. Khi sử dụng factory, phải đọc **dữ liệu gốc trước khi FE** bằng `src.data.read_dataset()` rồi split theo `src.protocol.holdout_indices()`. **Không** gọi factory trên `data/processed/train_features.csv` vì sẽ chạy FeatureBuilder lần hai (dù cùng tên cột thường bị ghi đè), và tránh các bước biến đổi trùng. Nếu muốn bắt đầu từ `data/processed/*.csv`, sử dụng phiên bản factory `engineered=False`; tài liệu hướng dẫn bên dưới giải thích cách làm.

### CV và đánh giá

- Dùng CV = `src.protocol.make_cv()` (5-fold shuffle seed 123) cho **cả bốn thuật toán**.
- Nhãn mục tiêu `SalePrice` theo USD. Nếu dùng `log1p`, estimator phải dự đoán trở lại USD trước khi đánh giá MAE/RMSE/R²/MAPE. Không được tính các metric USD trên log target.
- Imputation, scaling, encoding và optional feature selection phải fit bên trong mỗi training fold; target transform nằm trong training pipeline.
- Chọn siêu tham số bằng CV trên 2344 dòng. Chỉ dùng 586 dòng test sau khi quyết định chốt. Không tune dựa trên test.
- Ghi tên mô hình, quy trình tiền xử lý, search space, số fold, seed, MAE CV (mean ± std), RMSE/R²/MAPE trên test, thời gian chạy, phiên bản thư viện và các hạn chế.
- `results/member_a/{ridge,random_forest}_metrics.json`; `results/member_b/{elasticnet,lightgbm}_metrics.json`. Model artifacts (kích thước lớn) chỉ commit khi nhóm đồng ý.

## Gợi ý API khi dùng processed CSV

```python
import pandas as pd
from sklearn.pipeline import Pipeline
from src.preprocessing import make_preprocessor
X = pd.read_csv('data/processed/train_features.csv')
y = pd.read_csv('data/processed/train_targets.csv')['sale_price']
# FeatureBuilder đã được chạy trong CSV; không chạy lại:
pre = make_preprocessor('ridge', engineered=False)
# pipeline = Pipeline([('preprocess',pre),('estimator',<MODEL_TU_VIET>)])
```

Tuy nhiên cách sử dụng đầy đủ từ **raw source split** ở trên được ưu tiên, vì dễ kiểm tra nguồn gốc và tái lập.

## Trách nhiệm thành viên

Mỗi người tự xây estimator, tập tham số, quy trình tuning, chạy CV và test, residual/ablation/importance, phân tích overfitting, viết phần báo cáo (`report/member_a.tex` hoặc `report/member_b.tex`).
