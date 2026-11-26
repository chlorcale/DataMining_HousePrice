# 02 · Tiền xử lý, tạo đặc trưng và chống leakage

Tiền xử lý là một phần của mô hình. Các thống kê như median, trung bình, độ lệch chuẩn và tập category phải học từ training partition; validation/holdout chỉ được biến đổi bằng những thống kê đã học.

## Feature engineering: từ 79 lên 86 predictor

`src/features.py` chứa `FeatureBuilder`, tạo bảy cột theo từng dòng:

| Đặc trưng | Ý tưởng |
|---|---|
| `quality_x_living_area` | Chất lượng × diện tích sinh hoạt |
| `age_at_sale` | Năm bán − năm xây |
| `years_since_remodel` | Năm bán − năm remodel |
| `total_bathrooms` | Phòng tắm đầy đủ + nửa phòng tắm quy đổi 0.5 |
| `porch_total` | Tổng diện tích các loại hiên |
| `living_plus_total_basement_sf` | Diện tích sinh hoạt + tổng basement |
| `garage_presence` | Cờ có garage |

```python
from src.features import FeatureBuilder

# Biến đổi theo từng dòng; chưa ước lượng median/scale/category nào.
X_engineered = FeatureBuilder(engineered=True).fit_transform(X)
assert X_engineered.shape[1] == 86
```

## Ba nhánh preprocessing

```python
from src.preprocessing import make_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

model = Pipeline([
    ("preprocess", make_preprocessor("ridge")),
    ("estimator", Ridge(alpha=1.0)),
])
```

- **Ridge / ElasticNet:** numeric median imputation + missing indicators + StandardScaler; category Unknown fill + OneHotEncoder.
- **Random Forest:** numeric median imputation + missing indicators; category Unknown fill + OrdinalEncoder; không scale.
- **LightGBM:** numeric NaN được giữ cho native handling; categorical dùng pandas category; không scale.

## Vì sao phải gói trong Pipeline?

```python
from sklearn.model_selection import cross_validate
from src.protocol import make_cv

scores = cross_validate(
    model, X_dev, y_dev,
    cv=make_cv(),
    scoring="neg_mean_absolute_error",
)
cv_mae_usd = -scores["test_score"]
```

`cross_validate` clone và fit toàn bộ pipeline lại trong mỗi fold. Nếu impute/scale/encode trên toàn `X_dev` trước CV, fold validation đã ảnh hưởng preprocessing và kết quả lạc quan hơn thực tế. `data/fitted_previews/` chỉ để kiểm tra đầu ra; không phải đầu vào cho CV.

## Target lệch phải

Nhóm so sánh `SalePrice` gốc với giả thuyết `log1p(SalePrice)` bằng `TransformedTargetRegressor`. Khi chấm MAE/RMSE/R²/MAPE, dự đoán được đổi về USD trước. Log-transform không được áp dụng bừa lên X; đây là lựa chọn cho y đã được thử nghiệm trong CV.
