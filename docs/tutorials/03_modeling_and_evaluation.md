# 03 · Huấn luyện, so sánh và giải thích

## Bốn thuật toán trả lời bốn góc nhìn

### Ridge Regression

Tối thiểu hóa sai số bình phương cộng phạt L2: `RSS + α × Σβ²`. Alpha lớn co hệ số mạnh hơn. Numeric cần scale vì penalty phụ thuộc độ lớn hệ số.

### ElasticNet

Kết hợp L1 và L2: `(1−r) × L2 + r × L1`. L1 có thể đưa hệ số về 0; L2 ổn định nhóm biến tương quan. `l1_ratio` điều khiển tỷ lệ hai loại penalty.

### Random Forest

Bagging nhiều decision tree trên bootstrap samples, rồi lấy trung bình dự đoán. Tính ngẫu nhiên ở mẫu và predictors giúp giảm variance so với một cây đơn.

### LightGBM

Gradient boosting xây cây tuần tự để cải thiện loss của ensemble hiện tại. Tăng trưởng leaf-wise và histogram thường nhanh; `num_leaves`, `min_child_samples`, subsampling và regularization kiểm soát độ phức tạp.

## Chọn tham số bằng cùng giao thức

```python
from sklearn.model_selection import GridSearchCV
from src.protocol import make_cv

search = GridSearchCV(
    estimator=model,
    param_grid={"estimator__alpha": [0.01, 0.1, 1.0, 10.0]},
    cv=make_cv(),
    scoring="neg_mean_absolute_error",
    refit=False,
)
search.fit(X_dev, y_dev)
```

Ví dụ trên minh họa nguyên tắc; dùng tham số `best_params` và search space trong `members/member_a/` hoặc `members/member_b/` để tái lập thí nghiệm chính. CV là 5-fold shuffled KFold, seed 123, trên 2.344 development rows.

## Đọc metric

- **MAE:** độ lệch tuyệt đối trung bình, dễ diễn giải bằng USD.
- **RMSE:** cùng đơn vị nhưng phạt lỗi lớn mạnh hơn.
- **R²:** so với dự đoán trung bình; có thể âm trên holdout.
- **MAPE:** lỗi tương đối trung bình; nhạy với target gần 0.
- **CV mean ± SD:** sai số trung bình và dao động giữa folds; SD không phải khoảng tin cậy 95%.

586 dòng holdout được giữ riêng. Bảng trong ứng dụng hiển thị metric đã được các script training ghi ra JSON; không chạy tuning lại khi mở app.

## Giải thích có giới hạn

Hệ số tuyến tính phụ thuộc scaling và target transform. Random Forest impurity importance có thể ưu tiên biến nhiều mức; LightGBM gain đo mức loss giảm khi split. Đây là tín hiệu mô hình sử dụng, không phải tác động nhân quả hay khuyến nghị giao dịch. Xem thêm hình residual/importance trong `report/figures/`.
