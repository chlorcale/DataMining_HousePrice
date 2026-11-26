# 01 · Hiểu bài toán và dữ liệu

## Từ câu hỏi nghiệp vụ đến biến mục tiêu

Nghiên cứu hỏi: **các thuộc tính của một căn nhà liên hệ như thế nào với giá giao dịch đã ghi nhận?** Đây là hồi quy có giám sát. Mỗi dòng đã có đầu vào `X` và nhãn liên tục `SalePrice` (USD). Mô hình học một hàm `f(X)` để dự đoán giá của các giao dịch chưa dùng khi fit.

Dataset trong repo là Ames Housing: 2.930 dòng, 79 predictor gốc. Dataset mô tả Ames, Iowa; không phải Vietnam Real Estates. Một số cột mô tả thời điểm/điều kiện bán chỉ có sau khi giao dịch, vì vậy nghiên cứu 86 feature là phân tích giá lịch sử, chưa phải dự báo trước khi bán.

## Đọc dữ liệu và khóa giao thức split

```python
from src.data import read_dataset, summarize
from src.protocol import holdout_indices

X, y = read_dataset()  # y là sale_price, đơn vị USD
dev_idx, test_idx = holdout_indices(len(y))  # test_size=.2, seed=42

X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
print(summarize(X_dev, y_dev))
```

`development` dùng để chọn cấu hình bằng cross-validation. `holdout` chỉ dành cho đánh giá cuối. Trước khi nhìn giá trị, cần hiểu ý nghĩa cột, đơn vị, dữ liệu thiếu và thời điểm biến được biết.

## EDA có thể trả lời điều gì?

- Histogram giá cho thấy phân phối lệch phải; một số nhà giá cao tạo đuôi dài.
- Tỷ lệ thiếu phân biệt “không có tiện ích” với “không ghi nhận”. Không phải cứ thiếu là xóa dòng.
- Phân tích giá theo `overall_qual`, `neighborhood` hoặc diện tích giúp đặt giả thuyết, nhưng không chứng minh nguyên nhân.
- IQR tạo cờ để xem xét ngoại biên. Cần xem record và ngữ cảnh trước khi loại bỏ.

```python
price_summary = y_dev.describe(percentiles=[.25, .50, .75, .95, .99])
missing_rate = X_dev.isna().mean().sort_values(ascending=False)
q1, q3 = y_dev.quantile([.25, .75])
iqr = q3 - q1
review = y_dev[(y_dev < q1 - 1.5 * iqr) | (y_dev > q3 + 1.5 * iqr)]
```

Biểu đồ trong trang Dữ liệu & EDA được xuất từ development data và nằm trong `report/figures/`. Như vậy, cách nhìn EDA nhất quán với giao thức đã chốt.
