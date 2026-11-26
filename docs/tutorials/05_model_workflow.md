# 05 · Từ dữ liệu đến mô hình có thể nạp lại

Bài này nối các bước của quy trình khai phá dữ liệu với phần **Huấn luyện & lưu model** trong Streamlit. Ta theo dõi dữ liệu Ames Housing: một dòng là một căn nhà đã giao dịch, `SalePrice` là giá lịch sử bằng USD.

## 1. Xác định đúng tác vụ và lấy mẫu để xem

`SalePrice` là một đại lượng liên tục nên bài toán chính là **hồi quy có giám sát**. Không có nhãn lớp bị mất cân bằng như “bép phì/không bép phì”. K-Means cũng không thay thế mô hình dự báo giá: nó gom các căn nhà giống nhau, nhưng không học trực tiếp giá cần dự đoán.

Để trình bày record đại diện cho các mức giá, ta chia riêng target của development thành ba phân vị. Đây chỉ là cách chọn record để quan sát; không sửa target và không dùng nhóm đó làm nhãn huấn luyện.

```python
import pandas as pd
from src.data import read_dataset
from src.protocol import holdout_indices

X, y = read_dataset()                       # predictor gốc và SalePrice
dev_idx, holdout_idx = holdout_indices(len(y))
X_dev, y_dev = X.iloc[dev_idx], y.iloc[dev_idx]

price_band = pd.qcut(
    y_dev,
    q=3,
    labels=["thấp", "giữa", "cao"],
)
sample = X_dev.loc[price_band.eq("cao")].sample(10, random_state=42)
```

Trong ứng dụng, bảng record có thêm giá để người học nhìn được ngữ cảnh. Holdout được giữ riêng và không dùng để chọn những record minh họa.

## 2. Tiền xử lý và tìm siêu tham số bằng CV

Median imputation, scaling và encoding phải được **fit bên trong từng fold**. Nếu tính median hoặc danh sách category trên toàn bộ dữ liệu trước CV, thông tin từ fold validation có thể lọt vào bước huấn luyện. Các factory của dự án ghép `FeatureBuilder` và preprocessing trong cùng `Pipeline` để mỗi fold tự học các bước này.

Đoạn dưới minh họa cấu trúc Ridge với GridSearchCV. Tên tham số có tiền tố vì Ridge nằm trong `TransformedTargetRegressor`, vốn bọc `Pipeline`:

```python
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from src.preprocessing import make_linear_preprocessor

pipeline = Pipeline([
    ("preprocess", make_linear_preprocessor()),
    ("estimator", Ridge()),
])
model = TransformedTargetRegressor(
    regressor=pipeline,
    func=np.log1p,
    inverse_func=np.expm1,
    check_inverse=False,
)
cv = KFold(n_splits=5, shuffle=True, random_state=123)
search = GridSearchCV(
    model,
    param_grid={"regressor__estimator__alpha": np.logspace(-3, 3, 19)},
    scoring="neg_mean_absolute_error",
    cv=cv,
    n_jobs=-1,
)
search.fit(X_dev, y_dev)
print(search.best_params_, -search.best_score_)
```

Mục tiêu tìm kiếm là giảm MAE trên các fold development. Dự án đã lưu kết quả tìm kiếm vào CSV; trang Streamlit đọc các kết quả đó thay vì chạy lại grid nặng mỗi lần mở web. Ridge/ElasticNet dùng GridSearchCV; Random Forest/LightGBM dùng RandomizedSearchCV. Kết quả từng fold và `best_params` trong JSON cho phép kiểm tra lại lựa chọn.

## 3. Fit cấu hình cuối và lưu `.pkl`

Sau khi chọn cấu hình bằng CV, tạo estimator mới rồi fit pipeline trên toàn development. Bước này **không phải** fit lên holdout. Artifact lưu cả bộ tiền xử lý đã fit và model.

```python
import joblib
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from src.preprocessing import make_linear_preprocessor

final_model = TransformedTargetRegressor(
    regressor=Pipeline([
        ("preprocess", make_linear_preprocessor()),
        ("estimator", Ridge(alpha=best_alpha)),
    ]),
    func=np.log1p,
    inverse_func=np.expm1,
    check_inverse=False,
)
final_model.fit(X_dev, y_dev)
joblib.dump(final_model, "models/learnt_ridge.pkl", compress=3)
```

Trang tương tác tạo artifact riêng `models/learnt_ridge.pkl`, `learnt_random_forest.pkl`, `learnt_elasticnet.pkl` hoặc `learnt_lightgbm.pkl` theo model đang chọn. Thư mục `models/` có README; file nhị phân `.pkl` được Git bỏ qua và có thể tái tạo bằng nút **Fit và lưu model**. Khi trao đổi artifact, cần dùng phiên bản Python và thư viện tương thích với phiên bản đã lưu.

## 4. Nạp model để dự đoán

Model đã lưu nhận dữ liệu có cùng schema predictor gốc như `X` từ `read_dataset()`. Pipeline tự áp dụng đúng các phép biến đổi học được trong lúc fit.

```python
loaded_model = joblib.load("models/learnt_ridge.pkl")
prediction_usd = loaded_model.predict(X_new)[0]
```

Trang minh họa nạp `.pkl` và dự đoán một dòng thuộc holdout; giá thật không được hiện trong bước inference. Form demo người dùng nhập là ứng dụng khác, dùng Ridge 5 feature đã đóng gói JSON để chạy ngay trong trình duyệt/Streamlit. Không so sánh trực tiếp demo 5 feature với bốn pipeline nghiên cứu 86 feature.

## 5. So sánh và đánh giá

- **MAE**: trung bình độ chênh tuyệt đối giữa giá dự đoán và giá thật, dễ đọc theo USD.
- **RMSE**: phạt lỗi lớn mạnh hơn MAE.
- **R²**: mức biến thiên của target được giải thích so với baseline dự đoán trung bình; có thể âm nếu dự đoán kém baseline.
- **MAPE**: sai số tuyệt đối tương đối theo phần trăm; cần cẩn trọng khi target gần 0.
- **CV MAE ± độ lệch chuẩn**: hiệu năng và độ dao động giữa năm fold development. Độ lệch chuẩn không phải khoảng tin cậy 95%.
- **Holdout**: 586 giao dịch được giữ riêng và chỉ dùng đánh giá cuối theo protocol đã định. Không chọn siêu tham số theo holdout.

Kết quả áp dụng cho Ames, Iowa và các giao dịch lịch sử của bộ dữ liệu. Chúng không đảm bảo hiệu năng ở Việt Nam hoặc thị trường hiện tại. Một số cột như năm/điều kiện bán chỉ biết tại thời điểm giao dịch; do đó nghiên cứu 86 feature không tự động chứng minh được khả năng định giá trước khi bán.

## 6. Giải thích mô hình và phản biện

Trang đánh giá cho xem hệ số Ridge/ElasticNet hoặc feature importance của cây. Hệ số phụ thuộc vào mã hóa và scale; importance theo impurity/gain mô tả cách model đã dùng feature, không chứng minh feature gây ra giá. Có thể đối chiếu các tín hiệu như chất lượng tổng thể, diện tích sinh hoạt, khu vực với EDA, nhưng cần trình bày đó là liên hệ dự báo trong mẫu.

Khi báo cáo, nêu cả điểm tốt và giới hạn: quy trình CV giảm rò rỉ; holdout giúp đánh giá độc lập; nhưng dữ liệu chỉ gồm một thành phố và thời kỳ lịch sử, một lần chia holdout còn có dao động, và mô hình chưa tạo khoảng dự báo 95%.
