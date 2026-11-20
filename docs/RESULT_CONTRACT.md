# Contract bàn giao thuật toán

Mỗi thành viên tự viết code, tune, đánh giá và diễn giải **2 thuật toán**. Các script huấn luyện thuộc trách nhiệm từng thành viên và phải được commit vào `members/member_a/` hoặc `members/member_b/` khi hoàn thành; hiện tại **không có mã thuật toán giả, không có điểm số tạo sẵn**.

## Phân công đề xuất

- Thành viên A: `Ridge` và `RandomForest`.
- Thành viên B: `ElasticNet` và `LightGBM`.
- Người tích hợp (bạn): thống nhất dữ liệu, protocol, EDA, data cleaning rationale, report sections chung, mã tích hợp và giao diện; **không nhận công huấn luyện/đánh giá của hai thành viên**.

## Cách dùng dữ liệu

```python
from src.protocol import load_split, folds, scores
X_train, X_test, y_train, y_test, train_ids, test_ids = load_split(pre_sale=True)
cv = folds()
```

- Split cố định: 2.344 train/dev + 586 test, random state 42.
- **Không dùng** `SalePrice`, `PID`, `Order`, `YrSold`, `MoSold`, `SaleType`, `SaleCondition` làm đặc trưng ở bài toán định giá trước bán.
- Encoder/imputer/scaler phải `fit` **trong pipeline của từng training fold**; không fit trước split. Hyperparameter chọn bằng CV trên train/dev.
- Mỗi người tự chạy nested CV/đánh giá bổ sung khi phù hợp. Test chỉ dùng sau khi chốt cấu hình; holdout này từng được quan sát trong nghiên cứu tham khảo, cần ghi hạn chế.
- Từng thành viên giải thích thuật toán, tham số, alternative, lỗi, SHAP/permutation importance (nếu làm), và tự viết subsection trong báo cáo.

## Schema kết quả

Mỗi thuật toán nộp **một** JSON vào `results/member_a/` hoặc `results/member_b/`:

```json
{
  "owner": "Thành viên A",
  "algorithm": "Ridge",
  "pre_sale_features": true,
  "seed": 42,
  "n_train": 2344,
  "n_test": 586,
  "cv_mae_mean_usd": 0,
  "cv_mae_std_usd": 0,
  "test_metrics": {"mae_usd": 0, "rmse_usd": 0, "r2": 0, "mape_pct": 0},
  "notes": "Thay số 0 bằng kết quả THẬT; không commit file mẫu dưới dạng kết quả",
  "source_script": "members/member_a/train_ridge.py"
}
```

**JSON này chỉ là ví dụ schema, không phải kết quả.** Kèm code, bảng CV/tuning, biểu đồ, và văn bản giải thích cho report. Dùng `python scripts/collect_results.py` để kiểm tra schema và tạo bảng tổng hợp.

## Hợp nhất model/UI

Một thành viên muốn cung cấp model cho UI cần xuất bundle tương thích (scikit-learn joblib) tại `artifacts/deployment.joblib` sau khi team chốt lựa chọn. UI chỉ hiển thị dự đoán nếu bundle thật đã tồn tại; không mượn bản ghi giả và không dùng model tham khảo của dự án cá nhân trước đây.
