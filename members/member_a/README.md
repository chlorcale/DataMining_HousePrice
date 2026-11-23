# Member A — Ridge Regression & Random Forest

Phần này thực hiện đúng hợp đồng trong `docs/TEAM_MODEL_CONTRACT.md`; không đổi
dataset, split, feature engineering hay các file dữ liệu chung.

Nếu bạn mới học Data Mining, bắt đầu bằng
[`docs/LEARNING_GUIDE_MEMBER_A.md`](../../docs/LEARNING_GUIDE_MEMBER_A.md): tài
liệu giải thích KDD/CRISP-DM, leakage, CV, Ridge, Random Forest, metric và cách
static page chạy model trong browser.

## Chạy

Từ thư mục gốc repo:

```bash
python members/member_a/train_ridge.py
python members/member_a/train_random_forest.py
python members/member_a/train_static_ridge.py
python scripts/export_web_results.py
```

## Quy trình

1. Đọc raw Ames bằng `src.data.read_dataset()` (2,930 dòng, 79 predictors), rồi
   dùng `src.protocol.holdout_indices()` để lấy dev 2,344 dòng và holdout 586 dòng.
2. Truyền raw predictors vào factory tương ứng. Factory có `FeatureBuilder` nên
   tạo 86 cột; không truyền các CSV đã feature-engineer với mặc định
   `engineered=True` để tránh chạy FeatureBuilder lần hai.
3. Đặt factory chưa fit bên trong `Pipeline`. `GridSearchCV`/`RandomizedSearchCV`
   dùng `src.protocol.make_cv()` nên imputer, encoder, scaler được fit lại riêng
   trong từng training fold. Không dùng `data/fitted_previews/` khi CV.
4. Chọn cấu hình theo MAE CV trên dev. Chỉ sau khi khóa cấu hình mới fit lại toàn
   dev và chấm holdout; các metric được báo cáo trên USD gốc.

## Hai mô hình

- **Ridge:** numeric median imputation + missing indicators + StandardScaler;
  categorical `Unknown` fill + OneHotEncoder. Grid search `alpha` trên hai giả
  thuyết target `log1p(SalePrice)` và SalePrice identity. Chọn log nếu identity
  không giảm CV MAE hơn 1%; lưu top hệ số để diễn giải.
- **Random Forest:** numeric median imputation + missing indicators; categorical
  `Unknown` fill + OrdinalEncoder theo factory chung. Randomized search 30 cấu
  hình với 5-fold CV trên log1p target, sau đó so sánh log1p và identity tại cấu
  hình tốt nhất bằng CV MAE dev.
  Lưu top impurity feature importance; diễn giải lưu ý ordinal encoding áp đặt
  thứ tự giả cho biến danh mục.

**Lưu ý về lựa chọn target của Random Forest:** search space 30 cấu hình chọn
tree structure theo CV MAE với log1p target; sau đó so log và identity tại đúng
cấu hình đó bằng CV. Đây là so sánh target transform có kiểm soát, không phải
tối ưu lại mọi tham số riêng cho identity. Nếu muốn khẳng định identity/log tốt
hơn toàn cục, cần thêm search identity như một ablation dev-only.

- **Static web Ridge:** một thí nghiệm thứ hai với 5 numeric feature biết trước
  giao dịch. Script này không thay kết quả Ridge 86 feature dùng so sánh học
  thuật; nó xuất `docs/model/ridge_model.json` để form chạy inference trong
  browser mà không có backend. Export xác nhận công thức JS khớp sklearn trên
  holdout. Ridge static chọn alpha ở biên rất nhỏ của lưới (`1e-8`), nên dữ liệu
  5 biến hiện tại hầu như không cần regularization mạnh.

## Đầu ra

- `results/member_a/ridge_metrics.json`
- `results/member_a/random_forest_metrics.json`
- `results/member_a/ridge_cv_*.csv`, `random_forest_cv_results.csv`
- `results/member_a/{ridge,random_forest}_model.joblib` (artifact tạo khi chạy,
  không cần commit nếu nhóm chưa thống nhất)

Metrics JSON ghi cấu hình, search space, fold-level và mean ± std CV, holdout
MAE/RMSE/R²/MAPE, thời gian, phiên bản thư viện, và hạn chế. Script không tạo
ra số liệu cho đến khi được chạy trên máy; không chép kết quả từ Ultra/báo cáo cũ.

## Phạm vi nghiên cứu

Repo dùng Ames Housing, giao dịch lịch sử ở Ames, Iowa, đơn vị USD. Bài toán hiện
tại là phân tích giá giao dịch lịch sử, không phải định giá hiện hành ở Việt Nam.
Các trường `yr_sold`, `mo_sold`, `sale_type`, `sale_condition` có thể không biết
trước lúc bán; mô hình hiện tại không được quảng bá như pre-sale estimator. Nếu
thử loại trường hậu nghiệm, ghi thành ablation riêng và không sửa pipeline chung
mà chưa thống nhất với người tích hợp.

## Báo cáo

Điền `report/member_a.tex` sau khi chạy script và tự xem CV/holdout, biểu đồ sai
số, hệ số Ridge và importance của Random Forest. Giải thích quyết định, phản
biện ordinal encoding/feature importance và nêu rõ AI hỗ trợ ở phần nào theo quy
định môn học. Không coi khung báo cáo là kết quả đã xác nhận.
