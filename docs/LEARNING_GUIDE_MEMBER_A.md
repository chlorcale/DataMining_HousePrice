# Hướng dẫn học nhanh cho Member A: Data Mining và phần việc Ridge/RF

Tài liệu này giải thích thiết kế đang có trong repo. Hãy tự chạy, xem đầu ra và diễn giải bằng lời của bạn trước khi đưa vào báo cáo hoặc bảo vệ.

## 1. Bài toán đang giải quyết

Mỗi dòng dữ liệu mô tả một giao dịch nhà ở Ames, Iowa. Các cột mô tả căn nhà và giao dịch là predictors (X); `SalePrice` là target (y). Mục tiêu supervised learning là học hàm `f(X) ≈ y` từ giao dịch đã biết giá, rồi ước lượng giá cho mẫu chưa dùng khi fit.

Repo hiện có 2.930 giao dịch lịch sử, giá USD. Đây là hồi quy vì target là số liên tục. Đây không phải dataset Vietnam Real Estates hay mô hình giá hiện tại ở Việt Nam.

## 2. Data Mining nằm ở đâu trong quy trình?

KDD thường được mô tả qua các ý: chọn dữ liệu, tiền xử lý, biến đổi/biểu diễn, khai phá bằng thuật toán, rồi đánh giá và diễn giải tri thức. CRISP-DM gom công việc thành business understanding, data understanding, data preparation, modeling, evaluation và deployment. Phần Member A chủ yếu làm **modeling + evaluation**, nhưng phải hiểu các bước trước/sau để không dùng sai dữ liệu:

1. **Business question:** từ thông tin giao dịch/nhà, có dự đoán được giá bán lịch sử Ames không?
2. **Data understanding:** xem ý nghĩa cột, đơn vị, thiếu dữ liệu, phân phối SalePrice, outlier và cột chỉ biết sau giao dịch.
3. **Preparation:** `read_dataset()` chuẩn hóa tên, tách X/y, bỏ mã định danh; `FeatureBuilder` tạo 7 biến dẫn xuất (79 thành 86 predictors). Imputer/encoder/scaler còn lại được học trong pipeline.
4. **Modeling:** so sánh Ridge và Random Forest với 5-fold CV, chọn tham số chỉ trên tập development.
5. **Evaluation:** chốt cấu hình rồi chấm holdout độc lập; xem nhiều metric và residuals.
6. **Deployment:** mẫu `/docs` hiện là static mockup. Để đưa ra dự báo thật cần đầu vào khớp schema và gọi đúng model đã fit.

## 3. Vì sao phải chia train/dev/test?

- **Development set (2.344 dòng):** được dùng để học tham số mô hình và so cấu hình.
- **5-fold CV trên development:** chia development thành 5 phần. Mỗi lượt, 4 phần fit, 1 phần validation đo khả năng tổng quát hóa; lặp đủ 5 lượt. Trung bình/độ lệch giữa fold giúp thấy cấu hình có ổn định không.
- **Holdout (586 dòng):** cất riêng cho tới khi khóa lựa chọn. Nó mô phỏng dữ liệu mới. Nếu xem holdout rồi đổi tham số theo kết quả, nó đã trở thành một phần của quá trình tune và không còn đánh giá độc lập.

Seed 42 giữ split 80/20 lặp lại được; seed 123 giữ các fold CV lặp lại được. Đây là split ngẫu nhiên theo hàng, không phải đánh giá theo thời gian. Nếu mục tiêu là dự báo giao dịch tương lai thì cần thí nghiệm temporal split riêng.

## 4. Data leakage — ví dụ dễ hình dung

Giả sử ta điền giá trị thiếu và học trung bình/median trên toàn bộ 2.344 dòng rồi mới chia 5 fold. Khi đó những dòng validation đã góp phần quyết định median, scale hoặc danh sách category. Validation không còn hoàn toàn “mới”. Đây là leakage làm CV lạc quan hơn thực tế.

Vì vậy `make_linear_preprocessor()` hoặc `make_tree_preprocessor()` được để bên trong sklearn `Pipeline`. Khi CV gọi `fit` trên từng fold, cả `FeatureBuilder`, imputer, encoder, scaler đều được fit chỉ trên phần train của fold. Không dùng `data/fitted_previews/` đã fit trước để làm CV.

`read_dataset()` đưa raw 79 predictors vào pipeline. Factory mặc định chạy `FeatureBuilder(engineered=True)` và sinh 7 đặc trưng: quality × area, tuổi nhà khi bán, số năm từ remodel, tổng bathroom quy đổi, tổng porch, area + basement, garage presence. Do đó đừng đưa `data/processed/train_features.csv` vào factory mặc định, nếu không FeatureBuilder chạy lần hai. Contract có cách dùng CSV với `engineered=False`, nhưng raw source path là lựa chọn hiện tại.

## 5. Ridge Regression — tuyến tính có phạt L2

Hồi quy tuyến tính dự đoán tổ hợp có trọng số:

`ŷ = β₀ + β₁x₁ + ... + βₚxₚ`.

Ridge chọn hệ số bằng cách tối thiểu hóa sai số bình phương cộng penalty:

`Σ(yᵢ − ŷᵢ)² + α Σ βⱼ²`.

`α` lớn làm hệ số bị co về 0, giảm độ nhạy/overfit nhưng quá lớn gây underfit. Ridge thường không làm hệ số đúng bằng 0; nó giữ nhiều biến nhưng giảm trọng số.

Trong repo, số được median-impute, thêm missing indicators và StandardScaler; danh mục được điền `Unknown`, rồi one-hot encode. Scale quan trọng với Ridge vì penalty tác động lên hệ số: nếu một cột đo bằng hàng nghìn và cột khác là 0/1, thang đo khác nhau làm regularization không cân bằng.

Script thử hai giả thuyết:

- `identity`: Ridge học trực tiếp giá USD.
- `log1p`: Ridge học `log(1 + giá)`, dự đoán sau đó được inverse-transform về USD trước khi tính metric.

Giá có thể lệch phải: một số nhà giá rất cao kéo đuôi phân phối. Log thường nén đuôi lớn và đổi mục tiêu sang sai số tương đối hơn, nhưng không mặc nhiên tốt hơn. Script chọn theo dev CV MAE: identity phải tốt hơn hơn 1% mới thay log. Đây là chính sách nhóm chọn trước, không phải định luật thống kê.

## 6. Random Forest — nhiều cây quyết định

Một cây quyết định liên tục chia dữ liệu theo ngưỡng feature để nhóm các mẫu có target gần nhau. Cây sâu có thể nhớ dữ liệu train. Random Forest huấn luyện nhiều tree trên bootstrap samples và các tập con features, rồi lấy trung bình dự đoán. Sự đa dạng giữa cây giúp giảm variance so với một cây đơn lẻ và học quan hệ phi tuyến mà không cần tự đặt công thức tuyến tính.

Các tham số được khảo sát trong script:

- `n_estimators`: số cây; nhiều hơn thường ổn định hơn nhưng chạy lâu hơn.
- `max_depth`: giới hạn độ sâu; hạn chế cây ghi nhớ quá kỹ.
- `min_samples_split`, `min_samples_leaf`: cần tối thiểu bao nhiêu mẫu để tách/giữ lá; tăng lên thường làm mô hình mượt hơn.
- `max_features`: tỷ lệ feature mỗi split; tăng ngẫu nhiên giữa cây.
- `max_samples`: tỷ lệ mẫu bootstrap cho mỗi cây.

`make_tree_preprocessor()` điền số thiếu, tạo missing indicators và ordinal-encode danh mục. Lưu ý phản biện: mã category 0,1,2 tạo thứ tự giả; cây có thể tách `category_code <= 1.5` như thể category có trật tự. Đây là lựa chọn preprocessing nhóm đã khóa cho so sánh, nhưng phải nêu hạn chế. Random Forest không cần scale numeric như Ridge.

Search 30 cấu hình hiện chạy trên log1p target; sau đó so log/identity ở cấu hình tree đã chọn bằng CV dev. Điều này không tương đương việc chạy một search độc lập đầy đủ cho identity, nên báo cáo đúng phạm vi so sánh này.

## 7. Đọc các metric

- **MAE** = trung bình `|y − ŷ|`. Dễ diễn giải: sai trung bình bao nhiêu USD. Tối ưu MAE tương đối ít nhạy với lỗi cực lớn hơn RMSE.
- **RMSE** = căn bậc hai trung bình `(y − ŷ)²`. Cùng đơn vị USD; lỗi lớn bị phạt mạnh hơn. Nếu RMSE lớn hơn MAE nhiều, vài lỗi lớn có thể đang chi phối.
- **R²** = `1 − SSE/SST`. So với baseline luôn đoán trung bình: 1 là fit hoàn hảo, 0 ngang baseline đó, âm là tệ hơn baseline. Không buộc phải dương trên holdout.
- **MAPE** = trung bình `|y−ŷ|/y × 100%`. Dễ nói theo phần trăm nhưng có thể bất ổn khi y gần 0. Ames giá nhà không gần 0, code vẫn chặn mẫu số tối thiểu 1 USD.

Các metric được tính trên USD gốc, không phải log USD. CV mean ± standard deviation mô tả hiệu suất/độ biến động giữa 5 validation folds; holdout score là đánh giá cuối độc lập. Không chọn model bằng holdout.

## 8. Diễn giải và giới hạn

- Ridge top coefficients cần xem feature name, dấu, target transform và scaling. Nếu target log thì hệ số ở thang log-target; không nói “mỗi đơn vị làm tăng X USD” trực tiếp. Tương quan giữa predictors cũng làm hệ số khó diễn giải riêng.
- Random Forest impurity importance là mức giảm impurity tích lũy, không phải causal effect. Nó có thể thiên lệch về feature có nhiều điểm chia/category. Importance không có dấu tăng/giảm và không tự nó chứng minh feature gây ra giá.
- Holdout ngẫu nhiên cùng phân phối chỉ ước lượng generalization trong thiết kế này; không đảm bảo hiệu suất năm sau hoặc thành phố khác.
- Ames có `yr_sold`, `mo_sold`, `sale_type`, `sale_condition`. Một số cột chỉ biết khi/sau giao dịch. Bởi vậy đây là mô hình phân tích giá giao dịch lịch sử, chưa phải công cụ môi giới dùng trước lúc bán.
- Không có khoảng tin cậy 95% chỉ vì form hiển thị một khoảng. Muốn báo khoảng dự đoán phải xây và đánh giá phương pháp riêng, chẳng hạn conformal prediction trên dữ liệu tách phù hợp.

## 9. Static web hiện đã dự báo thế nào?

GitHub Pages chỉ host HTML/CSS/JavaScript, không chạy Python. Để không cần backend, repo có một thí nghiệm Ridge thứ hai (`train_static_ridge.py`) dùng 5 biến số numeric: diện tích sinh hoạt, chất lượng, số phòng ngủ, số phòng tắm đầy đủ và năm xây. Đây là các thuộc tính có thể biểu diễn trong form và có thể biết trước giao dịch. Script train trên cùng split/fold của nhóm, đánh giá holdout, rồi xuất `docs/model/ridge_model.json` gồm median, mean/scale, hệ số, intercept và target transform. JavaScript tính đúng công thức của Ridge đã fit; lúc export script kiểm tra công thức browser với sklearn predictions trên holdout.

Đây là model **riêng cho giao diện**, không thay thế Ridge 86 đặc trưng hoặc Random Forest trong bảng so sánh thành viên. Ít feature làm web gọn nhưng bỏ qua neighborhood và nhiều thông tin nên lỗi lớn hơn. Trang ghi rõ giá USD lịch sử ở Ames, không phải thị trường hiện tại hoặc Việt Nam, và không đưa khoảng tin cậy chưa được đánh giá.

Để cập nhật model và bảng so sánh sau khi thay đổi code/dữ liệu, chạy từ repo root:

```powershell
.\.venv\Scripts\python.exe members\member_a\train_static_ridge.py
.\.venv\Scripts\python.exe scripts\export_web_results.py
```

Browser formula với feature vector `x` là `expm1(intercept + Σ coefficientⱼ × ((xⱼ − meanⱼ) / scaleⱼ))`. Median được ghi vào artifact để mô tả preprocessing; form hiện bắt buộc nhập đủ năm biến, nên browser không cần điền missing.

**Khi cần mở rộng:** nếu muốn web dùng full 86-feature Ridge/RF, cần form đầy đủ schema Ames, serialization đầy đủ preprocessing/tree inference, rồi kiểm tra prediction parity. Nếu muốn backend, nạp `.joblib` qua FastAPI/Streamlit và để Pages gọi API; backend cần HTTPS/CORS, schema validation và deployment riêng.

## 10. Lộ trình học và tự làm

1. Mở `src/data.py`, `src/features.py`, `src/protocol.py`; tự giải thích raw X/y, 79→86 feature và ý nghĩa hai seed.
2. Vẽ trên giấy một vòng 5-fold: training partition, validation partition và lúc mỗi preprocessing step được fit.
3. Chạy Ridge trước; mở metrics JSON + CSV. Tự trả lời alpha/target nào thắng và MAE CV khác test ra sao.
4. Chạy Random Forest; xem cấu hình, độ lệch fold, importance. Chọn 3 feature và giải thích vì sao importance không phải tác động nhân quả.
5. Vẽ actual vs predicted và residual = prediction − actual; tìm vùng giá mô hình thường over/underpredict. Đây là việc cần bổ sung để báo cáo mạnh hơn.
6. Chạy `train_static_ridge.py`, xác nhận parity max absolute error gần 0, mở trang bằng HTTP server và thử các giá trị biên/hợp lệ.
7. Điền `report/member_a.tex` bằng số do chính lần chạy tạo, trích nguồn dữ liệu/thư viện, ghi limitation và khai báo minh bạch AI hỗ trợ.

Nếu bị hỏi “vì sao dùng Data Mining?”, trả lời ngắn: dữ liệu có nhiều thuộc tính và quan hệ phi tuyến/thiếu giá trị; ta dùng quy trình KDD để chuẩn hóa, tạo đặc trưng, học mô hình hồi quy, kiểm tra khả năng tổng quát hóa trên dữ liệu chưa dùng, rồi diễn giải giới hạn trước khi đưa thành ứng dụng.
