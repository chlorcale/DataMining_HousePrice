# Ames Housing / CO3029 — Repo nhóm (khung tích hợp)

**Trạng thái: INITIALIZED, CHƯA CÓ KẾT QUẢ THUẬT TOÁN CỦA NHÓM.** Các thành viên **tự triển khai, huấn luyện, đánh giá và viết báo cáo**.

## Phân công

| Vai trò | Phạm vi | Nơi nộp |
|---|---|---|
| Người tích hợp | Dataset, quality check, EDA, shared protocol, code tổng hợp, báo cáo phần chung, khung UI | `data/`, `src/`, `scripts/`, `report/` |
| Thành viên A | Ridge + Random Forest: preprocessing phù hợp, tuning, CV, đánh giá, báo cáo | `members/member_a/`, `results/member_a/` |
| Thành viên B | ElasticNet + LightGBM: preprocessing phù hợp, tuning, CV, đánh giá, báo cáo | `members/member_b/`, `results/member_b/` |

Tên thuật toán có thể đổi theo quyết định nhóm nhưng vẫn phải có tối thiểu 3, thống nhất trước khi thực hiện.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/check_dataset.py
python scripts/eda.py
python -m pytest -q
python scripts/collect_results.py
streamlit run app/streamlit_app.py
```

Không phải tải dataset. `data/ames.csv` gồm 2.930 mẫu đầy đủ. Bảng EDA và kiểm toán ngoại biên sinh tại `outputs/eda/` (không commit mặc định). Ba hình chính sinh đồng thời tại `report/figures/` và được commit để báo cáo biên dịch ngay sau clone.

## Đánh giá chung

`src/protocol.py` quy định cùng cách chia dữ liệu, các metric và kiểm tra schema. Hai thành viên **không chia sẻ nhãn test để tune**; cần đưa toàn bộ scaler/encoder/imputer vào pipeline của từng fold. Xem `docs/RESULT_CONTRACT.md`.

`python scripts/collect_results.py` chỉ tổng hợp file JSON thực sự được giao. Thiếu kết quả thì in PENDING, không chèn chỉ số giả.

## Báo cáo và UI

- `report/main.tex`: **chỉ có phần chung người tích hợp phụ trách**; phần hai thành viên để `\\input` trỏ tới hai placeholder riêng.
- `report/member_a.tex`, `report/member_b.tex`: hai thành viên thay bằng văn bản và số liệu do chính mình làm.
- `app/streamlit_app.py`: giao diện khởi tạo/hiển thị EDA và kết quả **nếu có**. Dự đoán chỉ bật sau khi team đưa model phù hợp vào `artifacts/`.
- Kết quả nghiên cứu cá nhân và mô hình `.joblib` cũ **không có trong repo này**.

Compile report (TeX Live + XeLaTeX):
```powershell
cd report
xelatex -interaction=nonstopmode -halt-on-error main.tex
xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## Trước khi nộp

Cập nhật tên/MSSV, xác nhận tác giả từng phần, nguồn trích dẫn, khai báo AI trung thực; chạy UI và tự phản biện từng mô hình. Đề cương CO3029 không cho phép nộp sản phẩm AI tạo ra toàn bộ hoặc phần lớn.
