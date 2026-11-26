# Hướng dẫn chạy ứng dụng tương tác CO3029

Ứng dụng Streamlit chia thành các trang điều hướng: tổng quan, dữ liệu/EDA, tiền xử lý/Python, thuật toán/đánh giá, quy trình huấn luyện/lưu model và demo nhập liệu. Nội dung học thuật được tách thành Markdown dưới `docs/tutorials/`; mỗi bài có code Python và giải thích.

## Chạy trên máy cá nhân

Mở terminal ở thư mục `DataMining_HousePrice`, cài dependencies và chạy:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app\streamlit_app.py
```

Streamlit sẽ in địa chỉ local (thường là `http://localhost:8501`). Dữ liệu đầu vào là `data/ames.csv`; các file kết quả đã chạy được đọc từ `results/member_a/` và `results/member_b/`.

## Những gì ứng dụng làm

- **Tổng quan:** phạm vi nghiên cứu, thống kê chính và kết quả CV.
- **Dữ liệu & EDA:** biểu đồ giá, missingness, area/price, quality, neighborhood và bảng record mẫu development.
- **Tiền xử lý & Python:** FeatureBuilder, ba preprocessing paths, giải thích leakage và xem source code thật.
- **Thuật toán & đánh giá:** so sánh metric thực nghiệm, fold variance, hệ số/feature importance và residual plots có sẵn.
- **Huấn luyện & lưu model:** lấy mẫu theo phân vị giá (hồi quy, không tạo lớp mất cân bằng giả), xem CV/search CSV, fit cấu hình đã chốt trên development, lưu `.pkl`, nạp lại và dự đoán holdout mà không hiển thị giá thật.
- **Demo nhập liệu:** nhập trực tiếp năm trường. Ridge 5 đặc trưng đã fit được đọc từ JSON và trả kết quả ngay, không cần fit lại.

Tab đánh giá hiển thị kết quả của bốn pipeline 86 feature. Demo nhập liệu dùng model Ridge riêng, giới hạn ở năm feature pre-sale để người dùng có thể khai báo đủ đầu vào. Hai mục đích này được phân biệt rõ; metric của model 5 feature không được gộp vào bảng so sánh 86 feature.

## Giới hạn cần trình bày khi demo

- Holdout là giao dịch lịch sử tại Ames, Iowa; giá bằng USD.
- Demo dùng năm biến; vị trí và phần lớn chi tiết tài sản không nằm trong form. Đây không phải biểu mẫu định giá đầy đủ.
- Dữ liệu có trường thời điểm/điều kiện bán có thể chỉ biết sau giao dịch. Kết quả hiện tại không chứng minh khả năng định giá trước khi bán.
- Ứng dụng đưa ra dự đoán điểm, chưa xây dựng khoảng tin cậy 95%.
- Hệ số và feature importance giải thích mô hình dự báo, không chứng minh quan hệ nhân quả.

## Triển khai

GitHub Pages chỉ phục vụ file tĩnh HTML/CSS/JavaScript; nó không chạy Python hoặc Streamlit. Để người khác truy cập ứng dụng này, cần một môi trường chạy Streamlit như Streamlit Community Cloud hoặc một server Python. Có thể tiếp tục dùng GitHub Pages cho trang giới thiệu/báo cáo và liên kết từ đó sang ứng dụng Streamlit được host riêng.
