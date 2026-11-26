# 04 · Demo nhập liệu và biến mô hình thành ứng dụng

## Vì sao demo dùng Ridge 5 feature?

Bảng nghiên cứu so sánh bốn pipeline 86 feature; việc yêu cầu người dùng nhập đầy đủ hàng chục thuộc tính sẽ khó dùng. Vì vậy form demo dùng một thí nghiệm Ridge riêng với năm biến numeric có thể khai báo trực tiếp: `gr_liv_area`, `overall_qual`, `bedroom_abvgr`, `full_bath`, `year_built`. Metric form phải đọc từ `docs/model/ridge_model.json`, không lấy từ Ridge 86 feature.

## Phép tính phía ứng dụng

```python
z = intercept
for feature, coefficient in zip(features, coefficients):
    z += coefficient * (value - feature["mean"]) / feature["scale"]
price_usd = max(0, expm1(z))
```

Đây là phép tính giống mô hình Ridge đã fit: chuẩn hóa từng giá trị bằng thống kê development, cộng hệ số trên target `log1p`, rồi đổi ngược về USD. JSON giữ hệ số, mean, scale, alpha và metric; không cần train lại khi người dùng bấm nút.

## Dùng ứng dụng

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app\streamlit_app.py
```

1. Nhập năm thuộc tính và bấm **Ước lượng giá lịch sử**.
2. Xem dự đoán điểm, MAE holdout của model demo, đóng góp toán học và cảnh báo ngoại suy nếu đầu vào nằm ngoài vùng phát triển thường gặp.
3. Thử thay đổi từng biến để quan sát mô hình phản ứng ra sao. Đây là tương tác với model, không chứng minh quan hệ nhân quả.

Không có khoảng tin cậy 95% hoặc khuyến nghị mua/bán. Đây là dữ liệu lịch sử Ames, Iowa, không phải thẩm định giá hiện tại hay thị trường Việt Nam.

## Localhost và GitHub Pages

Localhost phục vụ app Python trên máy đang chạy Streamlit. GitHub Pages chỉ phục vụ tài nguyên tĩnh nên không host trực tiếp server Streamlit. Muốn người khác dùng form, deploy app lên dịch vụ chạy Streamlit/Python rồi liên kết từ trang GitHub Pages sang địa chỉ app đó.
