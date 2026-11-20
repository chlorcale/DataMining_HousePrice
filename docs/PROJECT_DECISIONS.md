# Lý do cho các quyết định chung — người tích hợp phụ trách

| Bước | Tại sao chọn | Phản biện và giới hạn |
|---|---|---|
| Ames 2.930 mẫu | Đề tài 06 cho phép; đủ các loại biến và lỗi dữ liệu để nghiên cứu KDD | Dữ liệu Iowa 2006–2010, không thể suy rộng sang giá TP.HCM hiện nay |
| Không dùng PID/Order | Là mã nhận dạng, có nguy cơ học shortcut | Có thể hữu ích truy vết, nhưng không là thông tin nhà ở lúc định giá |
| Không dùng SalePrice trong X | Đây là biến mục tiêu | Mọi biến được tạo từ SalePrice trước split là leakage |
| Không dùng biến giao dịch hậu nghiệm | Pre-sale valuation không thể biết trước SaleCondition/SaleType/YrSold | Có thể dùng trong nghiên cứu hồi cứu nhưng phải tách bài toán |
| Split cố định 80/20 | Dễ so sánh cùng số mẫu | Không đủ kiểm chứng vượt thời gian/khu vực; holdout đã từng được xem trong project riêng trước đây |
| 5-fold CV trên train | So sánh ổn định hơn một lần chia | Tối ưu thông số qua CV có selection bias; test cần giữ riêng |
| Median/OneHot/Scaling | Đề xuất cho Linear baselines; outlier-robust, no ordinal giả định | Phải thử và chứng minh qua CV; không ép cách này cho mọi thuật toán |
| EDA chỉ trên dev | Tránh dùng phân phối target test ra quyết định trước đánh giá | Trực quan EDA không chứng minh quan hệ nhân quả |
| Preprocessing per fold | Ngăn imputation/scaling leakage | Mỗi thành viên phải tự bảo đảm, protocol không thể ép nếu code làm sai |
| MAE/RMSE/R²/MAPE | Sai số tuyệt đối, phạt đuôi, mức giải thích phương sai và sai số tương đối | R² không phải accuracy; MAPE nhạy với giá nhỏ; chọn model theo CV trước khi mở test |
| Tách nhiệm vụ theo thuật toán | Mỗi người có đầu ra/code độc lập dễ phản biện | Cần cùng split/đơn vị để so sánh công bằng; tránh trùng file |
| UI sau khi có model | Tránh hiển thị dự báo/model cũ như sản phẩm team | Chỉ demo khi có artifact do thành viên train và kiểm thử |

Quy định AI CO3029: nhóm phải tự làm và kiểm chứng phần lớn nội dung, kê khai công cụ/cách dùng. Đây là bộ khung phối hợp và phần nội dung khởi tạo, không phải báo cáo nộp tự động.
