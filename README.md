# 🛡️ Ứng Dụng Học Máy Phát Hiện Giao Dịch Gian Lận (Streamlit App)

Ứng dụng này được chuyển đổi tự động từ quy trình huấn luyện và đánh giá mô hình học máy trong Notebook phòng chống gian lận tài chính. Hệ thống hỗ trợ tải tệp dữ liệu lịch sử, phân tích trực quan hóa, tinh chỉnh cấu hình siêu tham số và triển khai dự đoán giao dịch rủi ro theo thời gian thực hoặc theo danh sách hàng loạt.

## 🤖 Mô Hình & Thuật Toán Tích Hợp
Ứng dụng tái hiện đầy đủ 3 thuật toán phân loại cốt lõi được sử dụng trong notebook kiểm định:
1. **Logistic Regression (Model 1)**: Mô hình tuyến tính cơ bản, tối ưu tốc độ.
2. **Decision Tree (Model 2)**: Cây quyết định trực quan hóa các biên phân tách lớp.
3. **Random Forest Classifier (Model 3)**: Mô hình học máy dạng Ensemble (Rừng ngẫu nhiên) mang lại độ chính xác vượt trội trên tập dữ liệu thử nghiệm của hệ thống.

## 📁 Cấu Trúc File Dữ Liệu Đầu Vào (Schema)
Để ứng dụng hoạt động chính xác và không bị lỗi hệ thống, tệp dữ liệu tải lên (hỗ trợ cả định dạng `.csv`, `.xlsx`, `.xls`) cần tuân thủ cấu trúc thuộc tính sau:
* **Các biến đầu vào (14 đặc trưng liên tục):** `X_1`, `X_2`, `X_3`, `X_4`, `X_5`, `X_6`, `X_7`, `X_8`, `X_9`, `X_10`, `X_11`, `X_12`, `X_13`, `X_14`.
* **Biến mục tiêu (Nhãn phân loại):** `default` nhận giá trị nhị phân (`0`: Giao dịch hợp lệ, `1`: Giao dịch gian lận/Rủi ro).

---

## 🛠️ Hướng Dẫn Cài Đặt & Khởi Chạy

### Bước 1: Chuẩn bị môi trường máy tính
Đảm bảo máy tính của bạn đã cài đặt phiên bản Python ổn định (Khuyến nghị bản từ `3.9` đến `3.12`).

### Bước 2: Cài đặt các thư viện cần thiết
Mở Terminal / Command Prompt tại thư mục chứa mã nguồn ứng dụng và chạy lệnh sau:
```bash
pip install -r requirements.txt
