## 🛠️ Hướng Dẫn Cài Đặt & Vận Hành


Để đơn giản hóa quá trình cấu hình, hệ thống được thiết kế để cài đặt trực tiếp các gói thư viện vào Python hệ thống mà không cần khởi tạo môi trường ảo phức tạp.

### 1. Cài đặt các thư viện thiết yếu
Mở Terminal (VS Code) hoặc Command Prompt (CMD) trên máy tính của bạn và chạy dòng lệnh sau để tự động tải toàn bộ môi trường AI:

```bash
pip install opencv-python numpy pillow rembg torch torchvision transformers


### 📂 2. Cấu hình thư mục dự án
Đảm bảo các file ảnh cần quét lỗi và file mã nguồn được đặt chung trong một thư mục để OpenCV nhận diện chính xác đường dẫn tuyệt đối:

```text
📂 Xulyanh/
├── main2.py             # File mã nguồn thuật toán chính
└── caibinh2.png         # Ảnh vật thể đầu vào cần quét khuyết tật

### 🚀 3. Khởi chạy hệ thống thực chiến
Tại Terminal của VS Code hoặc CMD, bạn thực hiện gõ tuần tự 2 dòng lệnh sau:

*   **Bước 1:** Di chuyển dòng lệnh Terminal vào đúng thư mục chứa code:
    ```bash
    cd D:\Xulyanh
    ```
*   **Bước 2:** Kích hoạt chạy file pipeline xử lý:
    ```bash
    python main2.py
    ```

---

> ### 💡 LƯU Ý QUAN TRỌNG CHO LẦN CHẠY ĐẦU TIÊN (FIRST RUN)
>
> *   **Treo máy tạm thời:** Ở lần đầu tiên bạn kích hoạt lệnh chạy code, hệ thống sẽ âm thầm kết nối Internet để tự động tải khoảng **650MB** dữ liệu trọng số mô hình thật (`CLIP OpenAI` và `ResNet-18`) trực tiếp từ máy chủ về ổ C.  
> *   **Hiện tượng:** Terminal sẽ tạm thời **"khựng lại" (treo nhẹ)** ở hai dòng log nạp mô hình trong khoảng 1 - 3 phút tùy thuộc vào tốc độ mạng của bạn.
> *   **Hành động:** Bạn **VUI LÒNG KHÔNG TẮT TERMINAL**. Từ lần chạy thứ 2 trở đi, code sẽ tự động nạp trực tiếp từ bộ nhớ cache ẩn trên ổ cứng siêu tốc chỉ trong vòng 3 giây và hoàn toàn có thể chạy offline không cần Internet[cite: 2]!