import cv2
import numpy as np
from rembg import remove
from PIL import Image

print("====================================================")
print(" 🚀 MAIN4.PY: THUẦN HSV - CHỈ QUÉT LÓA SÁNG (VIỀN MỎNG) 🚀")
print("====================================================")

path_anh_goc = "binhco1.jpg"

# 1. ĐỌC ẢNH VÀ CHẠY LETTERBOX 800x800 CHỐNG MÉO
img_goc = cv2.imread(path_anh_goc)
if img_goc is None:
    print(f"LỖI: Không tìm thấy file ảnh '{path_anh_goc}' tại thư mục D:\\Xulyanh")
    exit()

h, w = img_goc.shape[:2]
scale = 800 / max(h, w)
new_w, new_h = int(w * scale), int(h * scale)
img_resize = cv2.resize(img_goc, (new_w, new_h))

# Tạo khung đệm đen chuẩn 800x800
img_padded = np.zeros((800, 800, 3), dtype=np.uint8)
x_offset = (800 - new_w) // 2
y_offset = (800 - new_h) // 2
img_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resize

# 2. CHẠY REMBG ĐỂ XÓA PHÔNG NỀN (Cô lập vật thể)
print("[REMBG] Đang cô lập chiếc bình, loại bỏ phông nền...")
img_pil = Image.fromarray(cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB))
img_rembg_rgba = np.array(remove(img_pil))
img_clean = cv2.cvtColor(img_rembg_rgba, cv2.COLOR_RGBA2BGR)
mask_vat_the = img_rembg_rgba[:, :, 3] 

# 3. CHUYỂN SANG KHÔNG GIAN MÀU HSV ĐỂ QUÉT LÓA SÁNG
print("[OPENCV] Đang bóc tách đốm lóa bằng hệ màu HSV...")
hsv = cv2.cvtColor(img_clean, cv2.COLOR_BGR2HSV)

# Ngưỡng lọc màu HSV cho vùng cháy sáng gắt trên men gốm
lower_white = np.array([0, 0, 240])
upper_white = np.array([180, 40, 255])

# Tạo mặt nạ và ép chỉ lấy bên trong phạm vi chiếc bình
mask_loa = cv2.bitwise_and(cv2.inRange(hsv, lower_white, upper_white), mask_vat_the)

# 4. TÌM VÀ VẼ VIỀN VÙNG LÓA SÁNG
contours, _ = cv2.findContours(mask_loa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_output = img_clean.copy()

if len(contours) > 0:
    print(f"🎉 Phát hiện {len(contours)} vùng nghi vấn lóa sáng!")
    # 🟢 VIỀN XANH LỤC MỎNG (độ dày = 1) ôm khít đốm lóa
    cv2.drawContours(img_output, contours, -1, (0, 255, 0), 1)
else:
    print("⚠️ Không tìm thấy đốm lóa nào đạt ngưỡng sáng HSV cài đặt.")

# 5. LƯU THÀNH PHẨM NGHIỆM THU
output_path = "ket_qua_main4_hsv.jpg"
cv2.imwrite(output_path, img_output)

print("====================================================")
print(f"✅ CHẠY XONG! Mở file '{output_path}' để xem kết quả.")
print("====================================================")