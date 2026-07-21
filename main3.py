import cv2
import numpy as np
from rembg import remove
from PIL import Image

print("====================================================")
print(" 🚀 MAIN4.PY: TRIỆT TIÊU LEM VIỀN & VẾT NỨT (MẢNH DÀI) 🚀")
print("====================================================")

path_anh_goc = "ngu1.png"

# 1. ĐỌC ẢNH GỐC
img_goc = cv2.imread(path_anh_goc)
if img_goc is None:
    print(f"LỖI: Không tìm thấy file ảnh '{path_anh_goc}'")
    exit()

h_goc, w_goc = img_goc.shape[:2]

# 2. CHẠY REMBG ĐỂ CÔ LẬP VẬT THỂ
print("[REMBG] Đang cô lập chiếc bình từ ảnh gốc...")
img_pil = Image.fromarray(cv2.cvtColor(img_goc, cv2.COLOR_BGR2RGB))
img_rembg_rgba = np.array(remove(img_pil))

img_clean_goc = cv2.cvtColor(img_rembg_rgba, cv2.COLOR_RGBA2BGR)
mask_vat_the_goc = img_rembg_rgba[:, :, 3]

# 3. TẠO KHUNG ĐỆM VUÔNG CÂN ĐỐI TỪ TÂM BÌNH
M = cv2.moments(mask_vat_the_goc)
if M["m00"] != 0:
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
else:
    cX, cY = w_goc // 2, h_goc // 2

max_dist_x = max(cX, w_goc - cX)
max_dist_y = max(cY, h_goc - cY)
kich_thuoc_khung = max(max_dist_x, max_dist_y) * 2

img_clean = np.zeros((kich_thuoc_khung, kich_thuoc_khung, 3), dtype=np.uint8)
mask_vat_the = np.zeros((kich_thuoc_khung, kich_thuoc_khung), dtype=np.uint8)

tam_khung = kich_thuoc_khung // 2
x_offset = tam_khung - cX
y_offset = tam_khung - cY

img_clean[y_offset:y_offset+h_goc, x_offset:x_offset+w_goc] = img_clean_goc
mask_vat_the[y_offset:y_offset+h_goc, x_offset:x_offset+w_goc] = mask_vat_the_goc

# Co thụt mặt nạ vào trong 7 pixel để giảm bớt gánh nặng nhiễu rìa
kernel = np.ones((7, 7), np.uint8)
mask_trong_long_binh = cv2.erode(mask_vat_the, kernel, iterations=1)

# 4. CHUYỂN SANG HỆ LAB VÀ QUÉT VÙNG TỐI
lab = cv2.cvtColor(img_clean, cv2.COLOR_BGR2LAB)
L_channel, _, _ = cv2.split(lab)

lower_dark = 10
upper_dark = 115 
mask_bong_toi = cv2.bitwise_and(cv2.inRange(L_channel, lower_dark, upper_dark), mask_trong_long_binh)

# 5. TÌM CONTOURS VÀ DÙNG BỘ LỌC HÌNH HỌC ĐỂ SĂN BÓNG ĐỔ KHỐI
contours, _ = cv2.findContours(mask_bong_toi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_bong_do_xin = []

for c in contours:
    dien_tich = cv2.contourArea(c)
    chu_vi = cv2.arcLength(c, True)
    
    # Lọc bỏ nhiễu hạt quá bé
    if dien_tich < 40:
        continue
        
    # 🎯 CHIÊU THỨC CỐT LÕI: Tính độ gầy/mảnh của mảng tối (Thinness Ratio)
    # Công thức: Tỷ lệ = (Chu vi^2) / Diện tích
    # Một hình tròn hoàn hảo có tỷ lệ này là ~12.5. Hình càng mảnh dài thì chỉ số này càng tăng vọt lên hàng trăm, hàng ngàn.
    if dien_tich > 0:
        ty_le_manh_dai = (chu_vi ** 2) / dien_tich
    else:
        continue
        
    # Lọc theo hình dáng hộp bao quanh (Bounding Box) để đo độ dẹt
    x, y, w_box, h_box = cv2.boundingRect(c)
    do_det = max(w_box, h_box) / min(w_box, h_box) if min(w_box, h_box) > 0 else 1

    # --- LOGIC LOẠI TRỪ VÙNG MẢNH DÀI (NỨT + LEM VIỀN) ---
    # Nếu tỷ lệ mảnh dài > 80 (rất gầy) HOẶC hộp bao quanh bị dẹt quá mức (độ dẹt > 6.0)
    # Nghĩa là nó là sợi chỉ, rãnh nứt hoặc dải lem mép ảnh -> BỎ QUA LUÔN!
    if ty_le_manh_dai > 80 or do_det > 6.0:
        continue
        
    # Vùng tối đạt chuẩn: To, béo, khối tròn trịa mịn màng => ĐÍCH THỊ LÀ BÓNG ĐỔ
    contours_bong_do_xin.append(c)

# 6. V VẼ THÀNH PHẨM VÀ XUẤT FILE
img_output = img_clean.copy()

if len(contours_bong_do_xin) > 0:
    print(f"🎉 Xuất sắc! Đã lọc sạch nhiễu. Phát hiện {len(contours_bong_do_xin)} mảng bóng đổ chuẩn khối!")
    # 🔵 VIỀN XANH DƯƠNG MỎNG (độ dày = 1) chỉ vẽ lên vùng bóng đổ chuẩn
    cv2.drawContours(img_output, contours_bong_do_xin, -1, (255, 0, 0), 1)
else:
    print("⚠️ Không tìm thấy mảng bóng đổ khối nào đạt yêu cầu.")

output_path = "ket_qua_bong_do_lab.jpg"
cv2.imwrite(output_path, img_output)

print("====================================================")
print(f"✅ HOÀN THÀNH! Hãy kiểm tra kết quả sạch nhiễu tại: '{output_path}'")
print("====================================================")