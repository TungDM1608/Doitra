import os
# 🔥 TẮT HOÀN TOÀN CÁC THÔNG BÁO RÁC ONEDNN VÀ WARNING CỦA TENSORFLOW/TÙY BIẾN HỆ THỐNG
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from rembg import remove
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel

# =====================================================================
# --- KHỞI TẠO BỘ TRÍCH XUẤT DỊ THƯỜNG PATCHCORE-LITE CỤC BỘ ----------
# =====================================================================

class PatchCoreLiteInspector:
    """ Bộ trích xuất dị thường kết cấu sử dụng xương sống ResNet-18 mạng ẩn sâu """
    def __init__(self, device):
        self.device = device
        # Nạp mạng pretrained thật từ ImageNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2]).to(device)
        self.feature_extractor.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def get_anomaly_score(self, patch_img):
        """Tính toán độ biến động cấu trúc hình học nội bộ ô ảnh khách"""
        with torch.no_grad():
            tensor = self.transform(patch_img).unsqueeze(0).to(self.device)
            features = self.feature_extractor(tensor)
            
            # TỐI ƯU TOÁN HỌC: Lấy độ lệch chuẩn của ma trận đặc trưng sâu
            score = torch.std(features).item()
            
            # Chuẩn hóa động bằng hàm Sigmoid thu nhỏ để dải điểm nhạy hơn với vết nứt mảnh
            adjusted_score = 1.0 / (1.0 + np.exp(-5.0 * (score - 0.25)))
            return float(np.clip(adjusted_score, 0.0, 1.0))

# --- KHỞI TẠO THIẾT BỊ VÀ PHÂN PHỐI MODEL DEEP LEARNING ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Hệ thống] Đang chạy pipeline dự phòng trên: {device.type.upper()}")

print("[AI Nạp] Đang nạp mô hình CLIP (ViT-B/32 của OpenAI) thật...")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

print("[AI Nạp] Kích hoạt bộ kiểm định kết cấu PatchCore-Lite...")
patchcore_inspector = PatchCoreLiteInspector(device)


def chay_pipeline_tu_khoa_khong_mau(input_path, output_path):
    print("\n====================================================")
    print("   KÍCH HOẠT PIPELINE TỰ DOÀN KHÔNG MẪU (BẢN TỐI ƯU) ")
    print("====================================================")

    # --------------------------------------------------------------
    # BƯỚC 1: TIỀN XỬ LÝ VĨ MÔ, TÁCH NỀN VÀ XỬ LÝ ÁNH SÁNG
    # --------------------------------------------------------------
    img_goc = cv2.imread(input_path)
    if img_goc is None:
        print(f"LỖI: Không tìm thấy file ảnh gốc '{input_path}'!")
        return

    # Letterbox đưa về dạng chuẩn hóa hình học 800x800 chống méo
    h, w = img_goc.shape[:2]
    scale = 800 / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resize = cv2.resize(img_goc, (new_w, new_h))
    
    img_padded = np.zeros((800, 800, 3), dtype=np.uint8)
    x_offset, y_offset = (800 - new_w) // 2, (800 - new_h) // 2
    img_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resize

    # Bóc tách nền cô lập vật thể bằng Rembg AI
    img_pil = Image.fromarray(cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB))
    img_rembg_rgba = np.array(remove(img_pil))
    
    img_coc_sach = cv2.cvtColor(img_rembg_rgba, cv2.COLOR_RGBA2BGR)
    mask_coc = img_rembg_rgba[:, :, 3]
    tong_pixel_thanh_coc = cv2.countNonZero(mask_coc)

    if tong_pixel_thanh_coc == 0:
        print("LỖI: AI không nhận diện được bất kỳ vật thể nào!")
        return

    # Triệt tiêu cháy sáng, lóa men bằng CLAHE kênh LAB
    lab = cv2.cvtColor(img_coc_sach, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab_gop = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    img_can_bang = cv2.cvtColor(lab_gop, cv2.COLOR_LAB2BGR)
    
    # Hiệu chỉnh Gamma làm phẳng bóng đổ bề mặt
    table = np.array([((i / 255.0) ** (1.0 / 1.2)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    img_coc_sach = cv2.LUT(img_can_bang, table)
    img_phu_quet_ngam = cv2.bitwise_and(img_coc_sach, img_coc_sach, mask=mask_coc)

    # --------------------------------------------------------------
    # BƯỚC 2 & 3: QUÉT THÔ NGƯỠNG ĐỘNG CỤC BỘ & CẮT CÓ NGỮ CẢNH RỘNG
    # --------------------------------------------------------------
    gray = cv2.cvtColor(img_phu_quet_ngam, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    nghi_van_map_raw = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 19, 9)
    
    # Dùng mặt nạ co nhẹ để loại trừ rìa biên biên dạng sản phẩm gây nhiễu
    mask_trong_coc = cv2.erode(mask_coc, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
    nghi_van_map_raw = cv2.bitwise_and(nghi_van_map_raw, mask_trong_coc)

    nghi_van_map = np.zeros_like(nghi_van_map_raw)
    vung_quet_contours, _ = cv2.findContours(nghi_van_map_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for vung in vung_quet_contours:
        if cv2.contourArea(vung) < 12: continue
        cv2.drawContours(nghi_van_map, [vung], -1, 255, -1)

    # --------------------------------------------------------------
    # BƯỚC 4: HỘI ĐỒNG AI PHÂN XỬ TỐI ƯU (PATCHCORE LITE + CLIP HOA VĂN)
    # --------------------------------------------------------------
    contours, _ = cv2.findContours(nghi_van_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    mask_loi_chinh_thuc = np.zeros((800, 800), dtype=np.uint8)
    img_hien_thi_ket_qua = img_coc_sach.copy()
    
    so_vung_nghi_van = 0
    so_vung_loi_ai_duyet = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 12: continue
            
        so_vung_nghi_van += 1
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Mở rộng hộp bao gấp 1.8 lần lấy ngữ cảnh men nền xung quanh vùng lỗi
        cx, cy = x + w//2, y + h//2
        size_hop = max(int(max(w, h) * 1.8), 50)
        
        x1, y1 = max(0, cx - size_hop), max(0, cy - size_hop)
        x2, y2 = min(800, cx + size_hop), min(800, cy + size_hop)
        
        patch_co_ngu_canh = img_coc_sach[y1:y2, x1:x2]
        if patch_co_ngu_canh.size == 0: continue
        patch_zoom = cv2.resize(patch_co_ngu_canh, (128, 128), interpolation=cv2.INTER_CUBIC)

        # 🧠 TRỌNG TÀI 1: PatchCore-Lite cục bộ đo đứt gãy hình học bề mặt
        score_patchcore = patchcore_inspector.get_anomaly_score(patch_zoom)

        # 🖼️💬 TRỌNG TÀI 2: CLIP OpenAI 
        prompts = [
            "perfect ceramic pottery surface with intact decorated art patterns", # Nhãn 0: Hoa văn lành
            "damaged ceramic surface with structural cracks, scratches, and chipped edges" # Nhãn 1: Lỗi thật
        ]
        patch_pil = Image.fromarray(cv2.cvtColor(patch_zoom, cv2.COLOR_BGR2RGB))
        inputs = clip_processor(text=prompts, images=patch_pil, return_tensors="pt", padding=True).to(device)
        
        with torch.no_grad():
            outputs = clip_model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            
        score_clip = float(probs[1]) 

        # TOÁN TỬ ENSEMBLE TRỌNG SỐ MỀM CÂN BẰNG (60% Kết cấu + 40% Ngữ nghĩa)
        tong_diem_ai_chot = (0.60 * score_patchcore) + (0.40 * score_clip)

        # Quyết định chốt hạ Đỏ/Vàng
        if tong_diem_ai_chot >= 0.62: # Tăng nhẹ ngưỡng lọc để giảm thiểu tối đa báo động giả hoa văn
            so_vung_loi_ai_duyet += 1
            cv2.drawContours(img_hien_thi_ket_qua, [cnt], -1, (0, 0, 255), 1) # 🔴 ĐỎ: Vết nứt thật
            cv2.drawContours(mask_loi_chinh_thuc, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(img_hien_thi_ket_qua, [cnt], -1, (0, 255, 255), 1) # 🟡 VÀNG: Hoa văn/Nhiễu

    # --------------------------------------------------------------
    # THỐNG KÊ TOÁN HỌC & XUẤT FILE BÁO CÁO NGHIỆM THU ĐỒ ÁN
    # --------------------------------------------------------------
    tong_pixel_loi_that = cv2.countNonZero(mask_loi_chinh_thuc)
    ty_le_hu_hai = (tong_pixel_loi_that / tong_pixel_thanh_coc) * 100

    print("\n====================================================")
    print("        BÁO CÁO NGHIỆM THU THỊ GIÁC MÁY KHÔNG MẪU    ")
    print("====================================================")
    print(f"[REPORT] Tổng diện tích sản phẩm: {tong_pixel_thanh_coc} pixels")
    print(f"[REPORT] Số vùng nghi vấn quét thô (OpenCV): {so_vung_nghi_van}")
    print(f"[REPORT] Số khuyết tật thật được AI duyệt ĐỎ: {so_vung_loi_ai_duyet}")
    print(f"[REPORT] TỶ LỆ HƯ HẠI KHUYẾT TẬT TRÊN BỀ MẶT: {ty_le_hu_hai:.2f}%")
    print("====================================================")

    cv2.putText(img_hien_thi_ket_qua, f"Tong pixel vat the: {tong_pixel_thanh_coc}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    cv2.putText(img_hien_thi_ket_qua, f"Ty le hu hai be mat: {ty_le_hu_hai:.2f}%", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    cv2.imwrite(output_path, img_hien_thi_ket_qua)
    cv2.imshow("Anh Ket Qua DL Khong Mau (Do: Loi, Vang: Nhiễu hoa van)", img_hien_thi_ket_qua)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    chay_pipeline_tu_khoa_khong_mau('caihu2.png', 'anh_xu_ly.jpg')