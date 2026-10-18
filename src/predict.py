import os
from sahi.predict import get_sliced_prediction
from sahi.models.ultralytics import UltralyticsDetectionModel

def run_sahi_inference(image_path: str, model_path: str):
    print("🚀 Khởi tạo mô hình YOLO với lõi SAHI...")
    
    # 1. Bọc mô hình YOLO bằng Wrapper của SAHI, kích hoạt GPU
    detection_model = UltralyticsDetectionModel(
        model_path=model_path,
        confidence_threshold=0.4, # Bỏ qua các khung dự đoán có độ tin cậy dưới 40%
        device="cuda:0"           # Ép SAHI chạy trên card 3060 Ti
    )

    print(f"Đang dùng SAHI để rà quét bảng mạch: {image_path}")
    
    # 2. Cắt lát ảnh (Slicing) và Dự đoán
    # Thuật toán sẽ dùng một khung 256x256 trượt qua toàn bộ bức ảnh gốc
    result = get_sliced_prediction(
        image_path,
        detection_model,
        slice_height=256,         
        slice_width=256,
        overlap_height_ratio=0.2, # Độ chồng lấn 20% giúp vật thể ở mép ảnh không bị cắt đôi
        overlap_width_ratio=0.2
    )

    # 3. Xuất kết quả
    output_dir = "runs/predict_sahi/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Lưu ảnh đã vẽ bounding box
    result.export_visuals(export_dir=output_dir, file_name="sahi_result")
    
    print(f"🎉 Hoàn tất! Ảnh kết quả độ phân giải cao được lưu tại: {output_dir}/sahi_result.png")

if __name__ == "__main__":
    # Cập nhật đường dẫn file best.pt chuẩn xác từ log của bạn
    trained_weights = r"runs\detect\runs\pcb_defects\yolov11_baseline-7\weights\best.pt"
    
    # Chọn 1 bức ảnh bất kỳ trong tập val để test (đảm bảo file này tồn tại)
    test_image = r"data\images\val\l_light_01_missing_hole_11_1_600.jpg" 
    
    if os.path.exists(test_image) and os.path.exists(trained_weights):
        run_sahi_inference(test_image, trained_weights)
    else:
        print("Lỗi: Không tìm thấy ảnh test hoặc file trọng số!")