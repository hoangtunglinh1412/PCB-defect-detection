import os
from ultralytics import YOLO

def train_yolo_model():
    print("Khởi tạo mô hình YOLO...")
    model = YOLO("yolo11n.pt") 

    print("Bắt đầu quá trình huấn luyện bằng GPU 3060 Ti...")
    results = model.train(
        data="data/pcb_dataset.yaml",
        epochs=50,                  
        imgsz=640,                  
        batch=32,                   
        device=0,                   
        workers=2,
        cache=True,                 
        project="runs/pcb_defects", 
        name="yolov11_baseline",    
        patience=10                 
    )
    print("Huấn luyện hoàn tất! Trọng số đã được lưu.")

if __name__ == "__main__":
    train_yolo_model()