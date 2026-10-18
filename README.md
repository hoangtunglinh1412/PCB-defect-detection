# 🔬 PCB Defect Detection: Advanced Automated Quality Control

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CUDA_12.1-EE4C2C.svg)
![YOLO](https://img.shields.io/badge/YOLO11-Ultralytics-00FFFF.svg)
![SAHI](https://img.shields.io/badge/SAHI-Vision-green.svg)

## 📖 Project Overview
This project implements an end-to-end Computer Vision pipeline designed to automate Quality Control (QC) in electronics manufacturing. It detects and classifies microscopic physical defects on Printed Circuit Boards (PCBs) with high precision. 

To solve the inherent challenge of **Small Object Detection** (where tiny defects are often lost when high-resolution images are downscaled), this system integrates **Ultralytics YOLO11** with **SAHI (Slicing Aided Hyper Inference)**.

## ✨ Key Features & Technical Highlights
* **State-of-the-Art Architecture:** Trained from scratch using the YOLO11 Nano model, optimized for real-time edge deployment.
* **Small Object Detection (SAHI):** Utilizes sliding window inference to slice input images into `256x256` overlapping patches, allowing the model to identify microscopic anomalies without losing global context.
* **Hardware Acceleration:** Fully optimized Data Pipeline preventing Windows memory paging issues (Virtual Memory crashes) by fine-tuning dataloader workers and GPU memory allocation (`device=0`, `cache=True`).

## 📊 Dataset & Model Performance
The model is trained to detect 6 distinct classes of PCB defects:
`missing_hole`, `mouse_bite`, `open_circuit`, `short`, `spur`, `spurious_copper`

**Training Metrics (Validation Set):**
* **mAP@50:** `0.990` (99.0%)
* **Precision:** `0.977`
* **Recall:** `0.984`
* *The model demonstrates exceptional convergence, maintaining a false positive rate of less than 2.3% across all classes.*

## 📂 Project Structure
```text
pcb-defect-detection/
├── data/
│   ├── images/ (train/val)
│   ├── labels/ (train/val)
│   └── pcb_dataset.yaml      # YOLO dataset configuration
├── src/
│   ├── train.py              # Custom YOLO training script (GPU Optimized)
│   └── predict.py            # Inference script integrated with SAHI
├── runs/                     # Model weights (best.pt) and evaluation plots
├── requirements.txt
└── README.md