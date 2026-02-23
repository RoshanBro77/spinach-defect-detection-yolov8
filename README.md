# 🌿 Spinach Defect Detection using YOLOv8

## 📌 Project Overview

This project implements a YOLOv8-based object detection system to detect and localize defect regions on spinach leaves.

The trained model detects the following defect classes:

- GOOD
- YELLOW
- HOLE
- TRACK
- WSPOT
- FSPOT

The system uses bounding-box annotations in YOLO format and is evaluated using standard object detection metrics.

---

## 📊 Model Performance

- Model: YOLOv8n (Nano)
- Task: Multi-class object detection
- Metric: mAP@0.5
- Test mAP@0.5: ~0.869

Generated evaluation artifacts:

- Precision–Recall curves
- F1-score curve
- Confusion matrix
- Validation metrics summary

---

## 🗂 Project Structure

Spinach_Project/
│
├── data/
│ ├── raw/
│ │ ├── dataset1/
│ │ └── dataset2/
│ │
│ └── splits/
│ ├── train/
│ │ ├── images/
│ │ └── labels/
│ ├── val/
│ │ ├── images/
│ │ └── labels/
│ └── test/
│ ├── images/
│ └── labels/
│
├── 01_prepare_dataset.py
├── spinach_detection.ipynb
├── dataset.yaml
├── requirements.txt
└── README.md

---

# 🚀 Setup & Execution Guide

## 🧱 Step 1 — Install Packages (Terminal)

Create and activate a virtual environment:

python3 -m venv spinach_env
source spinach_env/bin/activate

Upgrade pip and install dependencies:

pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -r requirements.txt

---

## 📥 Step 2 — Download and Place the Datasets

Dataset 1:
https://data.mendeley.com/datasets/sy69db2nz5/2
Download All → Extract into:
Spinach_Project/data/raw/dataset1/

Dataset 2:
https://data.mendeley.com/datasets/fzgghkgf6g/1
Download All → Extract into:
Spinach_Project/data/raw/dataset2/

After extraction:

data/
├── raw/
│ ├── dataset1/
│ └── dataset2/

---

## ⚙️ Step 3 — Prepare & Split the Dataset (Terminal)

python 01_prepare_dataset.py

This script:

- Reads images from data/raw/
- Converts annotations to YOLO format
- Splits data into train / val / test
- Saves processed data into data/splits/

---

## 📄 dataset.yaml Configuration

path: ./data/splits

train: train/images
val: val/images
test: test/images

nc: 6

names:
0: GOOD
1: YELLOW
2: HOLE
3: TRACK
4: WSPOT
5: FSPOT

---

## 🚀 Training

yolo detect train model=yolov8n.pt data=dataset.yaml epochs=100 imgsz=640 batch=8

Best weights:
runs/detect/train/weights/best.pt

---

## 🧪 Evaluation

yolo detect val model=runs/detect/train/weights/best.pt data=dataset.yaml split=test

---

## ⚠️ Known Limitations

- Class imbalance (e.g., YELLOW underrepresented)
- Confusion between visually similar defect types
- YOLOv8n used due to hardware constraints

---

## 👨‍💻 Author

Roshan Sadha Sanker
Master of Data Science
