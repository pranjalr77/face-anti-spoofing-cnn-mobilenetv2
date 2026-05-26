# Face Anti-Spoofing: CNN Liveness Detection vs MobileNetV2 Transfer Learning

## Overview
A comparative research study reproducing and evaluating two published face anti-spoofing approaches on the **NUAA face anti-spoofing dataset**. Both pipelines are implemented from scratch in Jupyter Notebooks and evaluated using standard classification metrics.

---

## Research Papers Reproduced

| | Paper 1 | Paper 2 |
|---|---|---|
| **Title** | Improved Facial Biometric Authentication Using MobileNetV2 | Face Anti-Spoofing Using CNN Classifier and Face Liveness Detection |
| **Core Approach** | Transfer Learning (MobileNetV2) | Sequential Liveness + Custom CNN |
| **Classification** | Binary: Real vs Fake | Binary: Real vs Fake |
| **Liveness** | Not applicable | Motion-based frame analysis |
| **Deployment** | Jupyter Notebook | Jupyter Notebook + Webcam |

---

## Dataset
**NUAA Face Anti-Spoofing Dataset**
- Binary classification: real (genuine faces) vs fake (spoofed/printed faces)
- Prepared from raw split text files: client_train_raw.txt, client_test_raw.txt, imposter_train_raw.txt, imposter_test_raw.txt
- Split into train/ and test/ directories with real/ and fake/ subfolders
  
| Split | Real | Fake | Total |
|---|---|---|---|
| Train (70%) | 1,743 | 1,748 | 3,491 |
| Validation (30%) | — | — | 1,046 |
| Test | 3,362 | 5,761 | 9,123 |

> Dataset not included in this repository. Download from the official NUAA dataset source.

---

## Implementation Details

### Paper 1 — MobileNetV2 Transfer Learning (Paper - Improved Facial Biometric Authentication Using MobileNetV2)

**Architecture:**
- Base: MobileNetV2 pretrained on ImageNet (include_top=False)
- Head: GlobalAveragePooling2D → Dropout(0.30) → Dense(1, sigmoid)
- Input size: 224×224

**Model Size:** Total params: 2,259,265 (8.62 MB) | Trainable (Stage 1): 1,281

**Training Pipeline:**
- **Stage 1 — Feature Extraction:** Base model frozen, only classification head trained
  - Optimizer: Adam (lr=1e-4) | Epochs: 20 | Batch size: 25
- **Stage 2 — Fine-Tuning:** Last 30 layers of base model unfrozen
  - Optimizer: Adam (lr=1e-5) | Epochs: 10
- Validation split: 30%
- Data augmentation: rotation, shift, zoom, horizontal flip
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

**Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report

---

### Paper 2 — Sequential CNN + Liveness Detection (Paper - Face Anti-Spoofing Using CNN Classifier and Face liveness Detection)

**Architecture:**
- Custom CNN: Conv2D(32) → Batch Norm → MaxPool → Conv2D(64) → Batch Norm → MaxPool → Conv2D(128) → Batch Norm → MaxPool → Dense(256) → Dropout(0.5) → Dense(1, sigmoid)
- Input size: 128×128
- Optimizer: Adam | Loss: Binary Crossentropy | Epochs: 20 | Batch size: 32

**Model Size:** Total params: 6,517,185 (24.86 MB) | Trainable: 6,516,737

**Two-Stage Anti-Spoofing Pipeline:**
- **Stage 1 — Liveness Module:**
  - Real-time webcam face detection using Haar Cascades
  - Grayscale frame-to-frame comparison with motion thresholding (threshold=8.0)
  - Requires 5 consecutive live frames to pass as LIVE
  - Decision: LIVE / SPOOF / NO FACE
- **Stage 2 — CNN Classifier (runs only if Stage 1 = LIVE):**
  - Captured webcam frame passed to trained CNN
  - Final prediction: real or fake

**Practical Adaptation Note:**
The original paper uses eye blinking and lip movement detection on an Android application. This implementation adapts the liveness stage for desktop/Jupyter using webcam-based motion thresholding — preserving the same sequential liveness-first logic while ensuring reproducibility without Android dependencies.

**Evaluation:** Accuracy, Precision, Recall, F1-Score, Confusion Matrix, Classification Report

---

## Tech Stack
- Python 3.11
- TensorFlow 2.20 / Keras
- OpenCV (cv2)
- scikit-learn
- NumPy, Matplotlib

---

## Results

| Metric | MobileNetV2 (Paper 1) | CNN + Liveness (Paper 2) |
|---|---|---|
| **Accuracy** | 0.9570 | 0.8190 |
| **Precision** | 0.896 | 0.6714 |
| **Recall** |0.9994 | 0.9967 |
| **F1-Score** | 0.9449 | 0.8023 |


---

## Project Structure

face-anti-spoofing-cnn-mobilenetv2/

└── mobilenetv2_antispoofing.ipynb 

└── cnn_liveness_detection.ipynb  

└── README.md

---

## How to Run

1. Clone this repository
2. Download the NUAA dataset
3. Update the dataset_root path in both notebooks to your local dataset path
4. Install dependencies: pip install tensorflow opencv-python scikit-learn numpy matplotlib
5. Run each notebook end-to-end in Jupyter



