# 🧠 Multimodal Dementia Classification Using MRI & Clinical Data

This repository implements a **late-fusion multimodal deep learning framework** for classifying dementia severity using **3D MRI brain scans** and **clinical tabular data**.  
The model predicts one of four dementia stages: **Non-Demented, Very Mild, Mild, and Moderate**.

---

## 📌 Project Overview

Dementia diagnosis benefits from combining neuroimaging with clinical indicators.  
This project integrates:

- **3D structural MRI scans** (spatial brain patterns)
- **Clinical tabular features** (demographics & cognitive scores)

A **late-fusion neural architecture** is used, where each modality is processed independently before feature fusion and final classification.

---

## 🗂 Dataset

**Dataset:** OASIS-1 (Open Access Series of Imaging Studies)

### Imaging Data
- Format: 3D NIfTI (`.nii`)
- Modality: Structural MRI
- Preprocessing:
  - Resized to `128 × 128 × 128`
  - Intensity normalization
  - On-the-fly 3D augmentation (rotation, flipping)

### Tabular Data
Clinical features used:
- Age  
- Gender (M/F)  
- Education Level (`Educ`)  
- Socioeconomic Status (`SES`)  
- Mini-Mental State Examination (`MMSE`)  

Missing values are handled during preprocessing.

### Target Labels
- **Clinical Dementia Rating (CDR)** mapped to 4 classes:
  - `0` → Non-Demented  
  - `0.5` → Very Mild  
  - `1` → Mild  
  - `2` → Moderate  

---

## 🏗 Model Architecture

### 1️⃣ Image Branch – 3D CNN
Processes 3D MRI volumes.

- 3 × (Conv3D → MaxPool3D)
- GlobalAveragePooling3D
- Dense(64)

### 2️⃣ Tabular Branch – Feed-Forward Network
Processes clinical features.

- Dense(32)
- Dense(16)

### 3️⃣ Late Fusion Head
- Concatenation of image & tabular embeddings
- Dense(32) + Dropout(0.5)
- Softmax output (4 classes)

📌 **Why Late Fusion?**  
Each modality learns independently before combining complementary information, improving robustness and performance.

---

## ⚙️ Training Strategy

- **Optimizer:** Adam  
- **Learning Rate:** `1e-4`
- **Loss Function:** Categorical Crossentropy
- **Batch Handling:** Custom `MultimodalDataGenerator`
  - MRI loading using NiBabel
  - 3D augmentation using SciPy
  - Class weighting to address class imbalance

---

## 🧪 Experiments & Results

### Baseline Model (3D CNN + Tabular Fusion)
- **Validation Accuracy:** ~76%
- Stable convergence and strong per-class performance

### Transfer Learning Experiment (Discarded)
- Tested 2D pre-trained models:
  - VGG16
  - ResNet
  - Inception
- Used 3-plane slice stacking (Axial, Coronal, Sagittal)
- ❌ Observed **negative transfer**
- ✔️ Custom 3D CNN significantly outperformed all 2D approaches

---

## 📊 Evaluation

The final model is evaluated on a **20% hold-out test set**.

Evaluation metrics:
- Precision
- Recall
- F1-score
- Confusion Matrix (per-class performance visualization)

---

## 🧰 Tech Stack

- **Python:** 3.12  
- **Deep Learning:** TensorFlow / Keras  
- **Medical Imaging:** NiBabel  
- **Scientific Computing:** NumPy, SciPy  
- **Visualization:** Matplotlib, Seaborn  
- **Evaluation:** Scikit-learn  

---

## 📁 Repository Structure

