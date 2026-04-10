<p align="center">
  <img src="assets/images/bkai_pipeline.png" width="1000">
</p>

<h1 align="center">BKAI – Mask R-CNN + ResNet50</h1>

<p align="center">
  <b>Concrete Crack Detection & Instance Segmentation</b><br>
  Mask R-CNN | ResNet-50 | FPN | Detectron2
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0-red">
  <img src="https://img.shields.io/badge/Detectron2-FacebookAI-green">
  <img src="https://img.shields.io/badge/Task-Instance%20Segmentation-blue">
  <img src="https://img.shields.io/badge/Status-Research-orange">
</p>

---

## 🔥 Overview

BKAI is a deep learning framework designed for **automatic concrete crack detection and instance segmentation** in civil infrastructure.

The model enables:

- 🎯 Pixel-level crack segmentation  
- 🔍 Instance-level crack separation  
- 📏 Crack morphology analysis  
- ⚡ High accuracy in real-world conditions  

---

## 🧠 Computer Vision Tasks

<p align="center">
  <img src="assets/images/cv_tasks_overview.png" width="900">
</p>

---

## 🧬 Model Evolution

<p align="center">
  <img src="assets/images/model_evolution_timeline.png" width="900">
</p>

---

## 🏗️ Mask R-CNN Architecture

<p align="center">
  <img src="assets/images/mask_rcnn_architecture.png" width="950">
</p>

---

## 🔍 Feature Pyramid Network

<p align="center">
  <img src="assets/images/fpn_structure.png" width="900">
</p>

---

## ⚙️ ROI Align

<p align="center">
  <img src="assets/images/roi_align.png" width="900">
</p>

---

## 🧱 Backbone (ResNet-50)

<p align="center">
  <img src="assets/images/resnet50_backbone.png" width="900">
</p>

---

## 📊 Dataset Analysis

<p align="center">
  <img src="assets/images/dataset_analysis.png" width="1000">
</p>

- 24,000 training images  
- 1,000 validation images  
- Multi-scale crack distribution  
- Real-world + augmented data  

---

## 📉 Training Process

<p align="center">
  <img src="assets/images/training_loss.png" width="800">
</p>

<p align="center">
  <img src="assets/images/loss_components.png" width="800">
</p>

---

## 📈 Evaluation

<p align="center">
  <img src="assets/images/map.png" width="600">
</p>

<p align="center">
  <img src="assets/images/confusion_matrix.png" width="600">
</p>

<p align="center">
  <img src="assets/images/pr_curve.png" width="600">
</p>

<p align="center">
  <img src="assets/images/f1_curve.png" width="600">
</p>

---

## 🖼️ Prediction Results

<p align="center">
  <img src="assets/images/prediction_results.png" width="1000">
</p>

✔ Instance segmentation  
✔ Bounding box + mask  
✔ Confidence score  

---

## 🔥 Highlights

- Mask R-CNN (Instance Segmentation)
- ResNet-50 + FPN backbone
- COCO evaluation metrics (mAP, AP50, AP75)
- High precision (~99% F1-score)
- Optimized for thin crack detection
- Deployable via Streamlit

---

## ⚙️ Installation

```bash
git clone https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN.git
cd BKAI-Model-Mask-R-CNN
pip install -r requirements.txt
