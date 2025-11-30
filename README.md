# BKAI-Model Mask R-CNN
🇬🇧 BKAI – Concrete Crack Detection using Mask R-CNN + ResNet50
🇻🇳 BKAI – Phát hiện vết nứt bê tông bằng Mask R-CNN + ResNet50
1. Overview | Tổng quan

🇬🇧
This repository contains the trained Mask R-CNN (ResNet50-FPN) model, code, and evaluation results developed for the master’s thesis “Deep Learning-Based Detection and Segmentation of Concrete Cracks in Civil Infrastructure”.
The model is trained on more than 12,000+ concrete crack images using the COCO segmentation format and achieves high accuracy in crack detection and segmentation.

🇻🇳
Repository này chứa mô hình Mask R-CNN (ResNet50-FPN) đã huấn luyện, mã nguồn và toàn bộ kết quả đánh giá phục vụ luận văn thạc sĩ “Ứng dụng học sâu để phát hiện và phân đoạn vết nứt bê tông trong công trình hạ tầng”.
Mô hình được huấn luyện trên 12.000+ ảnh vết nứt bê tông theo định dạng COCO segmentation, đạt độ chính xác cao.

📂 2. Repository Structure | Cấu trúc thư mục
BKAI-Model-Mask-R-CNN/
│
├── models/
│     ├── mask_rcnn_resnet50_v7.pth      # Model weights
│     ├── metrics_v7.json                # Evaluation metrics
│
├── results/
│     ├── crack_pair_01.png              # Before–After image
│     ├── crack_pair_02.png
│     └── ...
│
├── notebooks/
│     ├── Model Mask R-CNN.ipynb         # Full inference notebook
│
└── README.md

3. Model Download | Tải mô hình

🇬🇧
The trained model (.pth) and evaluation files are available in the Releases section:

 https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN/releases

🇻🇳
Mô hình đã huấn luyện và file đánh giá được tải tại mục Releases:

 https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN/releases

 4. How to Load the Model in Google Colab | Hướng dẫn load mô hình trên Colab
 Step 1 — Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

 Step 2 — Install Detectron2
!pip install -U 'git+https://github.com/facebookresearch/detectron2.git'

Step 3 — Load configuration
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
))
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"

Step 4 — Load trained model (.pth)
cfg.MODEL.WEIGHTS = "/content/drive/MyDrive/BKAI_MaskRCNN/mask_rcnn_resnet50_v7.pth"
predictor = DefaultPredictor(cfg)
print("Model loaded successfully!")

5. Evaluation Summary | Tóm tắt kết quả đánh giá

🇬🇧
The model achieves strong performance in crack detection and segmentation:

🇻🇳
Mô hình đạt hiệu suất cao trong phát hiện và phân loại vết nứt:

mAP (BBox): 63.99%
mAP50 (BBox): 82.89%
mAP (Segmentation): 21.53%
F1-score (Crack/No-Crack): 0.991
Accuracy: 98%
Confusion Matrix:
Pred: No Crack	Pred: Crack
GT: No Crack	77	15
GT: Crack	1	907

6. Sample Results | Kết quả minh họa

All sample Before–After images are available in:
results/

Ví dụ:

crack_pair_01.png
crack_pair_02.png
...


These images show the model's ability to detect cracks, draw bounding boxes, and generate segmentation masks.

7. Reproduce Metrics | Tái lập chỉ số đánh giá

Load metrics:

import json
with open("/content/.../metrics_v7.json") as f:
    metrics = json.load(f)

metrics

 8. SHA256 Checksums | Kiểm chứng SHA256

🇬🇧
To ensure file integrity, each file in the release includes SHA256 signatures.

🇻🇳
Để đảm bảo tính toàn vẹn, mỗi file đều có mã SHA256:

File	SHA256
BKAI_Results.rar	208ea63f178430105bc938c0a9b144b4f368b9208639db76745549f29cbb6a4e
mask_rcnn_resnet50_v7.pth	7505777bd5a5cc709e23876ef2f9acbf3b8206b08a80a24aab082f27dfd3b378
metrics_v7.json	f51de9b3adf8b7311f698c5fadebdd3befcf0152d0cd4c96877e0eded18c622b

9. Citation | Trích dẫn
If you use this work, please cite as:
Nguyen Dat Thanh (2025). 
BKAI – Concrete Crack Detection using Mask R-CNN + ResNet50.
GitHub Repository: https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN

10. Author | Tác giả

Nguyễn Đạt Thạnh
Master’s Program – Civil Engineering & AI (BKAI Lab)
Email: nguyendatthanh26061996@gmail.com
