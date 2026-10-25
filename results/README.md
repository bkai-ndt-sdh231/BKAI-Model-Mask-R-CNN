
# Experimental Results

### BKAI — Progressive Mask R-CNN Framework for Concrete Crack Segmentation

**Published Research | Springer Nature | 2026**

This directory presents selected experimental results associated with the research article:

**An integrated mask R-CNN framework for concrete crack segmentation and skeleton-based quantification in automated infrastructure inspection**

**Authors:** Dat-Thanh Nguyen, Nhat-Quang Tran, Duc-Hoc Tran

**Journal:** *Journal of Building Pathology and Rehabilitation*, Volume 11, Article 228 (2026)

**Full Paper:** https://doi.org/10.1007/s41024-026-00900-z

---

## 1. Results Overview

The proposed framework was evaluated through two complementary experimental protocols:

- **Protocol 1 — Progressive Model Development:** Evaluation of the Mask R-CNN configurations M0–M3 to examine successive improvements in proposal adaptation, boundary refinement, and topology refinement.
- **Protocol 2 — Controlled Model Comparison:** Comparative evaluation of the proposed M3 configuration against Mask R-CNN baseline (M0), DeepLabV3+, and YOLOv8m-Seg using a fixed TEST1007 subset.

The M4 stage provides skeleton-based geometric characterization of refined crack masks rather than an additional segmentation accuracy improvement stage.

The results presented here are selected highlights. Detailed experimental settings, quantitative analyses, and scientific discussion are available in the published article.

---

## 2. Progressive Model Development (M0–M3)

The progressive framework introduces successive modifications to the Mask R-CNN baseline.

| Model | Configuration | Primary Objective |
|---|---|---|
| M0 | Baseline Mask R-CNN | Initial instance segmentation |
| M1 | Crack-Oriented Proposal Adaptation | Improve crack-region proposal representation |
| M2 | PointRend Boundary Refinement | Improve crack-mask boundaries |
| M3 | Morphological Topology Refinement | Enhance crack-mask continuity and topology |

### Selected Development Results

| Model | F1-score / Dice (%) | IoU (%) |
|---|---:|---:|
| M0 | 81.01 | 68.08 |
| M1 | 80.57 | 67.46 |
| M2 | 82.62 | 70.39 |
| **M3** | **84.76** | **73.54** |

These results illustrate the performance of the successive configurations under the development evaluation protocol.

Performance does not improve uniformly at every stage. The strongest development-stage F1-score and IoU are obtained by M3.

---

## 3. Controlled Comparison — TEST1007

A controlled comparison was performed using the fixed TEST1007 evaluation subset.

### Proposed M3 Performance

| Metric | Result |
|---|---:|
| Precision | 88.69% |
| Recall | 84.07% |
| **F1-score / Dice** | **86.32%** |
| **IoU** | **75.94%** |

### Comparison with Other Models

| Model | Precision (%) | Recall (%) | F1-score (%) | IoU (%) |
|---|---:|---:|---:|---:|
| M0 Mask R-CNN | 84.28 | 81.12 | 82.67 | 70.45 |
| **Proposed M3** | **88.69** | 84.07 | **86.32** | **75.94** |
| DeepLabV3+ | 90.02 | 80.41 | 84.94 | 73.83 |
| YOLOv8m-Seg | 75.19 | 93.26 | 83.26 | 71.32 |

Under this controlled evaluation, M3 achieved the highest F1-score and IoU among the compared models.

DeepLabV3+ achieved the highest precision, while YOLOv8m-Seg achieved the highest recall, demonstrating distinct precision–recall trade-offs.

Development-stage and controlled-comparison metrics should not be directly interchanged.

---

## 4. Qualitative Segmentation Results

Representative crack images demonstrate the ability of the framework to identify concrete cracks with different orientations, widths, and geometric characteristics.

The qualitative comparisons illustrate how the progressive configurations affect predicted mask boundaries, continuity, and local crack structures.

Representative figures are available in the main repository:

- Crack detection and segmentation examples
- M0–M3 qualitative mask comparison
- Controlled TEST1007 model comparison

**Research figures:** [View Selected Research Images](../assets/images/bkai_paper_2026/)

---

## 5. Skeleton-Based Geometric Characterization (M4)

The M4 stage processes refined crack masks to derive image-based geometric descriptors.

The main operations include:

- Crack-mask skeletonization
- Centerline extraction and characterization
- Distance-transform-based width estimation
- Crack-wise geometric analysis

These measurements describe crack geometry in the image domain.

**Measurement limitation:** Pixel-based crack length and width must not be interpreted as absolute physical dimensions without a valid spatial scale or camera calibration.

M4 extends the framework toward quantitative crack characterization; it should not be described as an additional segmentation benchmark with independently established accuracy improvements.

---

## 6. BKAI Research Demonstration

The BKAI Web/App illustrates the integration of concrete crack segmentation and image-based geometric analysis into an inspection-oriented workflow.

Selected functions include:

- Image upload and crack analysis
- Visualization of predicted crack masks
- Individual crack identification
- Quantitative image-level and crack-level information
- Graphical analysis of crack characteristics
- Structured inspection reports

**Research Website:** https://bkai.b12sites.com/index

The application is a research demonstration. Its outputs are not a substitute for calibrated structural inspection or professional engineering assessment.

---

## 7. Interpretation and Limitations

The experimental findings support the use of progressive Mask R-CNN refinement for concrete crack segmentation.

The proposed M3 configuration demonstrated a favorable balance of segmentation precision, recall, and spatial overlap under the controlled test protocol.

However, the results should be interpreted with consideration of dataset characteristics, evaluation settings, image quality, and differences between experimental protocols.

Additional field validation and physical measurement calibration remain important for practical engineering applications.

---

## 8. Complete Experimental Results

This directory intentionally presents a concise research summary.

For the full experimental methodology, parameter settings, ablation studies, comparative results, statistical analysis, geometric characterization, and discussion, please consult the peer-reviewed publication.

**Read the complete article:**

https://doi.org/10.1007/s41024-026-00900-z

---

## 9. Citation

Nguyen, D.-T., Tran, N.-Q., & Tran, D.-H. (2026).

*An integrated mask R-CNN framework for concrete crack segmentation and skeleton-based quantification in automated infrastructure inspection.*

Journal of Building Pathology and Rehabilitation, 11, Article 228.

https://doi.org/10.1007/s41024-026-00900-z

---

**BKAI Research Repository**

https://github.com/bkai-ndt-sdh231/BKAI-Model-Mask-R-CNN

© 2026 Dat-Thanh Nguyen. Reuse of repository materials is subject to the applicable copyright notices, publication agreements, and third-party rights.
