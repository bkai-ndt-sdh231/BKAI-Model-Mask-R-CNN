

<div align="center">

# An integrated mask R-CNN framework for concrete crack segmentation and skeleton-based quantification in automated infrastructure inspection

### Official Research Repository | Springer Nature | 2026

**Dat-Thanh Nguyen · Nhat-Quang Tran · Duc-Hoc Tran**

*Journal of Building Pathology and Rehabilitation*  
**Volume 11, Article 228 (2026)**

[![DOI](https://img.shields.io/badge/DOI-10.1007%2Fs41024--026--00900--z-blue)](https://doi.org/10.1007/s41024-026-00900-z)
[![Paper](https://img.shields.io/badge/Read_Full_Paper-Springer_Nature-1769aa)](https://doi.org/10.1007/s41024-026-00900-z)
![Year](https://img.shields.io/badge/Published-2026-success)

**Mask R-CNN | ResNet-50-FPN | PointRend | Crack Segmentation | Skeletonization**

</div>

---

## 1. Research Overview

Automated detection and characterization of concrete cracks are important for infrastructure inspection, structural condition assessment, and maintenance planning.

This research presents a progressive **Mask R-CNN framework (M0–M4)** that integrates instance-aware concrete crack segmentation, crack-oriented proposal adaptation, boundary refinement, morphological topology refinement, and skeleton-based geometric characterization.

The framework is designed to improve the representation of irregular, elongated, and discontinuous crack patterns while supporting downstream quantitative analysis.

The proposed approach is evaluated through progressive model development and controlled comparison with established segmentation architectures. An associated **BKAI Web/App** demonstrates the application of the framework to image-based crack analysis, visualization, and inspection reporting.

> **About this repository:** This GitHub repository provides a concise introduction to the published research, selected experimental results, visual demonstrations, and associated code resources. For detailed methodology, experimental configurations, complete statistical analysis, and scientific discussion, please refer to the published journal article.

**[Read the Full Research Article on Springer Nature](https://doi.org/10.1007/s41024-026-00900-z)**

---

## 2. Research Contributions

The principal contributions of this study include:

- **Progressive instance-aware crack segmentation:** Integration of crack-oriented proposal adaptation, PointRend boundary refinement, and morphological topology refinement within a Mask R-CNN framework.

- **Controlled experimental evaluation:** Assessment of successive model improvements and comparison against DeepLabV3+ and YOLOv8m-Seg using a consistent evaluation protocol.

- **Skeleton-based geometric characterization:** Extraction of crack centerlines and image-domain geometric descriptors to support crack-wise quantitative analysis.

- **Engineering-oriented integration:** Implementation of the research workflow in BKAI Web/App for crack visualization, analysis, and structured reporting.

The methodological contribution lies in the progressive integration and evaluation of established techniques rather than claiming independent invention of their underlying algorithms.

---

## 3. Proposed Mask R-CNN Framework

The proposed framework progressively develops the baseline Mask R-CNN model through five configurations.

| Stage | Configuration | Primary Function |
|:---:|---|---|
| **M0** | Mask R-CNN Baseline | Instance segmentation using ResNet-50-FPN |
| **M1** | Crack-Oriented Proposal Adaptation | Improve candidate-region representation for elongated cracks |
| **M2** | PointRend Boundary Refinement | Refine uncertain crack boundaries |
| **M3** | Morphological Topology Refinement | Improve mask continuity and suppress isolated artifacts |
| **M4** | Skeleton-Based Geometric Characterization | Extract centerlines and image-domain geometric descriptors |

### Progressive Framework Architecture

<div align="center">

<img src="assets/images/bkai_paper_2026/fig06_progressive_m1_m4.png" alt="Progressive Mask R-CNN Framework M1 to M4" width="85%">

*Progressive development of the proposed Mask R-CNN framework, illustrating the main processing stages from M1 to M4.*

</div>

The progressive design preserves an instance-aware processing pathway while addressing crack-specific challenges related to geometry, boundaries, topology, and downstream characterization.

Detailed architectural configurations, mathematical formulations, and implementation considerations are presented in the published article.

---

## 4. Dataset and Experimental Design

The study employs a curated multi-source concrete crack dataset comprising:

| Dataset Characteristic | Description |
|---|---|
| Original curated images | 10,000 |
| Crack images | 5,000 |
| Non-crack images | 5,000 |
| Input resolution | 640 × 640 pixels |
| Annotation format | COCO-compatible instance segmentation |
| Development dataset | Approximately 26,000 images including training augmentation |
| Controlled comparison | Fixed TEST1007 evaluation subset |

Two distinct experimental protocols were established.

**Protocol 1 — Progressive Development**

Evaluates the contributions of the successive Mask R-CNN configurations using training-only augmentation.

**Protocol 2 — Controlled Cross-Model Comparison**

Evaluates the proposed framework against established segmentation architectures using consistently prepared data and a fixed held-out test subset.

These protocols serve different analytical purposes and their results should not be directly mixed.

For dataset composition, annotation procedures, augmentation strategies, statistical characteristics, and experimental settings, consult the published article.

---

## 5. Selected Experimental Results

The proposed M3 configuration achieved the following results under the controlled **TEST1007** protocol.

<div align="center">

| Evaluation Metric | M3 Result |
|:---|---:|
| **Precision** | **88.69%** |
| **Recall** | **84.07%** |
| **F1-score / Dice** | **86.32%** |
| **Intersection over Union (IoU)** | **75.94%** |

</div>

### Comparison with Established Segmentation Models

<div align="center">

<img src="assets/images/bkai_paper_2026/fig20_test1007_comparison.png" alt="Controlled comparison between M0, M3, DeepLabV3+ and YOLOv8m-Seg" width="85%">

*Selected performance comparison under the controlled TEST1007 evaluation protocol.*

</div>

The proposed M3 configuration achieved the highest F1-score/Dice and IoU among the evaluated models under this controlled test protocol.

Different models exhibited distinct Precision–Recall trade-offs, highlighting the importance of evaluating both segmentation quality and application requirements.

The full experimental results, component-level evaluation, uncertainty analysis, and detailed comparisons are available in the journal article.

**[View the Complete Experimental Analysis](https://doi.org/10.1007/s41024-026-00900-z)**

---

## 6. Visual Demonstrations

### Concrete Crack Detection and Instance Segmentation

The framework produces instance-aware predictions that support crack localization and pixel-level mask visualization.

<div align="center">

<img src="assets/images/bkai_paper_2026/fig15_crack_prediction_examples.png" alt="Concrete crack detection and segmentation examples" width="85%">

*Representative examples of concrete crack detection and segmentation.*

</div>

### Progressive Segmentation Refinement

The following visualization demonstrates representative segmentation outputs across the progressive model configurations.

<div align="center">

<img src="assets/images/bkai_paper_2026/fig17_m0_m3_qualitative_masks.png" alt="Qualitative comparison of progressive Mask R-CNN configurations" width="85%">

*Qualitative visualization of crack segmentation from M0 to M3.*

</div>

The complete article provides further analysis of segmentation performance, crack-mask refinement, and experimental limitations.

---

## 7. Skeleton-Based Crack Characterization

The final stage, M4, extends the refined crack masks toward quantitative geometric characterization.

The processing workflow incorporates skeleton extraction, centerline characterization, and distance-based width estimation.

These operations provide image-based geometric descriptors for individual crack regions.

**Important limitation:** Without a known physical scale, camera calibration, or equivalent spatial reference, the estimated crack lengths and widths are image-domain measurements rather than absolute physical dimensions.

Detailed formulations and geometric characterization results are presented in the published article.

---

## 8. BKAI Web/App — Research Demonstration

The proposed framework is associated with the **BKAI AI-Based Concrete Crack Detection and Analysis System**, demonstrating its potential application in infrastructure inspection workflows.

Selected capabilities include:

- Concrete crack image analysis
- Visualization of predicted crack regions
- Instance-level crack information
- Image-based geometric characterization
- Quantitative result visualization
- Structured inspection reporting

### BKAI System Workflow

<div align="center">

<img src="assets/images/bkai_paper_2026/fig21_bkai_web_workflow.png" alt="BKAI AI-based concrete crack detection and analysis workflow" width="85%">

*Overview of the BKAI research demonstration and inspection-oriented analysis workflow.*

</div>

### Related Research Resources

- **Research Website:** [BKAI Website](https://bkai.b12sites.com/index)
- **AI Application Repository:** [BKAI AI App](https://github.com/ndtnguyen260696/bkai-ai-app)
- **Research Dataset:** [Roboflow Concrete Crack Dataset](https://universe.roboflow.com/thanh-nguyen-ws5r9/concrete-crack-dfd3i)
- **Published Article:** [Springer Nature](https://doi.org/10.1007/s41024-026-00900-z)

The BKAI interface represents a research-oriented demonstration. Its quantitative outputs should be interpreted according to the geometric assumptions and limitations described in the publication.

---

## 9. Source Code and Reproducibility

This repository accompanies the published research and contains supporting code and project resources.

The research employs a Mask R-CNN architecture with a ResNet-50-FPN backbone, developed using the PyTorch and Detectron2 ecosystem.

### Repository Access

```bash
git clone https://github.com/ndtnguyen260696/BKAI-Model-Mask-R-CNN.git
cd BKAI-Model-Mask-R-CNN
```

The availability of complete training configurations, model checkpoints, dependencies, and executable evaluation scripts should be verified from the repository contents.

The presence of research code does not by itself guarantee full reproduction of the published experiments.

For methodological details, model configurations, experimental datasets, and evaluation procedures, refer to the journal article.

---

## 10. Publication and Citation

### Published Research Article

**Nguyen, D.-T., Tran, N.-Q., & Tran, D.-H. (2026).**

An integrated mask R-CNN framework for concrete crack segmentation and skeleton-based quantification in automated infrastructure inspection.

*Journal of Building Pathology and Rehabilitation, 11*, Article 228.

**DOI:** https://doi.org/10.1007/s41024-026-00900-z

### BibTeX

If this research contributes to your academic or engineering work, please cite the published article:

```bibtex
@article{Nguyen2026IntegratedMaskRCNN,
  author  = {Nguyen, Dat-Thanh and Tran, Nhat-Quang and Tran, Duc-Hoc},
  title   = {An integrated mask R-CNN framework for concrete crack segmentation and skeleton-based quantification in automated infrastructure inspection},
  journal = {Journal of Building Pathology and Rehabilitation},
  year    = {2026},
  volume  = {11},
  pages   = {228},
  doi     = {10.1007/s41024-026-00900-z},
  url     = {https://doi.org/10.1007/s41024-026-00900-z}
}
```

---

## 11. Research Scope and Publication Rights

The repository is intended to support research visibility, technical understanding, and access to associated implementation resources.

The complete scientific methodology, experimental findings, and discussion are contained in the peer-reviewed publication.

Any reuse or redistribution of journal-formatted figures, published material, datasets, or software should comply with the applicable publisher agreements and respective licenses.

The repository's software license, where applicable, should be consulted separately from the journal publication rights.

---

## 12. Contact and Academic Affiliation

**Dat-Thanh Nguyen**

Faculty of Civil Engineering  
Ho Chi Minh City University of Technology (HCMUT)  
Vietnam National University Ho Chi Minh City  
Vietnam

**Academic Email:** ndthanh.sdh231@hcmut.edu.vn

**Research Interests:**
- Deep Learning and Computer Vision
- Concrete Crack Segmentation
- Structural Health Monitoring
- Automated Infrastructure Inspection
- AI Applications in Civil Engineering

---

<div align="center">

### Read the Full Published Research

For detailed methodology, mathematical formulations, complete experimental results, and scientific discussion:

### [An integrated mask R-CNN framework for concrete crack segmentation and skeleton-based quantification in automated infrastructure inspection](https://doi.org/10.1007/s41024-026-00900-z)

**Journal of Building Pathology and Rehabilitation · Springer Nature · 2026**

---

*Research Showcase | BKAI Mask R-CNN Framework*

</div>
