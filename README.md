# TerLiDAR Dataset & Our work in LoRA for PointNet++ in Airborne LiDAR Semantic Segmentation

> Parameter‑efficient fine‑tuning (LoRA) for 3D point cloud semantic segmentation with PointNet++, evaluated on airborne LiDAR datasets.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](#license)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-red.svg)]()

## ⬇️ TerLiDAR Dataset Download

Please respond to the form to receive the link to download the dataset:
[Link to form](https://docs.google.com/forms/d/e/1FAIpQLSfegWcIX8sTO9ordN91-pYJaLgRlv5eIx3mUzf8BNONHbjWKw/viewform?usp=header)

## 💡 Overview
This repository contains code and assets accompanying the paper:
> **Efficient Task and Domain Adaptation in ALS Semantic Segmentation via LoRA for PointNet++**  
Semantic segmentation of airborne LiDAR point clouds enables a broad range of urban and environmental applications. However, domain shifts between training and operational data, as well as the frequent emergence of new semantic classes, pose significant challenges for deploying deep learning models effectively. In this work, we explore the integration of Low-Rank Adaptation (LoRA), a parameter-efficient fine-tuning technique, into the PointNet++ architecture to address these challenges. We evaluate LoRA in two realistic scenarios: domain adaptation and incremental learning with novel classes, using subsets of large-scale LiDAR datasets under constrained labeled data settings. Our experiments show that LoRA outperforms traditional full fine-tuning, achieving notable gains (+3.1 IoU for specific classes and +0.3 mIoU on TerLiDAR, +2.7 mIoU on DALES), while exhibiting greater resistance to catastrophic forgetting and improved generalization, particularly for underrepresented classes. Furthermore, LoRA exceeds baseline accuracy with substantially fewer trainable parameters (73.4% reduction), highlighting its suitability for resource-constrained deployment scenarios. We also present TerLiDAR, a publicly available annotated airborne LiDAR dataset covering 51.4 km^2
 along the Ter River in Catalonia, Spain. It contributes to increasing the diversity of semantic segmentation benchmarks and advancing 3D scene understanding in remote sensing.

📄 **Paper link**: [Read the paper on ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2667393226000050?via%3Dihub)


<p align="center">
  <img src="figs/PN++Lora.png" alt="Model teaser" width="100%">
</p>
<p align="left">
  <em>
PN++ architecture. Set Abstraction (SA) layers sample input points, group them, and apply PointNet to obtain high-dimensional representations. Feature Propagation (FP) layers upsample points and propagate features back to the original resolution.
 </em>
</p>

### LoRA applied to PointNet++
<p align="center">
  <img src="figs/diagramaLora.png" alt="Model teaser" width="50%">
</p>
<p align="left">
  <em>
    Orange boxes indicate trainable modules.  
    The input point cloud is represented as x ∈ ℝ<sup>N×D</sup>, where N is the number of input points and D is the number of input features.  
    Each layer processes local neighborhoods, where N<sub>l</sub> is the number of sampled points at each level l, K is the number of neighboring points in the local region, and D<sub>in</sub> and D<sub>out</sub> are the input and output feature dimensions.  
    The output of the network is a per-point semantic prediction y ∈ ℝ<sup>N×C</sup>, where C is the number of semantic classes.
  </em>
</p>

## ✨ Key Contributions
- **LoRA-enabled PointNet++**: Integration of Low-Rank Adaptation modules into the PointNet++ architecture to enable parameter-efficient fine-tuning.
- **Task & domain adaptation**: Evaluation across different airborne LiDAR datasets and in class extension scenarios, demonstrating LoRA's flexibility with minimal parameter overhead.
- **TerLiDAR dataset**: We release **TerLiDAR**, an open, annotated airborne LiDAR dataset covering 51.4 km² of mixed urban and forested landscapes along the Ter River in Catalonia, Spain.  


### TerLiDAR Dataset
We present TerLiDAR, a fully open and annotated airborne LiDAR dataset that covers 51.4 km² of urban and forested areas along the Ter River in Catalonia, Spain. The data were acquired in July 2021 using an ALS system mounted on a georeferenced aircraft operated by ICGC. The dataset comprises 692 million colorized 3D points, each annotated with one of the semantic classes listed in the paper.

<p align="center">
  <img src="figs/overview.PNG" alt="TerLiDAR coverage area" width="100%">
</p>
<p align="center">
  <em>
    Geographic coverage of the TerLiDAR dataset along the Ter River in Catalonia, Spain.  
  </em>
</p>

<p align="center">
  <img src="figs/categories_table.png" alt="Model teaser" width="100%">
</p>

The classes are described as follows:
1. **Default** – Points that could not be classified during the classification process.  
2. **Ground** – Points belonging to the terrain.  
3. **Low vegetation** – Points corresponding to low vegetation such as shrubs, crops, and the lower parts of trees (0.3 m < height ≤ 2.0 m).  
4. **Medium vegetation** – Points belonging to medium vegetation, which may include taller shrubs, crops, and parts of tree canopies (2.0 m < height ≤ 3.0 m).  
5. **High vegetation** – Points corresponding to high vegetation, primarily points within tree canopies (height > 3.0 m).  
6. **Buildings** – Points generally classified on building rooftops.  
7. **Low Points** – Points with negative height relative to the ground, usually sensor noise.  
8. **Ground key points** – Simplified points previously classified as ground (Class 2), used for constructing the Digital Terrain Model.  
11. **Air points** – Points detected above the terrain, often spurious returns.  
13. **Other ground points** – Points near the ground, such as those pertaining to grass, that could not be classified as ground.  
14. **Power lines** – Points representing electric power lines.  
15. **Transmission tower** – Points representing electrical towers.  
16. **Façade** – Points belonging mostly to building façades.  
17. **Above buildings** – Points located above buildings, such as chimneys, solar panels, or awnings.  
18. **Other towers** – Points corresponding to towers not classified as transmission towers (e.g. observation towers). 
135. **Noise** – Points identified as noise produced by the sensor.  


<p align="center">
  <img src="figs/RGB.png" alt="TerLiDAR example point clouds" width="100%">
  <img src="figs/classification_legend.PNG" alt="TerLiDAR example point clouds" width="100%">
</p>
<p align="center">
  <em>
    Visualization of a LAS tile using RGB values and corresponding semantic classes.
  </em>
</p>


## 📦Code and Environment
[Code here](https://github.com/marionacaros/terlidar)

Environment:
- Python 3.9+
- PyTorch 1.12+ with CUDA 11.x (recommended)

Quick setup:
```bash
conda create -n lora-pn2 python=3.10 -y
conda activate lora-pn2
pip install -r requirements.txt
