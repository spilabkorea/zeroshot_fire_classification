## 1. Introduction
EdgeFlame is a lightweight, zero-shot fire and smoke classification framework designed for real-time inference on resource-constrained edge devices like drones or embedded systems. Inspired by CLIP, it aligns physics-inspired 3D Continuous Wavelet Transform (CWT) scalograms with semantic prompts using contrastive learning. The model enables label-free detection in unseen environments and achieves state-of-the-art performance across diverse datasets — all while maintaining ultra-fast inference (3281 FPS, 0.30ms) with a compact 0.12 MB model.
![](Figure/arch.png?raw=true)
## 2. Requirements

Python 3.10, PyTorch 2.7.1

## 3. Datasets
We utilized 3 datasets: Kaggle fire and smoke, fire and nonfire image dataset collected from Youtube, and FLAME 2 from IEEE. The downloadable links for all the datasets are given 

[Kaggle Fire and Smoke Dataset](https://drive.google.com/file/d/1L_TOG_sWp4xI9ojwe3YHu46VxmCS5xP8/view?usp=sharing)

[Youtube Fire and NonFire Dataset](https://drive.google.com/file/d/1hka8269BDt-UTmUxmGOAy6KdwABbQK_D/view?usp=sharing)

[FLAME 2 Dataset](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset)

### 4. CWT Conversion
To convert the RGB dataset to CWT Scalogram conversion run python 3d_cwt_scalogram_conversion.py

### 5. Model Training
To trained the model on CWT Scalogram images run cnn_clip_model.py file
