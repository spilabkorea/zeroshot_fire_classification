## 1. Introduction
EdgeFlame is a lightweight, zero-shot fire and smoke classification framework designed for real-time inference on resource-constrained edge devices like drones or embedded systems. Inspired by CLIP, it aligns physics-inspired 3D Continuous Wavelet Transform (CWT) scalograms with semantic prompts using contrastive learning. The model enables label-free detection in unseen environments and achieves state-of-the-art performance across diverse datasets — all while maintaining ultra-fast inference (3281 FPS, 0.30ms) with a compact 0.12 MB model.


![](Figure/arch.png?raw=true)
## 2. Requirements
- **Python 3.10**
### Python Packages
- **pytorch==2.7.1**
- **numpy==1.26.4**
- **pandas==2.2.3**
- **scikit-learn==0.24.2**
- **tqdm==4.28.1**
- **matplotlib==2.2.3**

## 3. Datasets
We utilized 3 datasets: Kaggle fire and smoke, fire and nonfire image dataset collected from Youtube, and FLAME 2 from IEEE. The downloadable links for all the datasets are given 

[Kaggle Fire and Smoke Dataset](https://drive.google.com/file/d/1L_TOG_sWp4xI9ojwe3YHu46VxmCS5xP8/view?usp=sharing) A large-scale dataset containing 23,730 images of diverse fire and smoke scenarios. It includes various contexts such as garbage burning, agricultural fires, and indoor cooking, captured under different environmental conditions (e.g., fog, night, different angles and backgrounds). It represents early to late fire stages and includes noise factors like clouds or steam, offering strong variability for robust model training and evaluation.

[Youtube Fire and NonFire Dataset](https://drive.google.com/file/d/1hka8269BDt-UTmUxmGOAy6KdwABbQK_D/view?usp=sharing) A custom dataset curated from publicly available YouTube drone footage, containing fire and non-fire scenes across different geographic regions, altitudes, and lighting conditions. This dataset simulates practical UAV-based monitoring scenarios and helps assess the model’s generalization to unconstrained, real-world visual input.

[FLAME 2 Dataset](https://ieee-dataport.org/open-access/flame-2-fire-detection-and-modeling-aerial-multi-spectral-image-dataset) A multimodal dataset collected using RGB and infrared (IR) sensors mounted on drones during actual wildfire events in California. Each sample includes paired fire and non-fire images, enabling cross-spectral analysis. It’s especially useful for evaluating real-world detection under thermal conditions and for testing fusion-based or IR-aware models.

### 4. CWT 
It is a mathematical technique that decomposes a signal into time-frequency space using wavelets—localized oscillations that can capture both short-lived high-frequency events and long-term low-frequency trends. In physics terms, CWT can be seen as tracing how energy evolves over time, much like observing a particle’s path in space-time
To convert the RGB dataset to CWT Scalogram conversion run 
```
python 3d_cwt_scalogram_conversion.py
```
### 5. Model Training
To trained the model on CWT Scalogram images run  file

```
python cnn_clip_model.py
```
