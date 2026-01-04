# BLPNet: Boosting Localization Perception Network for Foveal Avascular Zone Segmentation

This repository contains the official implementation of the paper **"BLPNet: Boosting localization perception network for foveal avascular zone segmentation"**.

The paper is available at: https://doi.org/10.1016/j.patcog.2025.112790

## Baseline

This project uses **Joint-Seg** as a baseline: [https://github.com/MLMIP/Joint-Seg/tree/main](https://github.com/MLMIP/Joint-Seg/tree/main)

## Requirements

The code is implemented using PyTorch. Please ensure you have the following dependencies installed:

- python >= 3.6
- pytorch
- torchvision
- numpy
- opencv-python
- matplotlib
- ml_collections
- tqdm

## Usage

### 1. Data Preparation

Configure your dataset path in `configs.py`. You need to set the `data_dir` variable and choose the corresponding `data_class`.

Supported datasets:
- **OCTA-500 (6mm)**: Set `data_class = 'OCTA_6mm'`
- **OCTA-500 (3mm)**: Set `data_class = 'OCTA_3mm'`
- **ROSE**: Set `data_class = 'rose'`

Example in `configs.py`:
```python
data_class = 'OCTA_6mm'
if data_class == 'OCTA_6mm':
    configs.data_dir = '/path/to/your/OCTA_6mm'
# ...
```

### 2. Training

To train the model, ensure `configs.mode` is set to `'TRAIN'` in `configs.py`, then run:

```bash
python main.py
```

You can adjust other hyperparameters such as `batch_size`, `epochs`, `learning_rate` in `configs.py`.

### 3. Testing

To evaluate the model, set `configs.mode` to `'TEST'` in `configs.py` and run:

```bash
python main.py
```

## Citation

If you find this repository helpful, please consider citing our paper:

```bibtex
@article{ZHAN2026112790,
title = {BLPNet: Boosting localization perception network for foveal avascular zone segmentation},
journal = {Pattern Recognition},
volume = {173},
pages = {112790},
year = {2026},
issn = {0031-3203},
doi = {https://doi.org/10.1016/j.patcog.2025.112790},
url = {https://www.sciencedirect.com/science/article/pii/S0031320325014530},
author = {Zihang Zhan and Xinpeng Zhang and Meng Zhao and Yao Zhang and Wei Zhou}
}
```
