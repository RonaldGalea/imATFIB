# Introduction

- This repository contains code for training, validating and testing medical image segmentation deep learning models.
- Pretrained models available at: [`models`](https://drive.google.com/drive/folders/1zAhlzh7EiQyU3chBUDn4GzzYxaE0ZXYB?usp=sharing)
- Methods and models tested on the ACDC challenge [1].

# Usage

- There are two models implemented in this repo: Unet [2] and DeepLab [3], plus variations and ensemble inference
- **ROI Localization:** A method to first extract an ROI around the labelled area of the image (using the output of a standard model), and subsequently feed the ROI to another model (ROI model) to perform the final segmentation.
- **Training and setup** for various models is presented in the following **Tutorial notebook** - [`tutorial_training_validating.ipynb`](https://github.com/RonaldGalea/imATFIB/blob/master/tutorial_training_validating.ipynb)
- **Double segmentation, ensembling and testing** is presented in the following **Tutorial notebook** - [`tutorial_doubleSeg_ensembling_testing.ipynb`](https://github.com/RonaldGalea/imATFIB/blob/master/tutorial_doubleSeg_ensembling_testing.ipynb)
- **Visualizing results** is presented in **Tutorial notebook** - [`tutorial_visualization.ipynb`](https://github.com/RonaldGalea/imATFIB/blob/master/tutorial_visualization.ipynb)

# Results

- Results obtained on an ACDC validation set are:

Model | Dice RV | Dice LV | Dice Myo | Dice mean
--- |--- |--- |--- |---
DeepLab    |90.50| 94.50 | 89.10  | 91.37
Unet     |89.32| 95.07 | 89.47  | 91.29
Double DeepLab    |90.58| 94.72 | 89.53  | 91.61
Double Unet     |90.65  | 95.02 | 89.94  | 91.87
DeepLab + Unet    |90.48| 95.12 | 89.89  | 91.83
Double DeepLab + Unet     |90.88| 94.98 | 90.11  | 91.99
All Ensemble    |91.26| 95.39 | 90.49  | 92.38

- Results obtained on the ACDC test set are:

Model | Dice RV | Dice LV | Dice Myo | Dice mean | Hauss RV | Hauss LV | Hauss Myo
--- |--- |--- |--- |--- |--- |--- |---
DeepLab    |90.22| 93.84 | 88.81  | 90.96  | 24.27 | 11.93 | 14.46 |
All Ensemble     |91.48| 93.88 | 90.26  | 91.87  | 13.21 | 10.99 | 12.08 |

**Note!**
All Ensemble* = Ensemble of Standard DeepLab, Unet and Double Segmentation DeepLab, Unet

- Results obtained on the validation set are consistent with the official test set, confirming that both double segmentation and ensembling (especially of distinct models, such as Unet and DeepLab) boost performance.

# Image Samples

![Sample one model](https://github.com/RonaldGalea/imATFIB/blob/master/sample.png)
