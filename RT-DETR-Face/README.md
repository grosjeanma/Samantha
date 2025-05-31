# RT-DETR-X Face Detection on WIDER Face

## Overview

This project demonstrates how to train a custom face detector using the [RT-DETR-X](https://github.com/ultralytics/ultralytics) model on the [WIDER Face](http://shuoyang1213.me/WIDERFACE/) dataset. The workflow covers dataset preparation, annotation conversion, model training, and evaluation. The final trained model (`best-rt-detr-x-face.pt`) is available in this repository.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Annotation Conversion](#annotation-conversion)
- [Training](#training)
- [Results & Evaluation](#results--evaluation)
- [File Descriptions](#file-descriptions)
- [How to Run](#how-to-run)
- [References](#references)

---

## Project Structure

```
RT-DETR-Face/
├── best-rt-detr-x-face.pt
├── train_label.txt    # Original WIDER Face train annotations
├── val_label.txt      # Original WIDER Face val annotations
├── wider.yaml         # Dataset config for Ultralytics
├── train2yolo.py      # Script: WIDER Face train to YOLO format
├── val2yolo.py        # Script: WIDER Face val to YOLO format
└── train-RT-DETR-X-Face.ipynb  # Main notebook
```

---

## Dataset

- **WIDER Face** is a challenging benchmark for face detection, containing thousands of images with faces in a wide range of poses, scales, and occlusions.
- The dataset is split into training and validation sets.
- Download links are provided in the notebook for both sets.

---

## Annotation Conversion

WIDER Face annotations are not in YOLO format by default. Two scripts are provided:

- **`train2yolo.py`**: Converts the training set annotations (`train_label.txt`) to YOLO format and organizes images/labels for Ultralytics.
- **`val2yolo.py`**: Converts the validation set annotations (`val_label.txt`) to YOLO format.

The conversion process:
- Reads the original annotation files.
- For each image, creates a `.txt` file with YOLO-format bounding boxes.
- Copies images into the correct folders.

---

## Training

- The main training workflow is in `train-RT-DETR-X-Face.ipynb`.
- **Model**: RT-DETR-X, a real-time transformer-based object detector.
- **Pretrained Weights**: Training starts from `rtdetr-x.pt` for better performance.
- **Config**: Training uses `wider.yaml` to specify dataset paths and class names.
- **Hyperparameters**: 300 epochs, image size 640, batch size 8, AMP enabled for speed.
- **Framework**: [Ultralytics](https://github.com/ultralytics/ultralytics) library.

---

## Results & Evaluation

- The model achieved **mAP@0.5 = 0.579** on the validation set.
    - **mAP@0.5**: Mean Average Precision at IoU threshold 0.5. This means the model detects faces with about 58% accuracy at a 50% overlap threshold.
- Training and validation plots are generated for loss, precision, recall, and mAP.
- The final trained model is saved as `best-rt-detr-x-face.pt`.

