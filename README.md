# Wildfire Smoke Detection with Explainable AI

This repository contains representative implementation components and
evaluation artifacts developed for intelligent
wildfire smoke detection using deep learning and explainable artificial
intelligence (XAI).

The project investigates the robustness and interpretability of object
detection models under domain shift and adverse environmental conditions,
with a focus on real-world and artificially corrupted wildfire smoke imagery.

---

## Project Overview

The main objectives of this project are:

- To evaluate YOLOv11 and Faster R-CNN architectures for wildfire smoke detection
- To analyze model robustness under synthetic corruptions (fog, low light, noise)
- To assess explanation consistency using Grad-CAM++ and D-RISE
- To provide transparent and accessible evaluation outputs


---

## Repository Structure

- `configs/` – Model and evaluation configuration templates  
- `data/` – Dataset split definitions and metadata (images must be downloaded separately)  
- `evaluation/` – COCO-style and image-level evaluation scripts  
- `notebooks/` – Exploratory data analysis and comparisons  
- `preprocessing/` – Dataset preparation and synthetic data generation scripts  
- `training/` – Training scripts and utilities for YOLOv11s and Faster R-CNN  
- `xai/` – Explainability scripts for YOLOv11s and Faster R-CNN  

---

## Installation

### Requirements

- Python 3.10+
- CUDA-enabled GPU recommended for training and inference

### Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate    # Linux / macOS
venv\Scripts\activate       # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

This section provides a high-level overview of the experimental workflow
implemented in this repository.

### Path Configuration

Most scripts rely on local file paths for accessing datasets, model weights
and output directories. Before executing any script, these paths must be aligned with the local environment and adapted to the user’s system.

### Preprocessing and Exploratory Data Analysis (EDA)

- Dataset exploration and descriptive analysis are provided in the Jupyter
  notebooks located in the `notebooks/` directory.

- Dataset splits and metadata files can be generated using the scripts
  in the `preprocessing/` directory.

- Pretrained and trained model weights should be downloaded or stored
  in the `weights/` directory prior to training and evaluation.

### Training

- YOLOv11s models are trained using `train_yolo.py`, with batch configurations
  as the example in `configs/yolov11/yolo_batch_example.csv` shows.

- Faster R-CNN models are trained using `train_faster_rcnn.py`, with example
  configuration provided in `configs/faster_rcnn/rcnn_config_example.yaml`.

### Evaluation and Explainability

- Detection performance can be evaluated using
  `eval_pyro_figlib_box.py` and `eval_figlib_image_level.py`.

- Grad-CAM++ explanations are generated using
  `run_gradcampp_faster_rcnn.py` and `run_gradcampp_yolo.py`.

- D-RISE explanations are generated using
  `run_drise_faster_rcnn.py` and `run_drise_yolo.py`.

---

### Interactive Demo

An interactive Streamlit application is provided in a separate repository
and enables qualitative inspection of model predictions and explanations. 

- Repository link: https://github.com/tdhorvathds/smoke-detection-demo

- Live demo: https://huggingface.co/spaces/tdhorvath/smoke-detection-demo