> [!TIP]
> This project builds on [PerX2CT (arXiv:2303.05297)](https://arxiv.org/abs/2303.05297), available on GitHub at [github.com/dek924/PerX2CT](https://github.com/dek924/PerX2CT).

# 🧠 X2CT-MMe3D API Setup & Training Guide

> [!NOTE]
> Synthetic CT volume generation at inference currently supports **CPU inference only** due to PerX2CT implementation constraints. GPU acceleration is not available yet.

---

## 📋 Table of Contents

1. [Project Structure Overview](#-project-structure-overview)
2. [Prerequisites & Model Checkpoints](#-prerequisites--model-checkpoints)
3. [Running the API with Docker](#-running-the-api-with-docker)
4. [Running the API manually](#-running-the-api-manually)
5. [Accessing the API](#-accessing-the-api)
6. [Dataset Preparation](#-dataset-preparation)
   * [Downloading the Dataset](#1-download-the-dataset)
   * [Generating Train/Test Splits](#2-optional-generate-traintest-splits)
   * [Generating Synthetic CT Scans](#3-generate-synthetic-ct-scans)
   * [Preprocessing Synthetic CTs](#4-preprocess-synthetic-ct-scans)
7. [Model Training](#-model-training)
8. [Running the Streamlit Demo](#%EF%B8%8F-running-the-streamlit-demo)

---

## 📁 Project Structure Overview

```
X2CT-MMe3D/
├── api/                      # FastAPI REST API code
│   └── main.py               # API entry point
├── data/                     # Dataset and processed data folders
│   ├── processed/            # Processed CSV files and preprocessed data
│   ├── raw/                  # Raw datasets and images
│   └── synthetic_cts/        # Generated synthetic CT volumes
├── docker/                   # Docker build files
│   ├── Dockerfile            # Linux Dockerfile
│   └── Dockerfile.macos      # macOS Dockerfile
├── models/                   # Model checkpoints for X2CT-MMe3D
│   └── checkpoints/
├── perx2ct/                  # PerX2CT model code and dependencies
├── scripts/                  # Utility and helper scripts
│   ├── generate_csv.py       # Generate train/test splits CSV
│   └── preprocess_ct.py      # Preprocess synthetic CTs
├── requirements.txt          # Python dependencies
└── train.py                  # Training script for X2CT-MMe3D
```

---

## 📥 Prerequisites & Model Checkpoints

1. Download the required model checkpoints from [Google Drive](https://drive.google.com/drive/folders/1wbhBSwKUv_Co5oI2Z8uKbEDQYTB9N5p6?usp=sharing).

2. Place the files as follows:

```
models/
└── checkpoints/
    └── 2-x2ct-20250604_215106

perx2ct/
└── checkpoints/
    └── PerX2CT.ckpt
```

---

## 🐳 Running the API with Docker

### 1. Build the Docker Image

```bash
docker build -f docker/Dockerfile -t med:latest .
```

> [!TIP]
> If building on macOS or if architecture errors occur, try building with the `Dockerfile.macos` file instead

### 2. Run the Docker Container

```bash
docker run -p 8000:8000 med:latest
```

---

## 🧪 Running the API Manually

### 1. Set Up PerX2CT Environment

```bash
cd perx2ct
# Follow instructions in perx2ct/README.md to install dependencies
```

### 2. Create and Activate the Conda Environment for API

```bash
conda create -n med python=3.10 -y
conda activate med
```

### 3. Install Project Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 4. Start the API Server

```bash
python PYTHONPATH=. api/main.py \
  --port 8000 \
  --checkpoint ./models/checkpoints/resnet18_20250523_084333_epoch13.ckpt \
  --perx2ct_python_path <path_to_perx2ct_conda_python_executable> \
  --perx2ct_config_path ./perx2ct/PerX2CT/configs/PerX2CT.yaml \
  --perx2ct_model_path ./perx2ct/PerX2CT/checkpoints/PerX2CT.ckpt
```

Replace `<path_to_perx2ct_conda_python_executable>` with the absolute path to the Python executable inside your PerX2CT conda environment.

---

## ✅ Accessing the API

* API Root URL:
  `http://localhost:8000`

* Interactive API Documentation (Swagger UI):
  `http://localhost:8000/docs`

---

## 📊 Dataset Preparation

### 1. Download the Dataset

Download the [Chest X-rays - Indiana University dataset](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) and place X-ray images in:

```
X2CT-MMe3D/
├── data/
│   ├── raw/
│   │   ├── indiana_reports.csv
│   │   ├── indiana_projections.csv
│   │   └── images/
│   │       └── ... (x-ray images)
```

---

### 2. (Optional) Generate Train/Test Splits

Run the CSV generator script from the project root to create new splits:

```bash
python generate_csv.py
```

This creates:

* `data/processed/indiana_reports.csv`
* `data/processed/indiana_reports.train.csv`
* `data/processed/indiana_reports.test.csv`

If skipped, existing CSVs in `data/processed` will be used.

---

### 3. Generate Synthetic CT Scans

Set up PerX2CT environment as per `perx2ct/README.md` and run:

```bash
python generate_synthetic_volumes.py \
  --save_dir ./data/synthetic_cts \
  --projection_dir ./data/raw/images \
  --csv_reports_path ./data/processed/indiana_reports.csv \
  --csv_projections_path ./data/processed/indiana_projections.csv \
```

See `perx2ct/README.md` for more detailed instructions.

---

### 4. Preprocess Synthetic CT Scans

Run preprocessing to prepare CT scans for training:

```bash
python preprocess_ct.py \
  --save_dir ./data/processed_cts \
  --ct_dir ./data/synthetic_cts \
  --csv_reports_path ./data/processed/indiana_reports.csv \
  --csv_projections_path ./data/synthetic_cts/projections_synth.csv
```

---

## 🚀 Model Training

Train the X2CT-MMe3D model using the prepared data:

```bash
python train.py \
  --reports ./data/processed/indiana_reports.train.csv \
  --projections ./data/processed/indiana_projections.csv \
  --xrays ./data/raw/images \
  --cts ./data/processed_cts \
  --model-dir ./models/checkpoints \
  --batch-size 8 \
  --epochs 30 \
  --lr 1e-3 \
  --weight-decay 1e-3 \
  --test-size 0.1 \
  --patience 10 \
  --scheduler-patience 8 \
  --pretrained \
  --wandb \
  --baseline False \
  --model-prefix x2ct_mme3d_model
```

### Training Parameters

| Argument               | Description                                         | Default / Notes           |
|------------------------|-----------------------------------------------------|----------------------------|
| `--reports`            | CSV with diagnosis labels                          | Required                   |
| `--projections`        | Projections metadata CSV                           | Required                   |
| `--xrays`              | Directory with X-ray images                        | Required                   |
| `--cts`                | Directory with preprocessed CT volumes             | Required                   |
| `--model-dir`          | Output directory for saving model checkpoints      | Required                   |
| `--batch-size`         | Training batch size                                | 8                          |
| `--epochs`             | Total number of epochs                             | 30                         |
| `--lr`                 | Learning rate                                      | 0.001                      |
| `--weight-decay`       | Weight decay for optimizer                         | 0.001                      |
| `--test-size`          | Validation data split ratio                        | 0.1                        |
| `--patience`           | Early stopping patience (in epochs)                | 10                         |
| `--scheduler-patience` | Learning rate scheduler patience                   | 8                          |
| `--pretrained`         | Whether to load pretrained model weights           | Enabled by default         |
| `--wandb`              | Enable Weights & Biases logging                    | Enabled by default         |
| `--baseline`           | Train the BiplanarCheXNet baseline model           | `False` = Use X2CT-MMe3D   |
| `--model-prefix`       | Filename prefix for saving model checkpoints       | `x2ct_mme3d_model`         |
| `--seed`               | Random seed for reproducibility                    | Optional                   |

### Final Model Runs on HPC

For training the final models used in the api and baseline comparison, we relied on the exact hyperparameters and environment specifications defined in the following SLURM job scripts:

- `hpc/train.slurm` – for the **X2CT-MMe3D** model  
- `hpc/train.baseline.slurm` – for the **BiplanarCheXNet baseline**
---

## 🎛️ Running the Streamlit Demo

Launch an interactive frontend to upload X-rays, run inference, and visualize 3D slices of the predicted CT scans.

### 1. Start the FastAPI Server First

Make sure the API is running and accessible at `http://localhost:8000`. You can do this either via Docker or manual setup.

### 2. Start the Streamlit App

```bash
streamlit run app.py
```

### 3. Usage

1. Upload **frontal** and **lateral** chest X-ray images.
2. Click **Submit**.
3. View the **diagnosis** and scroll through axial, sagittal, and coronal slices of the synthetic CT (raw + Grad-CAM version).

> [!NOTE]
> The demo assumes synthetic volumes are of shape `128×128×128` stored as `.npy` files.

---
