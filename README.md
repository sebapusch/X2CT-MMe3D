# 🧠 X2CT-MMe3D API Setup Guide

This guide provides step-by-step instructions to run the X2CT-MMe3D API, either via Docker or manually.

---

## 📁 Relevant files structure

```
X2CT-MMe3D/
├── api/                 # FastAPI-based REST API
│   └── main.py          # API entry point
├── docker/              # Docker-related files
│   └── Dockerfile       # Dockerfile for building the image
├── models/
│   └── checkpoints/     # Directory for model checkpoints
├── perx2ct/             # PerX2CT model and dependencies
└── requirements.txt     # Python dependencies
```

---

## 📥 Prerequisites

Before proceeding, ensure you have the necessary model files:

1. **Download the model checkpoint** from [this Google Drive folder](https://drive.google.com/drive/folders/1wbhBSwKUv_Co5oI2Z8uKbEDQYTB9N5p6?usp=sharing).
2. **Place the downloaded checkpoint** in the `models/checkpoints/` directory.

---

## 🐳 Running the API with Docker

### 1. Build the Docker Image

```bash
docker build -f docker/Dockerfile -t med:latest .
```

### 2. Run the Docker Container

```bash
docker run -p 8000:8000 med:latest
```

The API will be accessible at `http://localhost:8000`.

---

## 🧪 Running the API Manually

### 1. Set Up PerX2CT Environment

Navigate to the `perx2ct` directory and follow its setup instructions to install necessary dependencies.

```bash
cd perx2ct
# Follow setup steps as per perx2ct/README.md
```

### 1. Set Up the API Conda Environment

Create a new Conda environment with Python 3.10:
```sh
conda create -n med python=3.10 -y
conda activate med
```

> Ensure you have Conda version 4.11 or higher to support Python 3.10.

### 2. Install Project Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 3. Start the API

Run the following command from the project root:

```bash
python api/main.py \
  --port 8000 \
  --checkpoint ./models/checkpoints/resnet18_20250523_084333_epoch13 \
  --perx2ct_python_path <path_to_perx2ct_conda_python_executable> \
  --perx2ct_config_path ./perx2ct/PerX2CT/configs/PerX2CT.yaml \
  --perx2ct_model_path ./perx2ct/PerX2CT/checkpoints/PerX2CT.ckpt
```

---

## ✅ API Access

Once running, the API will be available at:

```
http://localhost:8000
```
