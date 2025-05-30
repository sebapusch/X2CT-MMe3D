# 🧠 X2CT-MMe3D API Setup Guide

This guide provides step-by-step instructions to run the X2CT-MMe3D API, either via Docker or manually.

> [\!WARNING]
> Synthetic CT volume generation is currently limited to CPU inference only, due to implementation constraints within the PerX2CT model. GPU support is not available at this time.

-----

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

-----

## 📥 Prerequisites

Download the required model checkpoints from [this Google Drive folder](https://www.google.com/search?q=https://drive.google.com/drive/folders/1wbhBSwKUv_Co5oI2Z8uKbEDQYTB9N5p6%3Fusp%3Dsharing) and place them in the following locations:

  * `resnet18_20250523_084333_epoch13.ckpt` → place inside `./models/checkpoints/`
  * `PerX2CT.ckpt` → place inside `./perx2ct/checkpoints/`

Make sure the final structure looks like this:

```
models/
└── checkpoints/
    └── resnet18_20250523_084333_epoch13.ckpt

perx2ct/
└── checkpoints/
    └── PerX2CT.ckpt
```

-----

## 🐳 Running the API with Docker

### 1\. Build the Docker Image

```bash
docker build -f docker/Dockerfile -t med:latest .
```

If the build fails due to incompatible architectures, try;

```bash
docker build -f docker/Dockerfile.macos -t med:latest .
```

### 2\. Run the Docker Container

```bash
docker run -p 8000:8000 med:latest
```

-----

## 🧪 Running the API Manually

### 1\. Set Up PerX2CT Environment

Navigate to the `perx2ct` directory and follow its setup instructions to install necessary dependencies.

```bash
cd perx2ct
# Follow setup steps as per perx2ct/README.md
```

### 2\. Set Up the API Conda Environment

Create a new Conda environment with Python 3.10:

```sh
conda create -n med python=3.10 -y
conda activate med
```

> Ensure you have Conda version 4.11 or higher to support Python 3.10.

### 3\. Install Project Dependencies

From the project root:

```bash
pip install -r requirements.txt
```

### 4\. Start the API

Run the following command from the project root:

```bash
python PYTHONPATH=. api/main.py \
  --port 8000 \
  --checkpoint ./models/checkpoints/resnet18_20250523_084333_epoch13.ckpt \
  --perx2ct_python_path <path_to_perx2ct_conda_python_executable> \
  --perx2ct_config_path ./perx2ct/PerX2CT/configs/PerX2CT.yaml \
  --perx2ct_model_path ./perx2ct/PerX2CT/checkpoints/PerX2CT.ckpt
```

-----

## ✅ API Access

Once running, the API will be available at:

```
http://localhost:8000
```

The documentation will be available at:

```
http://localhost:8000/docs
```

-----

## 📊 Recreating the Dataset and Training the Model

This section outlines the steps to recreate the dataset and train the X2CT-MMe3D model.

### 1\. Download the Dataset

Download and extract the [Chest X-rays - Indiana University dataset](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)

The rest of the steps will assume the x-rays are placed under `data/raw/images`
```
X2CT-MMe3D/
├── data/
│   ├── raw/
│   │   ├── indiana_reports.csv
│   │   ├── indiana_projections.csv
│   │   └── images/
│   │       └── ... (x-ray images)
├── ...
```
### 2\. (Optional) Generate New Train/Test Split

You can generate a new train/test split for the reports using the `generate_csv.py` script. If you skip this step, the existing `indiana_reports.csv` in `data/processed` will be used for training, along with `indiana_reports.train.csv` and `indiana_reports.test.csv` for the respective splits.

To run the script, navigate to the project root and execute:

```bash
python generate_csv.py
```

This script will process the raw data and create `indiana_reports.csv`, `indiana_reports.train.csv`, and `indiana_reports.test.csv` in the `data/processed` directory.

### 3\. Generate Synthetic CT Scans

Synthetic CT volumes are generated using the PerX2CT model. Follow the setup and inference instructions provided in the `perx2ct/README.md` file to set up the PerX2CT environment and generate the synthetic CT scans.

Specifically, you will use the `generate_synthetic_volumes.py` script from the project root.

Example command:

```bash
python generate_synthetic_volumes.py \
  --save_dir ./data/synthetic_cts \
  --projection_dir ./data/raw/images \
  --csv_reports_path ./data/processed/indiana_reports.csv \
  --csv_projections_path ./data/processed/indiana_projections.csv \
  --start_from 0 \
  --end_at 100 # Adjust as needed
```

  * `--save_dir`: Path to the directory where the generated synthetic CT volumes will be saved.
  * `--projection_dir`: Path to the directory containing the original X-ray projection images.
  * `--csv_reports_path`: Path to the CSV file containing the reports metadata.
  * `--csv_projections_path`: Path to the CSV file containing the projections metadata.
  * `--start_from`: (Optional) Starting index for processing reports.
  * `--end_at`: (Optional) Ending index for processing reports.

### 4\. Preprocess CT Scans

After generating the synthetic CT scans, preprocess them using the `preprocess_ct.py` script. This step applies necessary transformations to the CT volumes.

Example command:

```bash
python preprocess_ct.py \
  --save_dir ./data/processed_cts \
  --ct_dir ./data/synthetic_cts \
  --csv_reports_path ./data/processed/indiana_reports.csv \
  --csv_projections_path ./data/synthetic_cts/projections_synth_[0-100].csv # Adjust path based on your synthetic CT output
```

  * `--save_dir`: Path to the directory where the preprocessed CT volumes will be saved.
  * `--ct_dir`: Path to the directory containing the synthetic CT volumes generated in the previous step.
  * `--csv_reports_path`: Path to the CSV file containing the reports metadata.
  * `--csv_projections_path`: Path to the CSV file containing the projections metadata, including the paths to your generated synthetic CTs.

### 5\. Train the Model

Finally, train the X2CT-MMe3D model using the `train.py` script.

Example command:

```bash
python train.py \
  --reports ./data/processed/indiana_reports.train.csv \
  --projections ./data/raw/indiana_projections.csv \
  --xrays ./data/raw/images \
  --cts ./data/processed_cts \
  --batch-size 8 \
  --epochs 30 \
  --lr 1e-3 \
  --patience 4 \
  --pretrained True \
  --wandb True \
  --baseline-model False \
  --model-prefix x2ct_mme3d_model
```

  * `--reports`: Path to the CSV file containing the training reports (e.g., `indiana_reports.train.csv`).
  * `--projections`: Path to the CSV file containing the projection metadata.
  * `--xrays`: Path to the directory containing the X-ray images.
  * `--cts`: Path to the directory containing the preprocessed CT volumes.
  * `--batch-size`: Number of samples per batch.
  * `--epochs`: Number of training epochs.
  * `--lr`: Learning rate for the optimizer.
  * `--patience`: Number of epochs to wait for improvement before early stopping.
  * `--pretrained`: Use pre-trained weights for the model (default is `True`). Use `--no-pretrained` to disable.
  * `--wandb`: Enable Weights & Biases logging (default is `True`). Use `--no-wandb` to disable.
  * `--baseline-model`: Train the baseline BiplanarCheXNet model (default is `False`).
  * `--model-prefix`: Prefix for saving the trained model checkpoint files.