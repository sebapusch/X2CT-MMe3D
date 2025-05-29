# PerX2CT Wrapper & Synthetic CT Generation

This repository provides a wrapper and CLI utility for generating synthetic 3D CT volumes from paired biplanar X-ray images using the [PerX2CT](https://arxiv.org/abs/2303.05297) model. It is a crucial preprocessing component of the multimodal classification pipeline for enhanced pulmonary diagnosis.

## 📁 Project Structure

```
.
├── checkpoints/              # Contains PerX2CT model checkpoint
│   └── PerX2CT.ckpt
├── configs/                  # YAML configuration for the model
│   └── PerX2CT.yaml
├── patches/                  # Custom patches for the PerX2CT model
│   ├── INREncoderZoomAxisInAlign.py
│   └── model.py
├── generate_synthetic_volumes.py  # Script to create synthetic CTs
├── inference.py              # Inference wrapper class
├── save_to_volume.py         # Utilities to save CT volume in NIfTI/HDF5/NPY
├── listener.py               # Listener for inter-process generation
├── scripts/
│   ├── run_generate_dataset.slurm  # SLURM script for batch volume generation
│   └── setup.sh              # Setup script for environment
└── README.md                 # You are here
```

---

## ⚙️ Setup (`scripts/setup.sh`)

Before using the repository:

```bash
cd scripts
chmod +x setup.sh # Make executable
./setup.sh        # Installs patches, configuration and checkpoint
```

Inside `PerX2CT` root, Create a conda environment, activate it, and install the required packages:
```bash
conda create -n perx2ct python=3.8
conda activate perx2ct
pip install --upgrade pip
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirement.txt
```

Here's an additional section you can append to the README under a new heading `## 📡 Inter-Process Listener (listener.py)`, written in the same style as the existing documentation:

---

## 📡 Inter-Process Listener (`listener.py`)

This module provides a lightweight server for **inter-process communication (IPC)** to run PerX2CT inference via a socket-based interface. It receives X-ray image data as NumPy arrays, reconstructs a 3D CT volume using the `Inference` class, and sends the result back.

### Key Features:

* Socket-based array exchange using Python’s `multiprocessing.connection`
* Handles paired frontal/lateral image tensors
* Automatically shuts down on `EOFError`
* Returns an empty array if inference fails

### Usage

Start the listener on a given port (default: `6000`):

```bash
python listener.py \
  --checkpoint ./perx2ct/PerX2CT/checkpoints/PerX2CT.ckpt \
  --config ./perx2ct/PerX2CT/configs/PerX2CT.yaml \
  --port 6000
```

### Protocol

1. The listener waits for a connection.
2. It receives two NumPy arrays: the frontal and lateral projections.
3. It reconstructs a volume using the `Inference` model.
4. It sends back the result as a serialized NumPy array.

### Example Client

To interact with the listener, use the `ArrayTransferConnection` protocol:

```python
from multiprocessing.connection import Client
from listener import ArrayTransferConnection

conn = Client(('localhost', 6000))
atc = ArrayTransferConnection(conn)

atc.send(frontal_array)
atc.send(lateral_array)

volume = atc.receive()
```

---

## 🔍 Inference Wrapper (`inference.py`)

The `Inference` class handles PerX2CT loading and slice-wise reconstruction of 3D CT volumes from a pair of X-ray images.

### Key Features:
- Loads model and config via patched PerX2CT framework
- Handles preprocessing of frontal (PA) and lateral X-rays
- Iteratively reconstructs 128 axial slices
- Supports GPU or CPU inference

### Usage (Python):

```python
from inference import Inference
model = Inference(config_path='configs/PerX2CT.yaml',
                  ckpt_path='checkpoints/PerX2CT.ckpt',
                  dev=torch.device('cuda:0'))

volume = model('path/to/frontal.png', 'path/to/lateral.png')  # torch.Tensor [128, 128, 128]
```

---

## 🧪 Script: `generate_synthetic_volumes.py`

This script generates synthetic CT volumes for a dataset of paired X-ray projections and writes them to disk.

### Inputs:
- `--csv_reports_path`: Path to CSV with report metadata (includes `uid`)
- `--csv_projections_path`: Path to CSV with filenames and projections (must include `uid`, `projection`, `filename`)
- `--projection_dir`: Directory containing raw X-ray images
- `--save_dir`: Output directory for saving volumes and updated projections CSV

### Output:
- 3D CT volumes saved as `.h5` files named `{uid}_ct_synthetic.h5`
- New `projections_synth.csv` CSV file mapping each `uid` to the synthetic volume

### CLI Example:

```bash
python generate_synthetic_volumes.py \
  --csv_reports_path ./data/reports.csv \
  --csv_projections_path ./data/projections.csv \
  --projection_dir ./data/images/ \
  --save_dir ./outputs/
```

You can optionally use `--start_from` and `--end_at` to generate a subset.

---

## 💾 Volume Saving Options (`save_to_volume.py`)

Supports saving the output tensor as:
- `.nii` or `.nii.gz` (NIfTI format)
- `.h5` (default, HDF5 format)
- `.npy` (NumPy array)

Default used in the script is `.h5`. To change this, modify the `CT_EXTENSION` variable in `generate_synthetic_volumes.py`.

---

## 🚀 SLURM Batch Script (`scripts/run_generate_dataset.slurm`)

This script enables distributed generation of synthetic CT volumes on SLURM-managed HPC systems.

### Key SLURM Directives:
- Requests 1 GPU and 4 CPUs per task
- Uses SLURM job arrays for parallel generation
- Loads and activates the Python environment
- Runs `generate_synthetic_volumes.py` on a data chunk

### Usage:

Submit the job array like this:

```bash
sbatch scripts/run_generate_dataset.slurm
```

Make sure to adjust the paths to:
- Python environment (`ENV_PATH`)
- Script location
- CSV files and directories (`CSV_REPORTS`, `CSV_PROJECTIONS`, `PROJECTION_DIR`, `SAVE_DIR`)
- Array partitioning logic (`TOTAL_PARTS`, `PART_INDEX`) to fit your dataset

---

## 📚 References

- **PerX2CT**: Kyung et al., ICASSP 2023. [arXiv:2303.05297](https://arxiv.org/abs/2303.05297)
- **Med3D**: Chen et al., 2019. [arXiv:1904.00625](https://arxiv.org/abs/1904.00625)
