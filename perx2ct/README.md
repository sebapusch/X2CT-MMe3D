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
├── scripts/
│   ├── run_generate_dataset.slurm  # SLURM script for batch volume generation
│   └── setup.sh              # Setup script for environment
└── README.md                 # You are here
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

## ⚙️ Setup (`scripts/setup.sh`)

Before using the repository:

```bash
chmod +x scripts/setup.sh # Make executable
./scripts/setup.sh        # Installs patches, configuration and checkpoint
```

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
