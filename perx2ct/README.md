# PerX2CT (Modified Inference Setup)

This folder provides a lightweight, patched setup of [PerX2CT](https://github.com/dek924/PerX2CT) for running CT reconstruction inference using paired X-ray images.

It includes:
- Minimal patches to the original repo
- A clean setup script
- An inference-ready configuration
- A pretrained checkpoint (via Git LFS)

---

## 🚀 Quick Start

### 1. Clone and Set Up

Run this from the root of your main repository:

```bash
cd perx2ct
git lfs install
git lfs pull   # Make sure the checkpoint is downloaded
chmod +x setup.sh
./setup.sh
```

This will:
- Clone the original PerX2CT repo
- Apply required patches
- Copy inference script, config, and model checkpoint

After setup, the full project is ready in `./PerX2CT`.
---

### 2. Run Inference

You can now run inference on a pair of frontal/lateral X-rays:

```bash
cd PerX2CT

python inference.py \
  --config_path ./configs/PerX2CT.yaml \
  --ckpt_path   ./checkpoints/PerX2CT.ckpt \
  --save_dir    ./experiment
```

- `--config_path`: Path to the patched YAML config
- `--ckpt_path`: Path to the downloaded model checkpoint
- `--save_dir`: Output directory for reconstructed volumes (e.g. `.nii.gz`)

---

## 🧩 File Structure

```
perx2ct/
├── configs/
│   └── PerX2CT.yaml               # Cleaned config file
├── inference.py                   # Entry point for running inference
├── save_to_volume.py              # NIfTI saving utility
├── checkpoints/
│   └── PerX2CT.ckpt               # Pretrained model (via Git LFS)
├── patches/
│   ├── model.py                   # Patched NeRF logic
│   └── INREncoderZoomAxisInAlign.py
├── setup.sh                       # Setup script
└── README.md
```
