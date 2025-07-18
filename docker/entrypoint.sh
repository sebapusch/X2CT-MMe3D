#!/bin/bash
set -e

cd /app/api

echo '[entrypoint] Starting server...'
conda run --no-capture-output -n py310 python main.py \
  --port 8000 \
  --checkpoint  /app/models/checkpoints/2-x2ct-20250604_215106 \
  --perx2ct_python_path /opt/conda/envs/perx2ct/bin/python \
  --perx2ct_config_path /app/perx2ct/PerX2CT/configs/PerX2CT.yaml \
  --perx2ct_model_path  /app/perx2ct/PerX2CT/checkpoints/PerX2CT.ckpt
