#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

cd ..
source ./envs/mme3d/bin/activate

IX="$1"
MODEL_BASE="/scratch/$USER/models"
DATA_BASE="/scratch/$USER/data"

echo "Starting training..."
echo "$DATA_BASE,$IX"

python train.py \
--reports=./data/processed/indiana_reports.train.csv \
--xrays=$DATA_BASE/images/images_normalized \
--cts=$DATA_BASE/ct_processed_128 \
--patience=10 \
--scheduler-patience=5 \
--model-dir=$MODEL_BASE/x2ct \
--epochs=40 \
--no-pretrained \
--model-prefix=rnd-seed-$IX \
--seed=$IX
