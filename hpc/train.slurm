#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err

cd ..
source ./envs/mme3d/bin/activate

echo "Starting training..."

python train.py \
--reports=./data/processed/indiana_reports.train.csv \
--xrays=../data/images/images_normalized \
--cts=../data/ct_processed \
--model-prefix=sep-2d-branch