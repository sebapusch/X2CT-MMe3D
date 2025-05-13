#!/bin/bash
#SBATCH --job-name=gen_synth_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/gen_array_%A_%a.out
#SBATCH --error=logs/gen_array_%A_%a.err
#SBATCH --array=0-9  # 10 jobs (0 to 9)

# Load python module
module load Python/3.8.16-GCCcore-11.2.0

# ========== CONFIGURATION ==========
PROJECT_ROOT="/scratch/s5488079/X2CT-MMe3D"
ENV_PATH="$PROJECT_ROOT/envs/perx2ct"
SCRIPT="$PROJECT_ROOT/perx2ct/PerX2CT/generate_synthetic_volumes.py"

DATA_ROOT="/scratch/s5488079/data"

SAVE_DIR="$DATA_ROOT/out"
PROJECTION_DIR="$DATA_ROOT/images/images_normalized"
CSV_REPORTS="$DATA_ROOT/filtered_indiana_reports.csv"
CSV_PROJECTIONS="$DATA_ROOT/indiana_projections.csv"

TOTAL_ROWS=$(wc -l < "$CSV_REPORTS")
TOTAL_PARTS=10
PART_INDEX=$SLURM_ARRAY_TASK_ID

# Compute start and end rows for this job
CHUNK_SIZE=$(( (TOTAL_ROWS + TOTAL_PARTS - 1) / TOTAL_PARTS ))
START_FROM=$(( PART_INDEX * CHUNK_SIZE ))
END_AT=$(( START_FROM + CHUNK_SIZE - 1 ))

# Clamp end_at to total rows
if [ $END_AT -ge $TOTAL_ROWS ]; then
  END_AT=$((TOTAL_ROWS - 1))
fi
# ===================================

echo "Activating virtual environment..."
source "$ENV_PATH/bin/activate"

echo "Starting job part $PART_INDEX: rows $START_FROM to $END_AT"
cd "$PROJECT_ROOT/perx2ct/PerX2CT"

python "$SCRIPT" \
  --save_dir "$SAVE_DIR" \
  --projection_dir "$PROJECTION_DIR" \
  --csv_reports_path "$CSV_REPORTS" \
  --csv_projections_path "$CSV_PROJECTIONS" \
  --start_from "$START_FROM" \
  --end_at "$END_AT"

echo "Finished part $PART_INDEX at $(date)"

