#!/bin/bash

XTOCT="train.slurm"
BASELINE="train.baseline.slurm"

echo "Launching slurm jobs..."

for i in $(seq 1 30); do
    IX="$i" 

    echo "Submitting jobs with argument: $IX"

    sbatch "$XTOCT" "$IX"    
    sbatch "$BASELINE" "$IX" 
done

echo "All jobs submitted."
