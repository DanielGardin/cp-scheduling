#!/bin/bash
#SBATCH --job-name=pl_estimation
#SBATCH --output=logs/slurm/%A_%a.out
#SBATCH --error=logs/slurm/%A_%a.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
# SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --array=0-39

PDRS=("WSPT" "WMDD" "COverT" "ATC")

# Determine pdr and seed from array index
PDR_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
PDR=${PDRS[$PDR_INDEX]}

source ~/miniconda3/bin/activate
conda activate scheduling

python run/pl_parameter_estimation.py \
    --pdr $PDR \
    --lr 1e-4 \
    --log_dir "logs/parameter_random/$PDR" \
    --random_start \
    -q
