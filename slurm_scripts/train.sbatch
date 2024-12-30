#!/bin/bash
#SBATCH -N 1
#SBATCH --time=200:00:00
#SBATCH -p 3090-gcondo --gres-flags=enforce-binding
#SBATCH --mem=512Gb

# Use '%J' for job ID
#SBATCH -e outputs/%J.err
#SBATCH -o outputs/%J.out

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yilong_song@brown.edu

# Load necessary modules and activate environment
module load miniconda
module load cuda/11.8.0
module load cudnn/9.1.1.17
conda activate fc

# Enable full error messages
export HYDRA_FULL_ERROR=1

# Define the directory containing the training script
export AE_DIR=/users/ysong135/Documents/action_extractor/action_extractor

# Launch training with Accelerate using the specified number of GPUs
accelerate launch --num_processes=8 ${AE_DIR}/train.py ${train_args}
