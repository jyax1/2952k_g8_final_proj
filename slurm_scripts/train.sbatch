#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=200:00:00
#SBATCH -p 3090-gcondo --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH --mem=16Gb

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e outputs/%J.err
#SBATCH -o outputs/%J.out

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yilong_song@brown.edu

module load miniconda
module load cuda/11.8.0
module load cudnn/9.1.1.17
conda activate fc

export HYDRA_FULL_ERROR=1

export AE_DIR=/users/ysong135/Documents/action_extractor
python ${AE_DIR}/train.py ${train_args}
