#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH --time=150:00:00
#SBATCH -p 3090-gcondo --gres=gpu:1 --gres-flags=enforce-binding
#SBATCH --mem=8Gb

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e outputs/%J.err
#SBATCH -o outputs/%J.out

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yilong_song@brown.edu

module load miniconda
module load cuda/11.8.0
module load cudnn/9.1.1.17
conda activate fc

#export PYTHONPATH=$PYTHONPATH:/users/mjia11/workspace/videopo/mimicgen_envs/mimicgen_envs/robosuite
#export PYTHONPATH=$PYTHONPATH:/users/mjia11/workspace/videopo/mimicgen_envs/mimicgen_envs/robomimic
#export PYTHONPATH=$PYTHONPATH:/users/mjia11/workspace/videopo/3D-CNN-PyTorch
export HYDRA_FULL_ERROR=1
#export OMP_NUM_THREADS=1
#export USE_SIMPLE_THREADED_LEVEL3=1

# set mujoco variables for robosuite
#export MUJOCO_GL="osmesa"
#export MJLIB_PATH=$HOME/.mujoco/mujoco210/bin/libmujoco210.so
#export MJKEY_PATH=$HOME/.mujoco/mujoco210/mjkey.txt
#export MUJOCO_PY_MJPRO_PATH=$HOME/.mujoco/mujoco210/
#export MUJOCO_PY_MJKEY_PATH=$HOME/.mujoco/mujoco210/mjkey.txt

export AE_DIR=/users/ysong135/Documents/action_extractor
python ${AE_DIR}/train.py ${train_args}
