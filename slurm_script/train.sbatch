#!/bin/bash
#SBATCH -N 1
#SBATCH -n 60
#SBATCH --time=100:00:00
#SBATCH -p 3090-gcondo --gres=gpu:4 --gres-flags=enforce-binding
#SBATCH --mem=64Gb

# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -e outputs/%J.err
#SBATCH -o outputs/%J.out

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=yilong_song@brown.edu
#SBATCH --exclude=gpu2264,gpu2602,gpu2508,gpu2251,gpu2510,gpu2604

module load miniconda
module load cuda/11.8.0
module load cudnn/8.7.0.84
conda activate avdc

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

export AE_DIR=/users/ysong135/Documents/
accelerate launch ${VIDEO_DIR}/flowdiffusion/pretrain.py ${train_args}
