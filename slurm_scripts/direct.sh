#!/bin/bash

run()
{
    jn=TRAIN_note${note}

    # Append additional identifiers based on conditions
    if [ -n "$motion" ]; then
        jn="${jn}_motion"
    fi

    if [ -n "$image_plus_motion" ]; then
        jn="${jn}_image_plus_motion"
    fi

    if [ -n "$idm_model_name" ]; then
        jn="${jn}_idm_${idm_model_name}"
    fi

    if [ -n "$fdm_model_name" ]; then
        jn="${jn}_fdm_${fdm_model_name}"
    fi

    if [ -n "$freeze_idm" ]; then
        jn="${jn}_freeze_idm"
    fi

    if [ -n "$freeze_fdm" ]; then
        jn="${jn}_freeze_fdm"
    fi

    jn="${jn}_${date}"

    # Define training arguments
    export train_args="
    --architecture=${architecture}
    --epoch=${epoch}
    --batch_size=${batch_size}
    --resnet_layers_num=${resnet_layers_num}
    --horizon=${horizon}
    --data_modality=${data_modality}
    --action_type=${action_type}
    --cameras=${cameras}
    --note=${note}
    --learning_rate=${learning_rate}
    --val_demo_percentage=${val_demo_percentage}
    --demo_percentage=${demo_percentage}
    --coordinate_system=${coordinate_system}
    --standardize_data
    --loss=${loss_type}
    --vMF_sample_method=${vMF_sample_method}
    --num_gpus=${num_gpus}
    "

    # Define Slurm arguments for GPU allocation and task count
    slurm_args="--gres=gpu:${num_gpus} --ntasks=${num_gpus}"

    # Define job name
    jn1=${jn}_train

    # Submit the job to Slurm
    jobID_1=$(sbatch ${slurm_args} --job-name=${jn1} --export=ALL,NUM_GPUS=${num_gpus} slurm_scripts/train.sbatch | cut -f 4 -d' ')
}

train_only()
{
    train_script=slurm_scripts/train.sbatch
    run
}

date=$(date +%m%d)

# Define your parameters in one place
num_gpus=8
batch_size=1632  # Per GPU batch size

architecture="direct_S_variational_resnet"
epoch=500
resnet_layers_num=18
horizon=2
cameras="frontview_image,sideview_image"
data_modality="cropped_rgbd+color_mask"
coordinate_system="global"
note="S_variational-lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632*8-rejection"
loss_type="cosine+mse"
learning_rate=0.001
val_demo_percentage=0.0
demo_percentage=1.0
vMF_sample_method="rejection"

# First run with rejection sampling
action_type="delta_position+gripper"
train_only

# Second run with Wood's method
vMF_sample_method="wood"
note="S_variational-lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632*8-wood"
train_only

# Third run (if needed)
architecture="direct_N_variational_resnet"
note="N_variational-lift1000-cropped_rgbd+color_mask-delta_position+gripper-frontside-cosine+mse-bs1632*8"
train_only