#!/bin/bash

run()
{
    jn=TRAIN_${architecture}_latent_dim${latent_dim}_batch_size${batch_size}_horizon${horizon}_epoch${epoch}_demo${demo_percentage}_vps${vit_patch_size}_rln${resnet_layers_num}

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

    export train_args="
    --architecture=${architecture}
    --datasets_path=${dataset_path}
    --latent_dim=${latent_dim}
    --epoch=${epoch}
    --batch_size=${batch_size}
    --horizon=${horizon}
    --demo_percentage=${demo_percentage}
    --vit_patch_size=${vit_patch_size}
    --resnet_layers_num=${resnet_layers_num}
    $motion
    $image_plus_motion
    --idm_model_name=${idm_model_name}
    --fdm_model_name=${fdm_model_name}
    $freeze_idm
    $freeze_fdm
    "
    slurm_args=""

    jn1=${jn}_train
    jobID_1=$(sbatch ${slurm_args} --job-name=${jn1} --export=ALL ${train_script} | cut -f 4 -d' ')
}

train_only()
{
    train_script=slurm_scripts/train.sbatch
    run
}

date=$(date +%m%d)

# Parameters for the jobs
demo_percentage=.9
dataset_path="/users/ysong135/scratch/datasets/"
epoch=100
batch_size=1024
horizon=2
motion=""
image_plus_motion=""
idm_model_name="idm_model.pth"
fdm_model_name="fdm_model.pth"
freeze_idm=""
freeze_fdm=""
vit_patch_size=2
resnet_layers_num=18

# Job 1: direct_cnn_mlp
architecture="direct_cnn_mlp"
latent_dim=32
train_only

# Job 2: direct_cnn_vit
architecture="direct_cnn_vit"
latent_dim=32
vit_patch_size=8
train_only

# Job 3, 4: direct_resnet_mlp
architecture="direct_resnet_mlp"
train_only

resnet_layers_num=50
train_only

# Job 5: latent_cnn_unet
architecture="latent_cnn_unet"
latent_dim=32
train_only