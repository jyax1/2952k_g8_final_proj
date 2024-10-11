#!/bin/bash

run()
{
    jn=TRAIN_${architecture}_latent_dim${latent_dim}_batch_size${batch_size}_horizon${horizon}_epoch${epoch}_demo${demo_percentage}_vps${vit_patch_size}_rln${resnet_layers_num}_opt${optimizer}_lr${learning_rate}_note${note}

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
    --epoch=${epoch}
    --batch_size=${batch_size}
    --horizon=${horizon}
    --demo_percentage=${demo_percentage}
    --resnet_layers_num=${resnet_layers_num}
    --action_type=${action_type}
    --cameras=${cameras}
    --data_modality=${data_modality}
    --note=${note}
    --learning_rate=${learning_rate}
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
epoch=100
motion=""
image_plus_motion=""
idm_model_name=""
fdm_model_name=""
freeze_idm=""
freeze_fdm=""
architecture=""
data_modality="voxel"
learning_rate=0.001

# Restricted Absolute
action_type="absolute_pose"
dataset_path="/users/ysong135/scratch/datasets/random_abs_new"

horizon=1
batch_size=128

# architecture="direct_cnn_mlp"
# resnet_layers_num=0
# note="restricted_abs"
# train_only

# architecture="direct_resnet_mlp"
# resnet_layers_num=18
# note="18_restricted_abs"
# train_only

# resnet_layers_num=34
# note="34_restricted_abs"
# train_only

# resnet_layers_num=50
# note="50_restricted_abs"
# train_only

# batch_size=64

# resnet_layers_num=101
# note="101_restricted_abs"
# train_only

# resnet_layers_num=152
# note="152_restricted_abs"
# train_only

# resnet_layers_num=200
# note="200_restricted_abs"
# train_only

# Unrestricted rel

action_type="absolute_pose"
dataset_path="/users/ysong135/scratch/datasets/random_rel_new"

horizon=2
batch_size=256

architecture="direct_resnet_mlp"
resnet_layers_num=18
note="18_unrestricted_rel"
train_only

resnet_layers_num=34
note="34_unrestricted_rel"
train_only

resnet_layers_num=50
note="50_unrestricted_rel"
train_only

batch_size=128

resnet_layers_num=101
note="101_unrestricted_rel"
train_only

resnet_layers_num=152
note="152_unrestricted_rel"
train_only

resnet_layers_num=200
note="200_unrestricted_rel"
train_only