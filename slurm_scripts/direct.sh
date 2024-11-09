#!/bin/bash

cosine_similarity_loss_flag=""
if [ "$cosine_similarity_loss" = true ]; then
    cosine_similarity_loss_flag="--cosine_similarity_loss"
fi

run()
{
    jn=TRAIN_note${note}

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
    ${cosine_similarity_loss_flag}
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
demo_percentage=1.0
epoch=100
motion=""
image_plus_motion=""
idm_model_name=""
fdm_model_name=""
freeze_idm=""
freeze_fdm=""
architecture="direct_resnet_mlp"
learning_rate=0.001
val_demo_percentage=0.0
demo_percentage=1.0

resnet_layers_num=18
num_mlp_layers=3

# delta_position+gripper
action_type="delta_position+gripper"
horizon=2
batch_size=1632

cameras="frontview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=disentangled
note="delta_position+gripper|cropped_rgbd+color_mask_depth|disentangled|frontview"
train_only

cameras="frontview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=camera
note="delta_position+gripper|cropped_rgbd+color_mask_depth|camera|frontview"
train_only

cameras="frontview_image,sideview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=global
note="delta_position+gripper|cropped_rgbd+color_mask_depth|global|frontviewsideview"
train_only

cameras="agentview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=disentangled
note="delta_position+gripper|cropped_rgbd+color_mask_depth|disentangled|agentview"
train_only

cameras="agentview_image,sideagentview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=global
note="delta_position+gripper|cropped_rgbd+color_mask_depth|global|agentviewsideagentview"
train_only

cosine_similarity_loss_flag="--cosine_similarity_loss"

cameras="frontview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=disentangled
note="delta_position+gripper|cropped_rgbd+color_mask_depth|disentangled|frontview|cos"
train_only

cameras="frontview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=camera
note="delta_position+gripper|cropped_rgbd+color_mask_depth|camera|frontview|cos"
train_only

cameras="frontview_image,sideview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=global
note="delta_position+gripper|cropped_rgbd+color_mask_depth|global|frontviewsideview|cos"
train_only

cameras="agentview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=disentangled
note="delta_position+gripper|cropped_rgbd+color_mask_depth|disentangled|agentview|cos"
train_only

cameras="agentview_image,sideagentview_image"
data_modality="cropped_rgbd+color_mask_depth"
coordinate_system=global
note="delta_position+gripper|cropped_rgbd+color_mask_depth|global|agentviewsideagentview|cos"
train_only

data_modality="cropped_rgbd+color_mask"

cameras="agentview_image,sideagentview_image"
coordinate_system=global
note="delta_position+gripper|cropped_rgbd+color_mask|global|agentviewsideagentview|cos"
train_only

cameras="frontview_image"
coordinate_system=disentangled
note="delta_position+gripper|cropped_rgbd+color_mask|disentangled|frontview|cos"
train_only

cameras="frontview_image"
coordinate_system=camera
note="delta_position+gripper|cropped_rgbd+color_mask|camera|frontview|cos"
train_only

cameras="frontview_image,sideview_image"
coordinate_system=global
note="delta_position+gripper|cropped_rgbd+color_mask|global|frontviewsideview|cos"
train_only

cameras="agentview_image"
coordinate_system=disentangled
note="delta_position+gripper|cropped_rgbd+color_mask|disentangled|agentview|cos"
train_only

cameras="agentview_image,sideagentview_image"
coordinate_system=global
note="delta_position+gripper|cropped_rgbd+color_mask|global|agentviewsideagentview|cos"
train_only