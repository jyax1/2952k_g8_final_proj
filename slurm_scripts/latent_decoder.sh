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
    --results_path=${results_path}
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
results_path="/users/ysong135/Documents/action_extractor/results"
epoch=20
batch_size=256
horizon=2
motion=""
image_plus_motion=""
latent_dim=32
idm_model_name="latent_encoder_resnet_unet_cam['frontview_image']_emb['']_lat32_res18_vps2_fidmFalse_ffdmFalse_optsgd_lr0.001_mmt0.9__idm-1-4000.pth"
fdm_model_name="latent_encoder_resnet_unet_cam['frontview_image']_emb['']_lat32_res18_vps2_fidmFalse_ffdmFalse_optsgd_lr0.001_mmt0.9__fdm-1-4000.pth"
freeze_idm=""
freeze_fdm=""
vit_patch_size=16
resnet_layers_num=18

# Job 1: latent_decoder_mlp
architecture="latent_decoder_mlp"
train_only

# Job 2: latent_decoder_vit
# architecture="latent_decoder_vit"
# vit_patch_size=2
# train_only

architecture="latent_decoder_obs_conditioned_unet_mlp"
train_only

architecture="latent_decoder_obs_conditioned_unet_mlp"
freeze_idm="-fidm"
train_only


# Job 3: latent_decoder_aux_separate_unet_mlp
architecture="latent_decoder_aux_separate_unet_mlp"
train_only

freeze_idm=""
train_only

# freeze_fdm="-ffdm"
# train_only

# freeze_idm=""
# train_only

# architecture="latent_decoder_aux_separate_unet_vit"
# train_only

# freeze_idm="-fidm"
# train_only

# freeze_fdm="-ffdm"
# train_only

# freeze_idm=""
# train_only

# Job 4: latent_decoder_aux_combined_vit
# architecture="latent_decoder_aux_combined_vit"
# latent_dim=32
# vit_patch_size=2
# train_only