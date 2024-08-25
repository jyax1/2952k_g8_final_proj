run()
{
    # flag indicators
    fidm_indicator=""
    ffdm_indicator=""

    if [ "$use_fidm" = true ]; then
        fidm_indicator="_fidm"
    fi
    
    if [ "$use_ffdm" = true ]; then
        ffdm_indicator="_ffdm"
    fi

    jn=ae_${architecture}_dp_${datasets_path}_ls${latent_dim}_bs${batch_size}_e${epochs}_m${motion}_ipm${ipm}${fidm_indicator}${ffdm_indicator}_${date}

    train_args=" 
    -a=${architecture}
    -dp=${datasets_path}
    -ld=${latent_dim}
    -b=${batch_size}
    -e=${epochs}
    -idm=${idm}
    -fdm=${fdm}
    "
    
    if [ "$use_fidm" = true ]; then
        train_args="${train_args} -fidm"
    fi
    
    if [ "$use_ffdm" = true ]; then
        train_args="${train_args} -ffdm"
    fi

    slurm_args=""

    jn1=${jn}_train
    #echo "sbatch ${slurm_args} --job-name=${jn1} ${train_script} ${train_args}"
    jobID_1=$(sbatch ${slurm_args} --job-name=${jn1} --export=ALL ${train_script} "${train_args}" | cut -f 4 -d' ')
}

train_only()
{
    train_script=slurm_scripts/train.sbatch
    run
}

date=0824
motion=False
ipm=False

# Define flag variables (set to true or false)
use_fidm=false
use_ffdm=false

# Example of submitting a job with no flags
architecture=latent_decoder_aux_vit
datasets_path="/users/ysong135/scratch/datasets"
latent_dim=32
batch_size=64
epochs=100
idm="idm_latent_cnn_unet_lat_32_m_False_ipm_False-60.pth"
fdm="fdm_latent_cnn_unet_lat_32_m_False_ipm_False-60.pth"
train_only

# Example of submitting a job with both flags
use_fidm=true
use_ffdm=true
train_only

# Example of submitting a job with only -fidm
use_fidm=true
use_ffdm=false
train_only

# Example of submitting a job with only -ffdm
use_fidm=false
use_ffdm=true
train_only