run()
{
jn=ae_${architecture}_dp_${datasets_path}_ls${latent_dim}_bs${batch_size}_e${epochs}_m${motion}_ipm${ipm}_${date}

export train_args=" 
-a=${architecture}
-dp=${datasets_path}
-ld=${latent_dim}
-b=${batch_size}
-e=${epochs}
-idm=${idm}
"
slurm_args=""

jn1=${jn}_train
#echo "sbatch ${slurm_args} --job-name=${jn1} ${train_script} ${args}"
jobID_1=$(sbatch ${slurm_args} --job-name=${jn1} --export=ALL ${train_script} | cut -f 4 -d' ')
}



train_only()
{
train_script=slurm_scripts/train.sbatch
run
}

date=0823
motion=False
ipm=False


# architecture=latent_cnn_unet
# datasets_path="/users/ysong135/scratch/datasets"
# latent_dim=4
# batch_size=64
# epochs=100
# train_only

architecture=latent_decoder_vit
datasets_path="/users/ysong135/scratch/datasets"
latent_dim=8
batch_size=64
epochs=100
idm="idm_latent_cnn_unet_lat_32_m_False_ipm_False-20.pth"
train_only

# architecture=direct_cnn_vit
# datasets_path="/users/ysong135/scratch/datasets"
# latent_dim=16
# batch_size=64
# epochs=100
# train_only

# architecture=direct_cnn_vit
# datasets_path="/users/ysong135/scratch/datasets"
# latent_dim=32
# batch_size=64
# epochs=100
# train_only
