run()
{
jn=ACTIONEXTRACT_${architecture}_dp_${dataset_path}_lat_size${latent_size}_b_size${batch_size}_epochs${epochs}_${date}

export train_args=" 
-a=${architecture}
-dp=${datasets_path}
-ls=${latent_size}
-b=${batch_size}
-e=${epochs}
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

date=0812


architecture=direct_unet
datasets_path="/users/ysong135/scratch/datasets"
latent_size=4
batch_size=88
epochs=100
train_only

architecture=direct_unet
datasets_path="/users/ysong135/scratch/datasets"
latent_size=8
batch_size=88
epochs=100
train_only

architecture=direct_unet
datasets_path="/users/ysong135/scratch/datasets"
latent_size=16
batch_size=88
epochs=100
train_only

architecture=direct_unet
datasets_path="/users/ysong135/scratch/datasets"
latent_size=32
batch_size=88
epochs=100
train_only