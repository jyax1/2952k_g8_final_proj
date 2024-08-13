run()
{
jn=ACTIONEXTRACT_${architecture}_dp_${dataset_path}_lat_size${latent_size}_b_size${batch_size}_epochs${epochs}_${date}

parser.add_argument('--architecture', '-a', type=str, default='direct_unet', choices=['direct_unet'], help='Model architecture to train')
    parser.add_argument('--datasets_path', '-dp', type=str, default=dp, help='Path to the datasets')
    parser.add_argument('--latent_size', '-ls', type=int, default=16, help='Latent size')
    parser.add_argument('--epoch', '-e', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--batch_size', '-b', type=int, default=b, help='Batch size')

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


train_only