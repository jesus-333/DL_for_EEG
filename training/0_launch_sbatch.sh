#!/bin/sh

# Author :  Alberto Zancanaro <alberto.zancanaro@uni.lu>

# I have developed this script to launch the training of specific models on a SLURM cluster

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Set bash to exit immediately if any command fails (non-zero exit status)
set -e

# Set seed with a random positive integer (+ 1 to avoid seed=0)
# Note that the seed is generated before without using the SLURM job ID. The reason 
# 1. This method can be used also for the non-SLURM execution of the script, e.g. on a local machine.
# 2. I need the seed to create the copy of the config files for the specific training run. Moving the copy inside the SLURM script would implies that queued jobs would access the same config starting file.
seed=($RANDOM + 1)
seed=($(((RANDOM << 15) | RANDOM )) + 1)

echo "SEED = ${seed}"
echo "====================================="

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

# Variable for the path to config. This name will be used to create the path to the corresponding config file inside config/starting_config/
# I decided to use this method because it's quicked to change only the variable name instead of changing part of the path.
filter="" # If empty, no filter will be used. Options are : lowpass, highpass, bandpass, bandstop
model_name="eegnet"
loss="nll" # Possible options are : cross_entropy, nll 
optimizer="adamw" # Possible options are :  adam, adamw, lbfgs, rmsprop, sgd
lr_scheduler="" # If empty, no lr scheduler will be used. Options are : cosine_annealing_lr, cosine_annealing_warm_restarts, cyclic_lr, exponential_lr, step_lr

# Slurm
partition="gpu" # l40s, hopper, gpu
qos="normal" # besteffort, iris-hopper, normal
mem="10G"
time="28:13:00"
output="./training/logs/${model_name}/%j_std_%x.txt"
error="./training/logs/${model_name}/%j_err_%x.txt"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Script that will be launched with sbatch. Selected based on the model name.
script_name="./scripts_sh_V2/${model_name}_cent.sh"

# Job name (modify as needed)
job_name="grokking_${model_name}_${seed}"

if [ $partition = "l40s" ] ; then
	qos="besteffort"
elif [ $partition = "hopper" ] ; then
	qos="iris-hopper"
elif [ $partition = "gpu" ] ; then
	qos="normal"
elif [ $partition = "batch" ] ; then
	qos="normal"
else 
	echo "INVALID PARTITION"
fi

# Path dataset, training, optimizer and lr scheduler config (file to copy)
path_dataset_config="./training/config/starting_config/dataset/dataset.toml"
path_model_training_config="./scripts_sh_V2/config/training.toml"
path_optimizer_config="./scripts_sh_V2/config/optimizer_${optimizer}.toml"
path_lr_scheduler_config="./scripts_sh_V2/config/lr_sched_${lr_scheduler}.toml"

# Path for model config (file to copy)
# path_model_config="./scripts_sh_V2/config/model_${model_name}_micro.toml"
path_model_config="./scripts_sh_V2/config/model_${model_name}_tiny.toml"
# path_model_config="./scripts_sh_V2/config/model_${model_name}.toml"

# Create the folder for the config files
# TODO Create a variable and use it also for the path
mkdir -p ./scripts_sh_V2/config/training_hpc/${job_name}/

# New locations for the config file
new_path_dataset_config="./scripts_sh_V2/config/training_hpc/${job_name}/dataset.toml"
new_path_training_config="./scripts_sh_V2/config/training_hpc/${job_name}/training.toml"
new_path_optimizer_config="./scripts_sh_V2/config/training_hpc/${job_name}/optimizer.toml"
new_path_lr_scheduler_config="./scripts_sh_V2/config/training_hpc/${job_name}/lr_scheduler".toml
new_path_model_config="./scripts_sh_V2/config/training_hpc/${job_name}/model.toml"

# Print seed and paths
# TODO finish this section
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "seed: ${seed}"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Submit job

sbatch  --job-name=${job_name}\
		--partition=${partition}\
		--qos=${qos}\
		--mem=${mem}\
		--time=${time}\
		--output=${output}\
		--error=${error}\
		${script_name}\
			${path_src}\
			${new_path_dataset_config}\
			${new_path_model_config}\
			${new_path_training_config}\
			${new_path_optimizer_config}\
			${new_path_lr_scheduler_config}\
			${path_to_idx_files}\
			${seed}\
