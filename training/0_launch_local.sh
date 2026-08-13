#!/bin/sh

# Author :  Alberto Zancanaro <alberto.zancanaro@uni.lu>

# I have developed this script to launch the training of specific models on your local machine 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

# Set bash to exit immediately if any command fails (non-zero exit status)
set -e

# Set seed with a random positive integer (+ 1 to avoid seed=0)
# Note that the seed is generated before without using the SLURM job ID. The reason 
# 1. This method can be used also for the non-SLURM execution of the script, e.g. on a local machine.
# 2. I need the seed to create the copy of the config files for the specific training run. Moving the copy inside the SLURM script would implies that queued jobs would access the same config starting file.
seed=($RANDOM + 1)
seed=($(((RANDOM << 15) | RANDOM )) + 1)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Settings

# Variable for the path to config. This name will be used to create the path to the corresponding config file inside config/starting_config/
# I decided to use this method because it's quicked to change only the variable name instead of changing part of the path.
filter="" # If empty, no filter will be used. Options are :  bandpass, bandstop, highpass, lowpass, notch
model_name="EEGNet"
training_config_name="training_clf" # Specify the name of the training config file to use (without the .toml extension). The config file must be located in ./training/config/starting_config/train/
loss="nll" # Possible options are : cross_entropy, nll 
optimizer="adamw" # Possible options are :  adam, adamw, lbfgs, rmsprop, sgd
lr_scheduler="" # If empty, no lr scheduler will be used. Options are : cosine_annealing_lr, cosine_annealing_warm_restarts, cyclic_lr, exponential_lr, step_lr
wandb_config_name="wandb" # If empty, no wandb will be used. Otherwise, specify the name of the wandb config file to use (without the .toml extension). The config file must be located in ./training/config/starting_config/train/

# Job name (modify as needed)
job_name="${model_name}_${seed}"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Script that will be launched with sbatch. Selected based on the model name.
script_name="./training/${model_name}/train_${model_name}.sh"

# Path dataset, training, optimizer and lr scheduler config (files to copy)
path_dataset_config="./training/config/starting_config/dataset/dataset.toml"
path_model_config="./training/config/starting_config/model/${model_name}.toml"
path_training_config="./training/config/starting_config/train/${training_config_name}.toml"
path_loss_config="./training/config/starting_config/train/loss/${loss}.toml"
path_optimizer_config="./training/config/starting_config/train/optimizer/${optimizer}.toml"
path_dataloader_config="" # Not needed because the batch size is specified in the training config file. These options are required only if you want to set also the other parameters of the dataloader, e.g. num_workers, pin_memory, etc.

# Update paths requiring conditional logic based on the variable name

# Filter settings
if [ -z "$filter" ]; then
	path_filter_config=""
else
	path_filter_config="./training/config/starting_config/dataset/filter/${filter}.toml"
fi

# lr scheduler settings
if [ -z "$lr_scheduler" ]; then
	path_lr_scheduler_config=""
else
	path_lr_scheduler_config="./training/config/starting_config/train/lr_scheduler/${lr_scheduler}.toml"
fi

# wandb settings
if [ -z "$wandb_config_name" ]; then
	path_wandb_config=""
else 
	path_wandb_config="./training/config/starting_config/train/${wandb_config_name}.toml"
fi

# Path to store the new training run config files
path_new_location="./training/config/training_config/local/${job_name}/"
mkdir -p ${path_new_location}

# Print seed and paths (mostly for debugging purposes)
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
echo "Variables (shell script)"
echo "seed                      : ${seed}"
echo "path_dataset_config       : ${path_dataset_config}"
echo "path_filter_config        : ${path_filter_config}"
echo "path_model_config         : ${path_model_config}"
echo "path_training_config      : ${path_training_config}"
echo "path_loss_config          : ${path_loss_config}"
echo "path_optimizer_config     : ${path_optimizer_config}"
echo "path_dataloader_config    : ${path_dataloader_config}"
echo "path_lr_scheduler_config  : ${path_lr_scheduler_config}"
echo "path_wandb_config         : ${path_wandb_config}"
echo "path_new_location         : ${path_new_location}"
echo "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

python ./training/0_prepare_config.py\
	--path_dataset_config=${path_dataset_config}\
	--path_filter_config=${path_filter_config}\
	--path_model_config=${path_model_config}\
	--path_training_config=${path_training_config}\
	--path_loss_config=${path_loss_config}\
	--path_optimizer_config=${path_optimizer_config}\
	--path_dataloader_config=${path_dataloader_config}\
	--path_lr_scheduler_config=${path_lr_scheduler_config}\
	--path_wandb_config=${path_wandb_config}\
	--path_output_folder=${path_new_location}\
	--seed=${seed}\

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# "Submit" job

sh ${script_name}\
	${path_dataset_config}\
	${path_model_config}\
	${path_training_config}\
	${seed}
