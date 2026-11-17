#!/bin/sh

# Author :  Alberto Zancanaro <alberto.zancanaro@uni.lu>

# This script is used to train the EEGNet model on different pooling layer on all the subjects of a moabb dataset

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# SLURM arguments. These are used when the script is submitted to a SLURM cluster using the `sbatch` command.

#SBATCH --job-name="train_eegnet"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --mem=13G
#SBATCH --time=0-01:00:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail 
#SBATCH --output=./training/logs/EEGNet/%j_std_%x.txt
#SBATCH --error=./training/logs/EEGNet/%j_err_%x.txt

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Load python environment

echo "---------------------------------------------------"
echo $CONDA_DEFAULT_ENV
echo "---------------------------------------------------"
conda activate jesus_hpc_2
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"
echo $CONDA_DEFAULT_ENV
echo "+++++++++++++++++++++++++++++++++++++++++++++++++++"

# These two lines can be removed if you have already installed jade
hatchling build
pip install -e . 

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Get arguments passed to the script.

path_dataset_config=$1
path_model_config=$2
path_training_config=$3

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Print arguments (for debugging purposes).

echo "Variable (train_pooling.sh)"
echo "path_dataset_config  : ${path_dataset_config}"
echo "path_model_config    : ${path_model_config}"
echo "path_training_config : ${path_training_config}"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# Launch the training script with the specified configurations.
python ./training/EEGNet/train_pooling.py\
	--path_dataset_config="${path_dataset_config}"\
	--path_model_config="${path_model_config}"\
	--path_training_config="${path_training_config}"\
