#!/bin/sh

# Author :  Alberto Zancanaro <alberto.zancanaro@uni.lu>

# This script is used to train the hvEEGNet model. It is designed to be run on a SLURM cluster but can also be executed on a local machine.
# The script takes three arguments :
# 1. path_dataset_config 	: Path to the dataset configuration file (toml format).
# 2. path_model_config 		: Path to the model configuration file (toml format).
# 3. path_training_config 	: Path to the training configuration file (toml format).

# Note that the nested config (e.g. loss function config inside training config) should be already specified in the corresponding config file.
# Note also that the DTW reconstruction losses ("soft_dtw" and "block_dtw" in the elbo config) require the optional package dtw-loss-functions (pip install "jade[extra]").

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# SLURM arguments. These are used when the script is submitted to a SLURM cluster using the `sbatch` command.
# If you specify these arguments in the command line when submitting the job, they will override the default values specified here.
# Note that hvEEGNet with the SoftDTW is considerably slower than EEGNet, hence the longer time limit.

#SBATCH --job-name="train_hveegnet"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --partition=hopper
#SBATCH --qos=iris-hopper
#SBATCH --mem=13G
#SBATCH --time=0-06:00:00
#SBATCH --mail-user=alberto.zancanaro@uni.lu
#SBATCH --mail-type=end,fail
#SBATCH --output=./training/logs/hvEEGNet/%j_std_%x.txt
#SBATCH --error=./training/logs/hvEEGNet/%j_err_%x.txt

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

echo "path_dataset_config: ${path_dataset_config}"
echo "path_model_config: ${path_model_config}"
echo "path_training_config: ${path_training_config}"

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Launch the training script with the specified configurations.
python ./training/hvEEGNet/train_hvEEGNet.py\
	--path_dataset_config="${path_dataset_config}"\
	--path_model_config="${path_model_config}"\
	--path_training_config="${path_training_config}"\
