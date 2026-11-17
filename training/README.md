This folder contains scripts used to launch model training.

# List of files and folder
A graphical representation of the folder structure is shown below, in section [Folder structure](#folder-structure). The files and folders are described below.

- `0_launch_local.sh`: A shell script to launch training on a local machine.
- `0_launch_sbatch.sh`: A shell script to launch training on a hpc machine (or similar setup) that use the SLURM workload manager.
- `0_prepare_config.py`: A Python script to prepare the configuration files for a training run
- `README.md`: This file, which provides an overview of the training folder and its contents.
- `config/`: A folder containing configuration files for training runs. It contains two subfolders:
  - `starting_config`: This folder contains the starting configuration files for training runs. These files are readed by the `0_prepare_config.py` script to generate the final configuration files for a training run.
  - `training_config`: This folder contains the training configuration files for training runs. Each time a training run is launched, the `0_prepare_config.py` script generates a new configuration file in this folder, based on the starting configuration files and the parameters specified in the `0_launch_local.sh` or `0_launch_sbatch.sh` scripts. The reason for having a separate folder for training configuration files is to keep track of the different configurations used for different training runs, and to avoid overwriting the starting configuration files each time a new training run is launched.
- `{MODEL_NAME}/`: A folder containing the training scripts for a specific model. The name of the folder is a placeholder for the name of the model being trained. It will always contain at least two files:
  - `train_{MODEL_NAME}.py`: The main training script for the model. This script contains the code that defines the model architecture, loads the training data, and performs the training process.
  - `train_{MODEL_NAME}.sh`: A shell script that execute the python script. Initially developed to provide a shell script that can be launched with the `sbatch` command. Then, to keep `0_launch_sbatch.sh` and `0_launch_local.sh` scripts as simple as possible, this script is also used to launch the training on a local machine.
  - Other training files, specific to the model being trained, may be added to this folder as needed.

The `model_name` variable inside `0_launch_local.sh` and `0_launch_sbatch.sh` (and the other config names, e.g. `loss`, `optimizer`) is case insensitive : `model_name="eegnet"` finds the folder `EEGNet/`, the script `EEGNet/train_EEGNet.sh` and the config `config/starting_config/model/EEGNet.toml`, also on case-sensitive filesystems (e.g. Linux). The name written on disk is then used for the job name and the logs folder (`logs/{MODEL_NAME}/`).

# Folder structure

Summary of the folder structure. `{MODEL_NAME}` is a placeholder for the name of the model being trained. 

```
./
    0_launch_local.sh
    0_launch_sbatch.sh
    0_prepare_config.py
    README.md
    config/
        ├── starting_config
        └── training_config
    {MODEL_NAME}/
        ├── train_{MODEL_NAME}.py
        └── train_{MODEL_NAME}.sh
```
