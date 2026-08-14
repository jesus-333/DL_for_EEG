"""
Prepare the config files for the training run.

It should receive the path to the following config files:
- dataset config.
    - filter config (optional), nested inside dataset config
- model config (with eventual nested config, e.g. pooling for EEGNet)
- training config
    - loss config (required), nested inside training config
    - optimizer config (required), nested inside training config
    - dataloader config (optional), nested inside training config
    - lr scheduler config (optional), nested inside training config
    - wandb config (optional), nested inside training config
For the optional configs, if not provided, the default config will be used (for dataloader) or no config will be used (for filter, lr scheduler and wandb).
If you launch this script from a shell script, you can pass the optional configs as empty strings, which will be converted to None.

Furthermore, the script must receive other two argument
- the path to a folder where the config files will be copied. The folder will be created if it does not exist.
- a random seed for reproducibility (optional). If provided it will saved inside the training config, otherwise no seed will be set.

The script will read the config files, convert them to jade config objects (see the config module) and create a copy in the specified folder.
Note that the conversion to jade config is used to validate the config files, since the jade config objects are implemented as python dataclasses with type/input validation.

To summarize, the script can receive the following arguments:
- --path_dataset_config      : path to the dataset config file (required)
- --path_filter_config       : path to the filter config file (optional)
- --path_model_config        : path to the model config file (required)
- --path_training_config     : path to the training config file (required)
- --path_loss_config         : path to the loss config file (required)
- --path_optimizer_config    : path to the optimizer config file (required)
- --path_dataloader_config   : path to the dataloader config file (optional)
- --path_lr_scheduler_config : path to the lr scheduler config file (optional)
- --path_wandb_config        : path to the wandb config file (optional)
- --path_output_folder       : path to the output folder where the config files will be copied (required)
- --seed                     : random seed for reproducibility (optional)


Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CLI Arguments

import argparse

# Declare the command line arguments
parser = argparse.ArgumentParser(description = 'Create the config (in toml format) for a training run')
parser.add_argument('--path_dataset_config'              , type = str  , default = None, help = 'Path to the dataset toml config file. If not provided, an error will be thrown.')
parser.add_argument('--path_filter_config'               , type = str  , default = None, help = 'Path to the filter toml config file. This is a nested config inside dataset config. If not provided, no filtering will be applied.')
parser.add_argument('--path_model_config'                , type = str  , default = None, help = 'Path to the model toml config file. If not provided, an error will be thrown. Note that model can have nested config, e.g. pooling for EEGNet, but, since these are model-specific, this script expects the model config to be already completed.')
parser.add_argument('--path_training_config'             , type = str  , default = None, help = 'Path to the training toml config file. If not provided, an error will be thrown.')
parser.add_argument('--path_loss_config'                 , type = str  , default = None, help = 'Path to the loss toml config file. This is a nested config inside training config. If not provided, an error will be thrown.')
parser.add_argument('--path_optimizer_config'            , type = str  , default = None, help = 'Path to the optimizer toml config file. This is a nested config inside training config and is required. If not provided, an error will be thrown.')
parser.add_argument('--path_dataloader_config'           , type = str  , default = None, help = 'Path to the dataloader toml config file. This is a nested config inside training config. If not provided, the default dataloader config will be used.')
parser.add_argument('--path_lr_scheduler_config'         , type = str  , default = None, help = 'Path to the lr scheduler toml config file. This is a nested config inside training config. If not provided, no lr scheduler will be used.')
parser.add_argument('--path_wandb_config'                , type = str  , default = None, help = 'Path to the wandb toml config file. This is a nested config inside training config. If not provided, no wandb logging will be used.')
parser.add_argument('--path_output_folder'               , type = str  , default = None, help = 'Path to the output folder where the config files will be copied. If not provided, an error will be thrown. The folder will be created if it does not exist.')
parser.add_argument('--seed'                             , type = int  , default = None, help = 'Random seed for reproducibility. If not provided, no seed will be set.')

# Parse the command line arguments
args = parser.parse_args()

# Convert to None empty strings. This is useful when the script is called from a shell script, where the arguments are passed as empty strings instead of None.
for arg in vars(args) :
    if getattr(args, arg) == '' :
        setattr(args, arg, None)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

# External libraries
import os
import toml
import pprint

# Internal imports
import jade.config as jade_cfg

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

print(f"\nPreparing config files for training run... (python script {__file__})\n")

# Create the output folder if it does not exist
os.makedirs(args.path_output_folder, exist_ok = True)

# Common function to read a toml config file and return it as a dictionary.
# If the file does not exist, an error is thrown if required is True, otherwise an empty dictionary is returned.
def read_toml_config(path_toml_file : str, required : bool) -> dict :
    """
    Read a toml config file and return it as a dictionary.

    Parameters
    ----------
    path_toml_file : str
        Path to the toml config file.
    required : bool
        If True, an error will be thrown if the file does not exist. If False, an empty dictionary will be returned.

    Returns
    -------
    dict
        The config as a dictionary.
    """
    
    if path_toml_file is None:
        if required:
            raise ValueError(f'Path to the toml config file is required but not provided.')
        else:
            return None
    
    if not os.path.exists(path_toml_file):
        if required:
            raise ValueError(f'The toml config file {path_toml_file} does not exist.')
        else:
            return None
    
    return toml.load(path_toml_file)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Dataset config

# Get the dataset config and the filter config (if provided) and nest the filter config inside the dataset config.
dataset_config_dict = read_toml_config(args.path_dataset_config, required = True)
filter_config_dict  = read_toml_config(args.path_filter_config, required = False)
dataset_config_dict['filter'] = filter_config_dict

# Check config
dataset_config_obj = jade_cfg.dataset_config.from_dict(dataset_config_dict)

# Save the config to the output folder
path_dataset_config_output = os.path.join(args.path_output_folder, 'dataset.toml')
dataset_config_obj.to_toml(path_dataset_config_output)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(f'Dataset config OK\nSaved to : {path_dataset_config_output}')
print(dataset_config_obj)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Model config

# Read the model config
model_config_dict = read_toml_config(args.path_model_config, required = True)

# Check config
# Note that the from_dict() method from the model_config class will automatically get the proper model_config subclass based on the model `name` field the config dictionary
# E.g. if `name` is "EEGNet", the from_dict() method will return an instance of EEGNet_config, which is a subclass of model_config.
model_config_obj = jade_cfg.model_config.from_dict(model_config_dict)

# Save the config to the output folder
path_model_config_output = os.path.join(args.path_output_folder, 'model.toml')
model_config_obj.to_toml(path_model_config_output)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(f'Model config OK\nSaved to : {path_model_config_output}')
print(model_config_obj)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Training config

# Get the training config and the loss, optimizer, lr scheduler and wandb configs
training_config_dict     = read_toml_config(args.path_training_config    , required = True)
loss_config_dict         = read_toml_config(args.path_loss_config        , required = True)
optimizer_config_dict    = read_toml_config(args.path_optimizer_config   , required = True)
dataloader_config_dict   = read_toml_config(args.path_dataloader_config  , required = False)
lr_scheduler_config_dict = read_toml_config(args.path_lr_scheduler_config, required = False)
wandb_config_dict        = read_toml_config(args.path_wandb_config       , required = False)

# Add nested configs to the training config
training_config_dict['loss']         = loss_config_dict
training_config_dict['optimizer']    = optimizer_config_dict
training_config_dict['dataloader']   = dataloader_config_dict
training_config_dict['lr_scheduler'] = lr_scheduler_config_dict
training_config_dict['wandb']        = wandb_config_dict

# If a seed is provided, add it to the training config
if args.seed is not None :  training_config_dict['seed'] = args.seed

# Check config
training_config_obj = jade_cfg.training_config.from_dict(training_config_dict)

# Save the config to the output folder
path_training_config_output = os.path.join(args.path_output_folder, 'training.toml')
training_config_obj.to_toml(path_training_config_output)
print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
print(f'Training config OK\nSaved to : {path_training_config_output}')
print(training_config_obj)
