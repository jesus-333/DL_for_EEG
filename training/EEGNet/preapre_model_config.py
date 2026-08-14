"""
Ad hoc script that will prepare the model config for EEGNet.

What it does is to read the model config and the pooling config (if provided).
Then it insert the pooling config into the model config, check them with the config classes, and finally save the final model config in toml format.

Note that the script update the current model config, it does not create a new one from scratch.

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CLI Arguments

import argparse

parser = argparse.ArgumentParser(description = 'Prepare the model config for the EEGNet (classifier) in toml format')
parser.add_argument('--path_model_config'  , type = str  , default = None     , help = 'Path to the EEGNet model toml config file. If not provided, an error will be thrown.')
parser.add_argument('--pooling_type'       , type = str  , default = 'defualt', help = 'Pooling type to use. If not provided, the default pooling (average pooling) will be used.')
parser.add_argument('--path_pooling_config', type = str  , default = None     , help = 'Path to the pooling layers config. If you specify a non-default pooling type, you can provide its config file here, to update its hyperparameters (e.g. `gain` for the `blur` pooling). If not provided, the default values will be used.')

args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import toml

from jade.config import EEGNet_config
from jade.model.pooling_1d import POOLING_REGISTRY

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Read config files

# Read model config
if args.path_model_config is None:
    raise ValueError("You must provide a path to the model config file.")
else :
    eegnet_model_config_dict = toml.load(args.path_model_config)

# Check pooling type
if args.pooling_type != 'default' and args.pooling_type not in POOLING_REGISTRY:
    raise ValueError(f"Pooling type {args.pooling_type} is not supported. Supported pooling types are: {list(POOLING_REGISTRY.keys())}")

# Read pooling config
if args.path_pooling_config is None and args.pooling_type != 'default':
    pooling_config_dict = None
else :
    pooling_config_dict = toml.load(args.path_pooling_config)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Update model config with pooling config

# Update pooling type in model config
eegnet_model_config_dict['pooling']['pooling_type_1'] = args.pooling_type
eegnet_model_config_dict['pooling']['pooling_type_2'] = args.pooling_type

# (OPTIONAL) Update pooling config in model config
if pooling_config_dict is not None :
    eegnet_model_config_dict['pooling']['params_1'] = pooling_config_dict
    eegnet_model_config_dict['pooling']['params_2'] = pooling_config_dict

# Convert to config classes to check validity
eegnet_model_config_obj = EEGNet_config.from_dict(eegnet_model_config_dict)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Save updated model config

eegnet_model_config_obj.to_toml(args.path_model_config)
