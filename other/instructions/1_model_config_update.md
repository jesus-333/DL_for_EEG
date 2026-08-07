I want to solve the absence of a proper model config. It's the only part of the codebase that still used the old dictionary configuration.

# Model config, EEGNet and 1d pooling

I already create a `model` submodule inside `config` and put an empty file for EEGNet configuration. `EEGNet_cfg.py`. 
From now on each model will have to have its own configuration file here. The name of the file should be `{model_name}_cfg.py`. The `cfg` suffix is added to clearly distinguish configuration files from those with the implementation of the model.
The config of each model MUST have its name inside (it will be used later). When the instance of the config class is created the parameters should be checked. Raise error if needed.

Your first task is to write the config for EEGNet (`EEGNet_cfg.py`, whit eventual check to the parameters, and then update the EEGNet implementation (`EEGNet.py`).

Then add the config for the pooling submodule (`pooling_1d.py`). Since pooling is intended to be multiple times inside a single model I'm okay with having a single config file for all the various pooling module.
You can ignore aEEGNet instead.

# Train function update

Then you have to slightly modify the implementation of the train functions.
I rename the previous `train` function to `train_model()`. Then I add a NEW train function called `train()`.

The definition of the new functions is 
```python
def train(training_config, model_config, train_dataset, validation_dataset = None, wandb_model_artifact = None)
```

The main difference is that this receives the model config and the model is created inside. 
I have decided to keep the old function (now renamed `train_model`) because maybe in the future I want a function that directly train a model received in input.
I already added the skeleton for the function.
The test function inside `test_functions.py` is ok as it is since its purpose is to test a model.

The model should be created through an ad-hoc function. You can put it inside the `__init__.py` of the `model` module or inside `support_models.py`. Weights which option is the best and ask me if you're not sure.
The method should receive in input an instance of the model config and return the proper model. The selection of which model create is based on the model name inside the config.

Add also a new function to `support_training`. It should be called `cross_check_config` and must be used to handle all the cross-checks between `training_config` and `model_config` (E.g. the loss function NLLLoss/Cross entropy vs the activation of the last layer of the EEGNet classifier).

# New cli function : `createa-default-config`

Lastly I want to add a new functionality to the `cli` module. I want a command called `createa-default-config`.
When launched this command will create, at the specified location, a folder called `default_config` with the default config saved in the `toml` format.
Note that not all the possible parameters inside the configs have default value. When default is not present use some reasonable value (e.g. `epochs`). 
The purpose of this cli function is not a config that can be used directly but a skeleton config with all the possible parameters that can be easily modified.

The possible parameters for this cli command should between
- `path` : specify the path where the `default_config` folder is created. If not used will default to current directory.
- `type` : specify which configs you want (`train`, `model`, `dataset`, `all`).If specified create only the default config for that specific type. If not specified default to `all`

## How to handle nested config

I will do the example with train config, but the same logic applies with every nested config (also in the future).

The complete train config should contain various subsections (`optimizer`, `lr_scheduler`, `loss`, `dataloader`, `wandb`).
Some of them can have multiple choices (e.g. `optimizer`). In this case create a separate config file with for each one of the options.
E.g. for the optimizers create `adam.toml`, `adamw.toml` etc etc.
Then in the main `training.toml` put only the section with some kind of comments like "this section could have multiple options. See the files..."

## How the cli function should work

I don't need, nor want, to have the information of all the `toml` file inside some python script. It's noisy and clumsy.
I have created a folder called `deafult_config` inside `src/jade/config/`.
Just create here the config `toml` file and let the `cli` function copied them. Also add comments inside the `toml` file to help understand what each parameter does.

If you think that this approach is wrong let me know, explain why and give me some suitable alternatives.

# Final remarks

VERY IMPORTANT. Update all the content inside `llm_wiki_jade` accordingly.
If you need more informations/instructions/clarifications let me know.
