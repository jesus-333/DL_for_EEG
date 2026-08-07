I want to solve the absence of a proper model config. It's the only part of the codebase that still used the old dictionary configuration.

I already create a `model` submodule inside `config` and put an empty file for EEGNet configuration. `EEGNet_cfg.py`. 
From now on each model will have to have its own configuration file here. The name of the file should be `{model_name}_cfg.py`. The `cfg` suffix is added to clearly distinguish configuration files from those with the implementation of the model.
The config of each model MUST have its name inside (it will be used later). When the istance of the config class is created the parameters should be checked. Raise error if needed.

Your first task is to write the config for EEGNet (`EEGNet_cfg.py`, whit eventual check to the parameters, and then update the EEGNet implementation (`EEGNet.py`).

Then add the config for the pooling submodule (`pooling_1d.py`). Since pooling is intended to be multiple times inside a single model I'm okay with having a single config file for all the various pooling module.

Then you have to sligthly modify the implementation of the train functions.
I rename the previous `train` function to `train_model()`. Then I add a NEW train function called `train()`.

The firmature of the new functions is 
```python
def train(training_config, model_config, train_dataset, validation_dataset = None, wandb_model_artifact = None)
```

The main difference is that this receives the model config and the model is created inside. 
I have decided to keep the old function (now renamed `train_model`) because maybe in the future I want a function that train a model received in input

The model should be created through an ad-hoc function. You can put it inside the `__init__.py` of the `model` module or inside `support_models.py`. Weights which option is the best and ask me if you're not sure.
The method should receive in input an instance of the model config and return the proper model. The selection of which model create is based on the model name inside the config.

Add also a new function to `support_training`. It should be called `cross_check_config` and must be used to handle all the cross checks between `training_config` and `model_config` (E.g. the loss function NLLLoss/Cross entropy vs the activation of the last layer of the EEGNet classifier).

Update all the content inside `llm_wiki_jade` accordingly.
