
After you have finish the rename let's move to the training implementation. I have copied an old submodule of another project that I have written that do exactly that. The code in inside `training_OLD`. 
Take inspiration from that for the general logic of the code. 

Here some other details. 
- In future I will probably use this codebase for both finetuning (classification) and pretraining (reconstruction or similar objective). For now, you have to implement only the classification part. 
- I have prepara two files `classifier.py` and `classifier_epoch.py`. The latter should implement a single epoch of the training process while the first the entire training cycle.
- The firm of the functions is specified below
- Keep the implementation of `wandb`. You can copy it from my code. If you have any suggestion to improve it let me know.
- For the `test()` function create and ad-hoc file called `test_functions.py`. I think you could copy my old version. It should be generic enough to work even here.
- Inside `config`, create the `training.`
- Loss function, optimizer and lr scheduler should be fully configurable

# Functions firm

## `classifier.py`
```python
def train(training_config, model, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
```

Note tha `training_config` should be an istance of the `training_config` class

## `classifier_epoch.py`
```python
def train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict = None, print_var = True) :
```

