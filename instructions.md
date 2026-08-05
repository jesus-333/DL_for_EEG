
After you have finish the rename let's move to the training implementation. I have copied an old submodule of another project that I have written that do exactly that. The code in inside `training_OLD/`. 
Take inspiration from that for the general logic of the code. 

Here some other details. 
- In future, I will probably use this codebase for both finetuning (classification) and pretraining (reconstruction or similar objective). For now, you have to implement only the classification part. 
- I have prepared two files `classifier.py` and `classifier_epoch.py`. The latter should implement a single epoch of the training process while the first the entire training cycle.
- The firm of the functions is specified below
- Keep the implementation of `wandb`. You can copy it from my code. If you have any suggestion to improve it let me know.
- For the `test()` function create and ad-hoc file called `test_functions.py`. I think you could copy my old version. It should be generic enough to work even here.
- Inside `config`, update `training.py` accordingly with all the parameters from the training function
- For `wandb` create a separate config class (`wandb_training.py`)
- Loss function, optimizer and lr scheduler should be fully configurable and their own dedicated subfolder inside `config` (`loss_functions/`, `lr_schedulers/`, `optimizers/`).
    - Inside each one of this subfolder there should be a config file for each of the implement loss function/optimizer/lr scheduler.
    - For now implement the config `Adam`, `AdamW`, `SGD`, the exponential decay lr scheduler (forget the name) and a classification loss function
    - Each config implemented should ask for the necessary parameters also requested by the torch class. All the non-necessary parameters should have the default value assigned by torch and be modified only if the user specifies them.
    - If you think that I should also have other configure already implemented let me know
- In general, substitutes the prints with a proper log.
- Copy/reimplement also the metrics file. Check if can be improved or if it is okay as it is.
- Be sure to keep the `support_training.py` for all the shared code across modules. Or for code that will made the other files to long/noisy/difficult to read.
- Keep the extensive comment style (see the `train_functions` inside `training_OLD/`)
- I also have uploaded another instruction file with guideline regarding code style (`coding_style_instructions_ENG.md`). Read it, and be sure to follow it.

As always if you have any doubt just let me know.

# Functions firm

## `classifier.py`
```python
def train(training_config, model, train_dataset, validation_dataset = None, wandb_model_artifact = None) :
```

Note tha `training_config` should be an istance of the `training_config` class.
Model is an istance of `torch.nn.Module`.

## `classifier_epoch.py`
```python
def train_epoch_function(model, train_loader, loss_function, optimizer, device, log_dict = None, print_var = True) :
```

