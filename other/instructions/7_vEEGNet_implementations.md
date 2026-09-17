# Objective

In this session you have access to two repository `DL_for_EEG` and `Variational-Autoencoder-for-EEG-analysis` and your objective is to implement the vEEGNet model inside `DL_for_EEG`

# Material

You have access to this source material
- The original vEEGNet paper (`./other/material/vEEGNet_paper.pdf`)
- The original source code of vEEGNet that I developed year ago. It is inside `Variational-Autoencoder-for-EEG-analysis`, in the folder `/library/model/vEEGNet.py`

# Implementation details

Note that, while functional, I no longer consider the original implementation up to my current standards.
So you do not have to simply copy and paste the original implementation.

What I want is a new implementation from scratch, fully integrated into the way the repository works and with adequate coding standards (style, syntax, structure, clarity, comments, etc).
The original vEEGNet implementation has two methods called `reconstruct` and `generate`. Also, the new implementation should have these methods.
Just ignore the reference to MBEEGNet in the original implementation. This new implementation should be based only on EEGNet. No multibranch structure.
As for anything else the class should have its own config dataclass inside the `config` module.

vEEGNet is a combination of VAE (a variational autoencoder) and a classifier. For now, you can avoid implementing the classification part and focus only on the generative side.
Since we're working with a VAE we will need to implement its loss function, the ELBO. This is therefore an excellent opportunity to implement a module for custom loss functions. The module should be located in `src/jade/training/custom_loss_function/`.
Also, the ELBO loss function should have its own config dataclass.


Remember to add the default config for both the vEEGNet and the ELBO in `./src/jade/config/default_config/`

For the training at the moment there is only the script to train a classifier. `./src/jade/training/classifier.py` and `./src/jade/training/classifier_epoch.py`
I need you to add also the code for training non-classifier model. You can call the two new file `vae.py` and `vae_epoch.py`
Currently `classifier.py` also has the code related to `wandb` training. Since most of the code is not dependent from the type of model, separate the code and put it in ad-hoc module.

# Final remarks

The complete description of the codebase are inside the folder `llm_wiki_jade`. You can start from the file `0_wiki_quickstart.md` and then read the part you need for this task.
VERY IMPORTANT. Update all the content inside `llm_wiki_jade` accordingly when you finish this task.
If you need more informations/instructions/clarifications let me know.
