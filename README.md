# DL_for_EEG
Personal repository with experiment on DL for EEG

The purpose of this repository is twofold. On one hand, it contains the Jade package I am developing. On the other, it will be used to conduct experiments on EEG data using Jade.

# Jade

Jade is an acronym that stands for "Just Another Deep learning EEG package". 

Why did I create this package? Well, mainly for three reasons.
1) It's fun and satisfying.
2) I like having well-structured, easy-to-use code (even if no one else ever uses it). I also have other projects that follow this philosophy.
3) Since it started as a side project, this was one of my two "testing grounds" for AI-assisted code development.

The package is structured in modules. Currently, the modules available are:
- `cli` : It contains the code for CLI utilities implemented within the package.
- `config` : Contains the implementation of the config code. See information below.
- `dataset` : Contains the code related to data download, preprocess, dataset creation etc.
- `model` : Contain models implementation.
- `traiing` : Contain train related code.

## Note on the config module

In the past, when developing similar codebases, I typically managed the configuration of the various components (models, datasets, training, etc.) using parameter dictionaries. For instance, the training function would take a single dictionary as input containing all the parameters that influenced the training process. This allowed me to keep the function signatures reasonably clean, but using simple dictionaries was still "crude".

For this project, I decided to step things up and create custom data classes to save the various configurations. This choice offers me several benefits.
- Each dataclass is extensively documented in its docstring, effectively providing a complete description of the corresponding functionality. For example, the docstring for the training configuration dataclass contains descriptions of every possible parameter that can be used during training. Furthermore, the docstrings have been written to be compatible with the Sphinx documentation tool.
- For each dataclass I have implemented parameters checks when it is instantiated. This should reduce the possiblities of wrong configuraions.
- Each dataclass inherits from a template class that provides some common methods. Specifically, each class inherits methods for reading and writing configurations from TOML and JSON files. This allows you to write your own configuration file in your preferred format and use it immediately within Jade.
- I find the codebase more structured and easy to mantain. If something is missing its configuration dataclass it means that it's not ready.

# EEG Experiments

😉

<!-- [DTW Loss functions](https://github.com/jesus-333/dtw_loss_functions) -->
<!-- [Deep Learning for Alzheimer Detection](https://github.com/jesus-333/DL_for_AD_detection) -->
<!-- [Jesus LLMs Tools](https://github.com/jesus-333/JLT/tree/LLM-Development) -->
