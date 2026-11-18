"""
Load the settings and the weights of a hvEEGNet training run and plot the reconstruction of a specific trial and channel.

The settings are read from the folder created by 0_prepare_config.py for the run (e.g. ./training/config/training_config/local/hvEEGNet_159692052/), which contains dataset.toml, model.toml and training.toml.
The data are loaded and scaled exactly as in ./training/hvEEGNet/train_hvEEGNet.py (same dataset, same scale factor and same train/validation split, recreated with the seed of the run).
The reconstruction error is computed with the reconstruction term used during training (the [loss] section of training.toml, e.g. SoftDTW or SoftDTW divergence), for the selected channel and for the whole trial.

Every argument is optional. If an argument is not passed, the value in the "Default settings" section is used, so the script can also be launched without arguments after editing that section.

Example (from the root of the repository) :
python ./analysis/hvEEGNet/reconstruct_signal.py --path_run_config ./training/config/training_config/local/hvEEGNet_495361435 --weights_file model_BEST.pth --split test --idx_trial 10 --channel C3

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Default settings. Used only if the corresponding argument is not provided when launching the script.

# Folder with the configs of the training run (created by 0_prepare_config.py)
default_path_run_config = './training/config/training_config/local/hvEEGNet_495361435'

# Weights of the trained model. If None, the weights are taken from the folder of the run (the path_to_save_model inside training.toml, e.g. model_weights/hvEEGNet_495361435/) and default_weights_file is used.
default_path_weights = None
default_weights_file = 'model_END.pth' # model_END.pth or model_BEST.pth

# Which set ("train", "validation" or "test"), which trial (index inside that set) and which channel (name, e.g. "C3", or index, e.g. 7)
default_split     = 'test'
default_idx_trial = 0
default_channel   = 'C3'

# Number of reconstructions. The reconstruction is stochastic (the latent variables are sampled), so with more than 1 the mean is plotted, with the band of +- one std.
default_n_reconstructions = 5

# If True, every latent variable is set to the mean of its distribution instead of being sampled, so the reconstruction is deterministic (the same at every run).
# In this case a single reconstruction is computed and default_n_reconstructions is ignored.
default_use_mean = False

# Same values used by train_hvEEGNet.py
default_scale_factor          = 1e6
default_percentage_validation = 0.1

# Plot settings
plot_settings = dict(
    figsize = (14, 6),
    fontsize = 14,
    save_plot = True,
    path_to_save_plot = './analysis/hvEEGNet/plots/',
    format_list = ['png'],
    show_plot = True,
)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# CLI Arguments

import argparse

parser = argparse.ArgumentParser(description = 'Reconstruct a trial and a channel with a trained hvEEGNet.')
parser.add_argument('--path_run_config'      , type = str  , default = default_path_run_config      , help = 'Folder with the configs of the training run (dataset.toml, model.toml, training.toml).')
parser.add_argument('--path_weights'         , type = str  , default = default_path_weights         , help = 'Path to the weights of the model. If not provided, the weights are taken from the path_to_save_model of the training config of the run (see --weights_file).')
parser.add_argument('--weights_file'         , type = str  , default = default_weights_file         , help = 'Weights file inside the folder of the run (e.g. model_END.pth or model_BEST.pth). Used only if --path_weights is not provided.')
parser.add_argument('--split'                , type = str  , default = default_split                , choices = ['train', 'validation', 'test'], help = 'Set from which the trial is taken.')
parser.add_argument('--idx_trial'            , type = int  , default = default_idx_trial            , help = 'Index of the trial inside the selected set.')
parser.add_argument('--channel'              , type = str  , default = default_channel              , help = 'Channel to plot, as name (e.g. C3) or as index (e.g. 7).')
parser.add_argument('--n_reconstructions'    , type = int  , default = default_n_reconstructions    , help = 'Number of (stochastic) reconstructions. If > 1, the mean and the std are plotted.')
parser.add_argument('--use_mean'             , action = argparse.BooleanOptionalAction, default = default_use_mean, help = 'Use the mean of the latent distributions instead of sampling them (deterministic reconstruction). If set, --n_reconstructions is ignored.')
parser.add_argument('--scale_factor'         , type = float, default = default_scale_factor         , help = 'Factor multiplied to the EEG data after loading. Must be the same used during training.')
parser.add_argument('--percentage_validation', type = float, default = default_percentage_validation, help = 'Fraction of the training set used for validation. Must be the same used during training.')
args = parser.parse_args()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Imports

import matplotlib.pyplot as plt
import numpy as np
import os
import toml
import torch

from jade.config import dataset_config, elbo_config, hvEEGNet_config
from jade.dataset import moabb_dataset
from jade.model import hvEEGNet
from jade.training.custom_loss_function import support_loss

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load configs

dataset_config_dict  = toml.load(os.path.join(args.path_run_config, 'dataset.toml'))
model_config_dict    = toml.load(os.path.join(args.path_run_config, 'model.toml'))
training_config_dict = toml.load(os.path.join(args.path_run_config, 'training.toml'))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load data (same steps of train_hvEEGNet.py)

dataset_config_obj = dataset_config.from_dict(dataset_config_dict)
eeg_dataset_train, eeg_dataset_test = moabb_dataset.get_dataset(dataset_config_obj)

eeg_dataset_train.data *= args.scale_factor
eeg_dataset_test.data  *= args.scale_factor

ch_list = eeg_dataset_test.ch_list
fs      = eeg_dataset_test.fs

# Recreate the train/validation split of the training run
if args.split == 'test' :
    eeg_dataset = eeg_dataset_test
elif args.percentage_validation > 0 :
    split_seed   = training_config_dict.get("seed", dataset_config_obj.seed)
    n_validation = int(len(eeg_dataset_train) * args.percentage_validation)
    n_train      = len(eeg_dataset_train) - n_validation
    eeg_dataset_train_split, eeg_dataset_validation = torch.utils.data.random_split(eeg_dataset_train, [n_train, n_validation], generator = torch.Generator().manual_seed(split_seed))
    eeg_dataset = eeg_dataset_train_split if args.split == 'train' else eeg_dataset_validation
elif args.split == 'train' :
    eeg_dataset = eeg_dataset_train
else :
    raise ValueError("With percentage_validation = 0 the test set was used as validation set. Use --split test")

if not 0 <= args.idx_trial < len(eeg_dataset) :
    raise ValueError(f"idx_trial must be in [0, {len(eeg_dataset) - 1}] for the {args.split} set. Current value : {args.idx_trial}")

x, label = eeg_dataset[args.idx_trial]
x = x.unsqueeze(0) # [1 x 1 x C x T]

# Get the index of the channel (the channel can be passed by name or by index)
if args.channel.isdigit() :
    idx_ch = int(args.channel)
elif ch_list is not None and args.channel in ch_list :
    idx_ch = ch_list.index(args.channel)
else :
    raise ValueError(f"Channel {args.channel} not found. Available channels : {ch_list}")
ch_name = ch_list[idx_ch] if ch_list is not None else str(idx_ch)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Load model (same config updates of train_hvEEGNet.py)

C, T = x.shape[2], x.shape[3]
model_config_dict["input_size"] = (1, 1, C, T)
model_config_dict["C"]          = C
model_config_dict["T"]          = T
model_config_dict["c_kernel_2"] = (C, 1)

model = hvEEGNet(hvEEGNet_config.from_dict(model_config_dict))
path_weights = args.path_weights if args.path_weights is not None else os.path.join(training_config_dict['path_to_save_model'], args.weights_file)
print(f"Weights loaded from : {path_weights}")
model.load_state_dict(torch.load(path_weights, map_location = 'cpu'))
model.eval()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Reconstruction

# With use_mean the reconstruction is deterministic, so one is enough
if args.use_mean : args.n_reconstructions = 1
x_r_full = [model.reconstruct(x, use_mean = args.use_mean) for _ in range(args.n_reconstructions)] # Each element is [1 x 1 x C x T]

# Reconstruction loss used during training (the [loss] section of the training config, e.g. SoftDTW with the same implementation, gamma and normalize)
loss_config_obj     = elbo_config.from_dict(training_config_dict['loss'])
reconstruction_loss = support_loss.get_reconstruction_loss_function(loss_config_obj.reconstruction_type, loss_config_obj)

# Loss of each reconstruction, for the selected channel and for the whole trial (all channels, i.e. the reconstruction term of the training loss for this trial)
# Note that the DTW is computed independently for every channel and then summed, so the value of the channel is its contribution to the value of the trial.
with torch.no_grad() :
    loss_channel = np.array([reconstruction_loss(x_r[:, :, idx_ch:idx_ch + 1], x[:, :, idx_ch:idx_ch + 1], loss_config_obj).item() for x_r in x_r_full])
    loss_trial   = np.array([reconstruction_loss(x_r, x, loss_config_obj).item() for x_r in x_r_full])

x_r_list = [x_r[0, 0, idx_ch].numpy() for x_r in x_r_full]
x_r_mean = np.mean(x_r_list, axis = 0)
x_r_std  = np.std(x_r_list, axis = 0)
x_orig   = x[0, 0, idx_ch].numpy()

loss_name = loss_config_obj.reconstruction_type + (" (divergence)" if loss_config_obj.reconstruction_type in ("soft_dtw", "block_dtw") and loss_config_obj.dtw_normalize else "")
print(f"Trial {args.idx_trial} ({args.split} set, label {int(label)}). Latent variables : {'mean' if args.use_mean else 'sampled'}. Reconstruction loss : {loss_name}" + (f" (mean +- std over {args.n_reconstructions} reconstructions)" if args.n_reconstructions > 1 else ""))
print(f"\tChannel {ch_name:<5} : {loss_channel.mean():.4g}" + (f" +- {loss_channel.std():.4g}" if args.n_reconstructions > 1 else ""))
print(f"\tWhole trial   : {loss_trial.mean():.4g}" + (f" +- {loss_trial.std():.4g}" if args.n_reconstructions > 1 else ""))

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# Plot

t = np.arange(T) / fs

plt.rcParams.update({'font.size': plot_settings['fontsize']})
fig, ax = plt.subplots(1, 1, figsize = plot_settings['figsize'])

ax.plot(t, x_orig, label = 'Original', color = 'black')
if args.use_mean                 : label_reconstructed = 'Reconstructed (latent mean)'
elif args.n_reconstructions == 1 : label_reconstructed = 'Reconstructed'
else                             : label_reconstructed = f'Reconstructed (mean of {args.n_reconstructions})'
ax.plot(t, x_r_mean, label = label_reconstructed, color = 'red')
if args.n_reconstructions > 1 :
    ax.fill_between(t, x_r_mean - x_r_std, x_r_mean + x_r_std, color = 'red', alpha = 0.25)

ax.set_xlabel('Time [s]' if fs != 1 else 'Time samples')
ax.set_ylabel('Amplitude [µV]' if args.scale_factor == 1e6 else 'Amplitude')
ax.set_xlim(t[0], t[-1])
ax.set_title(f'{os.path.basename(os.path.normpath(args.path_run_config))} - {args.split} trial {args.idx_trial} - channel {ch_name} - {loss_name} = {loss_channel.mean():.4g}')
ax.legend()
ax.grid(True)
fig.tight_layout()

if plot_settings['save_plot'] :
    os.makedirs(plot_settings['path_to_save_plot'], exist_ok = True)
    file_name = f"{os.path.basename(os.path.normpath(args.path_run_config))}_{args.split}_trial_{args.idx_trial}_ch_{ch_name}" + ("_latent_mean" if args.use_mean else "")
    for format in plot_settings['format_list'] :
        fig.savefig(os.path.join(plot_settings['path_to_save_plot'], f"{file_name}.{format}"), format = format)

if plot_settings['show_plot'] : plt.show()
