"""
Function to the creation the PyTorch dataset with EEG data in format channels x time samples

Authors
-------
Alberto Zancanaro <alberto.zancanaro@uni.lu>
"""

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Imports

from __future__ import annotations

# Full module imports
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

# Specific imports
from torch.utils.data import Dataset

# Internal imports
from jade.dataset.support_plot_freq import compute_frequency_spectrum

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% Plot configuration

# Extensions accepted when a figure is saved on disk.
VALID_PLOT_FORMATS = ('png', 'jpeg', 'pdf', 'eps')

def check_plot_config(plot_config : dict = None) -> dict :
    """
    Validate the plot configuration and fill the missing entries with their default value.

    If ``plot_config`` is ``None`` or an empty dictionary a brand new dictionary with every default value is returned.
    The two boolean flags (``add_title`` and ``save_plot``) also accept the strings ``'True'`` / ``'False'`` (case insensitive) which are converted to the corresponding boolean value.

    Parameters
    ----------
    plot_config : dict, optional
        Dictionary with (a subset of) the plot configuration keys.
        The recognized keys are:

        - ``figsize`` : tuple, the size of the figure. Default ``(12, 8)``.
        - ``fontsize`` : int, the font size used for labels, ticks and title. Default ``16``.
        - ``linewidth`` : float, the width of the plotted line. Default ``1.3``.
        - ``color`` : str, the color of the plotted line. Default ``'black'``.
        - ``x_lim`` : tuple or None, the limits of the x-axis. Default ``None`` (automatic).
        - ``y_lim`` : tuple or None, the limits of the y-axis. Default ``None`` (automatic).
        - ``add_title`` : bool, whether to add the title to the plot. Default ``True``.
        - ``title_prefix`` : str or None, the prefix used both for the title and for the file name. Default ``None``.
        - ``save_plot`` : bool, whether to save the figure on disk. Default ``False``.
        - ``plot_format`` : list of str, the extensions used to save the figure. Valid extensions are ``'png'``, ``'jpeg'``, ``'pdf'`` and ``'eps'``. Default ``['png']``.
        - ``plot_path`` : str, the folder where the figure is saved (created if it does not exist). Default ``'plot/'``.

    Returns
    -------
    checked_plot_config : dict
        A new dictionary (the input is never modified in place) with every key set to a valid value.
    """

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Default configuration

    default_plot_config = dict(
        figsize      = (12, 8),
        fontsize     = 16,
        linewidth    = 1.3,
        color        = 'black',
        x_lim        = None,
        y_lim        = None,
        add_title    = True,
        title_prefix = None,
        save_plot    = False,
        plot_format  = ['png'],
        plot_path    = 'plot/',
    )

    # If nothing (or an empty dict) is passed simply return a copy of the default configuration.
    if plot_config is None or len(plot_config) == 0 : return default_plot_config

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Merge the input with the default values

    # Work on a copy so the dictionary passed by the user is never modified.
    checked_plot_config = dict(plot_config)

    # Add the missing keys with their default value.
    # Note that x_lim, y_lim and title_prefix accept None as a valid value, so they are not overwritten when explicitly set to None.
    keys_that_allow_none = ('x_lim', 'y_lim', 'title_prefix')
    for key, default_value in default_plot_config.items() :
        if key not in checked_plot_config :
            checked_plot_config[key] = default_value
        elif checked_plot_config[key] is None and key not in keys_that_allow_none :
            checked_plot_config[key] = default_value

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Normalize and validate some values

    # Convert the two flags to real booleans (they may be passed as the strings 'True' / 'False').
    checked_plot_config['add_title'] = _to_bool(checked_plot_config['add_title'])
    checked_plot_config['save_plot'] = _to_bool(checked_plot_config['save_plot'])

    # Always work with a list of formats (a single string is wrapped in a list).
    if isinstance(checked_plot_config['plot_format'], str) :
        checked_plot_config['plot_format'] = [checked_plot_config['plot_format']]

    # Check that every requested extension is supported.
    for plot_format in checked_plot_config['plot_format'] :
        if plot_format not in VALID_PLOT_FORMATS :
            raise ValueError(f"Invalid plot format {repr(plot_format)}. Valid formats are {VALID_PLOT_FORMATS}.")

    return checked_plot_config

def _to_bool(value) -> bool :
    """
    Convert a value to a boolean, accepting also the strings ``'True'`` / ``'False'`` (case insensitive).

    Parameters
    ----------
    value : bool or str
        The value to convert.

    Returns
    -------
    bool
        The converted boolean.
    """

    if isinstance(value, str) :
        if value.strip().lower() == 'true' :
            return True
        elif value.strip().lower() == 'false' :
            return False
        else :
            raise ValueError(f"Cannot convert {repr(value)} to a boolean. Use a bool or the strings 'True'/'False'.")
    return bool(value)

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#%% PyTorch Dataset

class EEG_Dataset(Dataset):

    def __init__(self, data, labels, ch_list, normalize : int = -1, fs : float = 1):
        """
        PyTorch dataset wrapping EEG trials in the format channels x time samples.

        Parameters
        ----------
        data : numpy.ndarray
            The EEG data used for the dataset. Must have shape ``[Trials x 1 x channels x time samples]``.
            Note that for EEG data the depth dimension (the second axis) has value 1 (it is required by the Conv2d layers).
        labels : numpy.ndarray
            The integer class labels, one per trial (shape ``[Trials]``).
        ch_list : list of str or None
            The channel names, used by the plot methods to resolve a channel passed by name.
        normalize : int, default -1
            Normalization scheme applied at construction time. ``-1`` none, ``1`` min-max over the whole dataset, ``2`` per-trial per-channel, ``3`` per-trial.
        fs : float, default 1
            The sampling frequency (in Hz) of the signal. It is stored as ``self.fs`` and used by the plot methods (both to convert samples into ms and to label the frequency axis).
            The default ``1`` means a normalized frequency: each sample is treated as 1 ms and the frequency axis is labeled as normalized.
        """

        if len(data.shape) != 4 or data.shape[1] != 1 :
            raise ValueError(f"The input shape of data must be [Trials x 1 x channels x time samples]. Current shape {data.shape}")

        # Transform data in torch array
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).long()

        self.ch_list = ch_list

        # Sampling frequency (in Hz). Kept as an attribute so the plot methods do not need it passed again.
        self.fs = fs

        # (OPTIONAL) Normalize
        if normalize == 1:
            self.minmax_normalize_all_dataset(-1, 1)
        elif normalize == 2:
            self. normalize_trial_by_trial_channel_by_channel(-1, 1)
        elif normalize == 3:
            self. normalize_trial_by_trial(-1, 1)
            
    def __getitem__(self, idx : int) :
        return self.data[idx], self.labels[idx]
    
    def __len__(self) -> int :
        return len(self.labels)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    def minmax_normalize_all_dataset(self, a, b):
        """
        Normalize the entire dataset between the value a and the value b.
        """
        self.data = ((self.data - self.data.min()) / (self.data.max() - self.data.min())) * (b - a) + a

    def normalize_trial_by_trial_channel_by_channel(self, a, b):
        """
        Normalize each channel of each trial so the value are between a and b
        """
        
        # N.b. self.data.shape = [trials, 1 , channels, eeg samples]
        # The dimension with the 1 is needed because the conv2d require a 3d tensor input
        
        for i in range(self.data.shape[0]): # Cycle over EEG trials
            for j in range(self.data.shape[2]): # Cycle over channels
                tmp_ch = self.data[i, 0, j]
                
                normalize_ch = ((tmp_ch - tmp_ch.min()) / (tmp_ch.max() - tmp_ch.min())) * (b - a) + a
                
                self.data[i, 0, j] = normalize_ch

    def normalize_trial_by_trial(self, a, b):
        """
        Normalize each trial so the value are between a and b
        """
        
        # N.b. self.data.shape = [trials, 1 , channels, eeg samples]
        # The dimension with the 1 is needed because the conv2d require a 3d tensor input
        
        for i in range(self.data.shape[0]): # Cycle over EEG trials
            tmp_trial = self.data[i, 0, :]
            
            normalize_trial = ((tmp_trial - tmp_trial.min()) / (tmp_trial.max() - tmp_trial.min())) * (b - a) + a
            
            self.data[i, 0, :] = normalize_trial


    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Plot methods

    def plot_trial_time(self, idx_sample : int, idx_ch, plot_config : dict = None) :
        """
        Plot a single channel of a single trial in the time domain.

        The sampling frequency used to convert the sample index into ms is taken from ``self.fs`` (set at construction time).

        Parameters
        ----------
        idx_sample : int
            The index of the trial to plot.
        idx_ch : int or str
            The channel to plot. It can be either the integer index of the channel or the channel name (in which case ``self.ch_list`` is used to retrieve the corresponding integer index).
        plot_config : dict, optional
            The plot configuration. See :func:`~jade.dataset.dataset_time.check_plot_config` for the list of the accepted keys and their default value.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes
            The axis of the plot.
        """

        # Check the config and resolve the channel.
        plot_config = check_plot_config(plot_config)
        idx_ch_int, ch_label = self._resolve_channel(idx_ch)

        # Create the figure and draw the time signal.
        fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
        self._plot_time_on_ax(ax, idx_sample, idx_ch_int, plot_config = plot_config)

        # Add the title (if requested) and save the figure (if requested).
        if plot_config['add_title'] : ax.set_title(self._build_title(plot_config, idx_sample, ch_label), fontsize = plot_config['fontsize'])
        fig.tight_layout()
        self._save_figure(fig, plot_config, idx_sample, ch_label)

        return fig, ax

    def plot_trial_freq(self, idx_sample : int, idx_ch, plot_config : dict = None, freq_config : dict = None) :
        """
        Plot a single channel of a single trial in the frequency domain (power spectral density).

        The sampling frequency is taken from ``self.fs`` (set at construction time). When it is left at its default value of ``1`` the x-axis is labeled as a normalized frequency and the y-axis drops the ``/Hz`` unit.

        Parameters
        ----------
        idx_sample : int
            The index of the trial to plot.
        idx_ch : int or str
            The channel to plot. It can be either the integer index of the channel or the channel name (in which case ``self.ch_list`` is used to retrieve the corresponding integer index).
        plot_config : dict, optional
            The plot configuration. See :func:`~jade.dataset.dataset_time.check_plot_config` for the list of the accepted keys and their default value.
        freq_config : dict, optional
            The configuration of the time to frequency transformation. See :func:`~jade.dataset.support_plot_freq.check_freq_config` for the list of the accepted keys and their default value.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        ax : matplotlib.axes.Axes
            The axis of the plot.
        """

        # Check the config and resolve the channel.
        plot_config = check_plot_config(plot_config)
        idx_ch_int, ch_label = self._resolve_channel(idx_ch)

        # Create the figure and draw the frequency spectrum.
        fig, ax = plt.subplots(1, 1, figsize = plot_config['figsize'])
        self._plot_freq_on_ax(ax, idx_sample, idx_ch_int, plot_config = plot_config, freq_config = freq_config)

        # Add the title (if requested) and save the figure (if requested).
        if plot_config['add_title'] :
            ax.set_title(self._build_title(plot_config, idx_sample, ch_label), fontsize = plot_config['fontsize'])
        fig.tight_layout()
        self._save_figure(fig, plot_config, idx_sample, ch_label)

        return fig, ax

    def plot_trial_time_and_freq(self, idx_sample : int, idx_ch, plot_config : dict = None, freq_config : dict = None) :
        """
        Plot a single channel of a single trial in both the time and the frequency domain, side by side in a 1 x 2 layout.

        The sampling frequency is taken from ``self.fs`` (set at construction time) and used both for the time axis (samples to ms conversion) and for the frequency axis. See :meth:`plot_trial_freq` for the behavior when it is left at its default value.

        Parameters
        ----------
        idx_sample : int
            The index of the trial to plot.
        idx_ch : int or str
            The channel to plot. It can be either the integer index of the channel or the channel name (in which case ``self.ch_list`` is used to retrieve the corresponding integer index).
        plot_config : dict, optional
            The plot configuration. See :func:`~jade.dataset.dataset_time.check_plot_config` for the list of the accepted keys and their default value.
        freq_config : dict, optional
            The configuration of the time to frequency transformation. See :func:`~jade.dataset.support_plot_freq.check_freq_config` for the list of the accepted keys and their default value.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure.
        axs : numpy.ndarray of matplotlib.axes.Axes
            The two axes of the plot (time on the left, frequency on the right).
        """

        # Check the config and resolve the channel.
        plot_config = check_plot_config(plot_config)
        idx_ch_int, ch_label = self._resolve_channel(idx_ch)

        # Create the 1 x 2 figure and draw the two domains (the frequency plot uses the sampling frequency also on the time axis).
        fig, axs = plt.subplots(1, 2, figsize = plot_config['figsize'])
        self._plot_time_on_ax(axs[0], idx_sample, idx_ch_int, plot_config = plot_config)
        self._plot_freq_on_ax(axs[1], idx_sample, idx_ch_int, plot_config = plot_config, freq_config = freq_config)

        # Add a single title for the whole figure (if requested) and save it (if requested).
        if plot_config['add_title'] :
            fig.suptitle(self._build_title(plot_config, idx_sample, ch_label), fontsize = plot_config['fontsize'])
        fig.tight_layout()
        self._save_figure(fig, plot_config, idx_sample, ch_label)

        return fig, axs

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Plot helper methods

    def _resolve_channel(self, idx_ch) :
        """
        Convert a channel identifier (integer index or channel name) into its integer index and a human readable label.

        Parameters
        ----------
        idx_ch : int or str
            The channel identifier.

        Returns
        -------
        idx_ch_int : int
            The integer index of the channel.
        ch_label : int or str
            The label used for the title and the file name (the channel name when available, otherwise the integer index).
        """

        if isinstance(idx_ch, str) :
            # The channel is given by name: look for its position inside self.ch_list.
            if self.ch_list is None :
                raise ValueError("idx_ch was given as a channel name but self.ch_list is None, so the name cannot be resolved.")
            ch_list = list(self.ch_list)
            if idx_ch not in ch_list :
                raise ValueError("Channel {} is not present in self.ch_list.".format(repr(idx_ch)))
            return ch_list.index(idx_ch), idx_ch
        else :
            # The channel is given by index: keep it and use the channel name (when available) as label.
            idx_ch_int = int(idx_ch)
            ch_label = self.ch_list[idx_ch_int] if self.ch_list is not None else idx_ch_int
            return idx_ch_int, ch_label

    def _plot_time_on_ax(self, ax, idx_sample : int, idx_ch_int : int, plot_config : dict) :
        """
        Draw the time domain signal of a single channel of a single trial on a given axis.

        The sampling frequency (used to convert the sample index into ms) is read from ``self.fs``.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis where the signal is drawn.
        idx_sample : int
            The index of the trial.
        idx_ch_int : int
            The integer index of the channel.
        plot_config : dict
            The (already checked) plot configuration.
        """

        # Extract the signal (self.data has shape [Trials x 1 x channels x time samples]) and build the time axis in ms.
        signal = self.data[idx_sample, 0, idx_ch_int].detach().cpu().numpy()
        time_axis_ms = np.arange(signal.shape[0]) / self.fs * 1000

        # Draw the signal.
        ax.plot(time_axis_ms, signal, color = plot_config['color'], linewidth = plot_config['linewidth'])

        # Labels, grid, limits and ticks.
        ax.set_xlabel("Time [ms]", fontsize = plot_config['fontsize'])
        ax.set_ylabel(r"Amplitude [$\mu V$]", fontsize = plot_config['fontsize'])
        ax.grid(True)
        if plot_config['x_lim'] is not None : ax.set_xlim(plot_config['x_lim'])
        if plot_config['y_lim'] is not None : ax.set_ylim(plot_config['y_lim'])
        ax.tick_params(labelsize = plot_config['fontsize'])

    def _plot_freq_on_ax(self, ax, idx_sample : int, idx_ch_int : int, plot_config : dict, freq_config : dict) :
        """
        Draw the frequency domain (power spectral density) of a single channel of a single trial on a given axis.

        The sampling frequency is read from ``self.fs``. When it is left at the default value of 1 the axis are labeled as normalized.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axis where the spectrum is drawn.
        idx_sample : int
            The index of the trial.
        idx_ch_int : int
            The integer index of the channel.
        plot_config : dict
            The (already checked) plot configuration.
        freq_config : dict
            The configuration of the time to frequency transformation.
        """

        # Extract the signal and estimate its power spectral density.
        signal = self.data[idx_sample, 0, idx_ch_int].detach().cpu().numpy()
        freqs, psd = compute_frequency_spectrum(signal, fs = self.fs, freq_config = freq_config)

        # Draw the spectrum.
        ax.plot(freqs, psd, color = plot_config['color'], linewidth = plot_config['linewidth'])

        # The unit of the axis depends on whether the sampling frequency was actually specified (self.fs != 1).
        if self.fs != 1 :
            ax.set_xlabel("Frequency [Hz]", fontsize = plot_config['fontsize'])
            ax.set_ylabel(r"PSD [$\mu V^2 / Hz$]", fontsize = plot_config['fontsize'])
        else :
            # Normalized frequency: no Hz on the x-axis and no /Hz on the y-axis.
            ax.set_xlabel("Normalized frequency", fontsize = plot_config['fontsize'])
            ax.set_ylabel(r"PSD [$\mu V^2$]", fontsize = plot_config['fontsize'])

        # Grid, limits and ticks.
        ax.grid(True)
        if plot_config['x_lim'] is not None : ax.set_xlim(plot_config['x_lim'])
        if plot_config['y_lim'] is not None : ax.set_ylim(plot_config['y_lim'])
        ax.tick_params(labelsize = plot_config['fontsize'])

    def _build_title(self, plot_config : dict, idx_sample : int, ch_label) -> str :
        """
        Build the title of a plot with the format ``{title_prefix} - {idx_trial} - {idx_ch}``.

        When ``title_prefix`` is ``None`` the prefix (and the leading ``-``) is dropped.

        Parameters
        ----------
        plot_config : dict
            The (already checked) plot configuration.
        idx_sample : int
            The index of the trial.
        ch_label : int or str
            The label of the channel.

        Returns
        -------
        str
            The title of the plot.
        """

        if plot_config['title_prefix'] is not None :
            return "{} - {} - {}".format(plot_config['title_prefix'], idx_sample, ch_label)
        return "{} - {}".format(idx_sample, ch_label)

    def _save_figure(self, fig, plot_config : dict, idx_sample : int, ch_label) :
        """
        Save a figure on disk in every requested format.

        Nothing is done when ``plot_config['save_plot']`` is ``False``.
        The destination folder is created if it does not exist.
        The file name has the format ``{title_prefix}_{idx_trial}_{idx_ch}.{extension}``.
        When ``title_prefix`` is ``None`` the prefix (and the leading ``_``) is dropped.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            The figure to save.
        plot_config : dict
            The (already checked) plot configuration.
        idx_sample : int
            The index of the trial.
        ch_label : int or str
            The label of the channel.
        """

        if not plot_config['save_plot'] :
            return

        # Create the destination folder if needed.
        os.makedirs(plot_config['plot_path'], exist_ok = True)

        # The file name uses the title_prefix (dropped, together with its leading '_', when it is None).
        if plot_config['title_prefix'] is not None :
            file_name = "{}_{}_{}".format(plot_config['title_prefix'], idx_sample, ch_label)
        else :
            file_name = "{}_{}".format(idx_sample, ch_label)

        # Save the figure in every requested (and valid) format.
        for plot_format in plot_config['plot_format'] :
            full_path = os.path.join(plot_config['plot_path'], "{}.{}".format(file_name, plot_format))
            fig.savefig(full_path, format = plot_format, bbox_inches = 'tight')

class EEG_Dataset_ChWi(Dataset):

    def __init__(self, data, labels = None, ch_list : list = None, norm_type : int = -1):
        """
        Dataset used for the ChWi autoencoder. All EEG signals are saved independently from the trial.
        This means that an EEG matrix of shape B x C x T will be saved as a matrix of shape (B * C) x 1 x T
        The dimension of size 1 is used for the depth map of 1d convolution.

        @param data: data used for the dataset. Must have shape [Trials x C x T] with C = channels and T = time samples
        """

        # Get data
        if 'torch' in str(type(data)) :
            self.data = data.float()
        else :
            self.data = torch.from_numpy(data).float()

        # Reshape matrix
        self.data = self.data.reshape(-1, 1, self.data.shape[2])

        # (OPTIONAL) Normalize the data
        if norm_type > 0 : self.__normalize(norm_type)

        # Saved labels
        if labels is None :
            # If no labels is passed an array of ones is created. This is only used to keep dataset consistent with training functions
            self.labels = torch.ones(self.data.shape[0])
        else :
            if 'torch' in str(type(data)) :
                self.labels = labels.long()
            else :
                self.labels = torch.from_numpy(labels).long()

        # Saved channels
        self.ch_list = ch_list

        # Used to return only data. False by default. It can be set to True after dataset creation in the code.
        self.return_only_data = False
        
    def __getitem__(self, idx : int):
        if self.return_only_data :
            return self.data[idx]
        else :
            return self.data[idx], self.labels[idx]
    
    def __len__(self) -> int :
        return self.data.shape[0]

    def __normalize(self, norm_type : int) :
        """
        Normalize every EEG signal. The type of normalization is decided by the norm_type parameter.

        @param norm_type : (int) Decide the type of normalization. 1 = Minmax (everything between 0 and 1). 2 = Standardization (mean 0 and std 1)
        """
        for i in range(self.data.shape[0]) :
            tmp_eeg = self.data[i, 0]

            if norm_type == 1 : # Minmax
                self.data[i, 0] = (tmp_eeg - tmp_eeg.min()) / (tmp_eeg.max() - tmp_eeg.min())
            elif norm_type == 2 : # Standardization
                self.data[i, 0] = (tmp_eeg - tmp_eeg.mean()) / tmp_eeg.std()
            else :
                raise ValueError("norm_type must have values 1 (minmax), 2 (standardization). Current value is {}".format(norm_type))

#%% End file
