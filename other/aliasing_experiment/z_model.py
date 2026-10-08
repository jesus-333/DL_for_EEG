# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

import torch

from jade.model import EEGNet_Classifier
from jade.config import EEGNet_config

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class EEGNet_aliasing(EEGNet_Classifier) :

    def __init__(self, config : EEGNet_config) -> None :
        super().__init__(config)

        self.pooling_factor_1 = config.pooling.pooling_factor_1
        self.pooling_factor_2 = config.pooling.pooling_factor_2

    def compute_aliased_energy_block_1(self, x : torch.Tensor, fs : float) -> torch.Tensor :
        """
        Compute the aliased energy of the EEGNet model for the first block (temporal convolution + spatial convolution).

        Note that the pooling is effectively a downsampling of the signal, and therefore can cause aliasing.
        If the original sampling frequency is fs, after pooling the new sampling frequency is fs / pooling_factor_1, and, therefore, the Nyquist frequency is fs / (2 * pooling_factor_1).
        Frequencies above this value will be aliased.

        The final output is a tensor of shape (N, D) where N is the number of trials and D is the number of depth channels in the spatial convolution.
        N.b. Note that D is not the number of EEG Channels, but the number of depth channels in the spatial convolution. For example, if the spatial convolution has 8 depth channels, D will be 8.
        Unfortunately, there is a little bit of terminology overlap between EEG and deep learning.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (N, 1, C, T), where N is the number of trials, C is the number of channels, and T is the number of time samples.
        fs : float
            The original sampling frequency of the signal.

        Outputs
        -------
        aliased_energy_ratio : torch.Tensor
            A tensor of shape (N, D) where each element is the ratio of the aliased energy to the total energy for each trial and depth channel D.

        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Check if the model is in train or eval mode and set it to eval mode for the computation
        was_in_train_mode = self.training
        if was_in_train_mode : self.eval()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Aliased energy computation

        # Compute the output up to the the layer preceding the pooling layer of the spatial convolution
        x = self.eegnet.temporal_filter(x)
        x = self.eegnet.spatial_filter[0](x) # Convolution
        x = self.eegnet.spatial_filter[1](x) # BatchNorm
        x_before_pooling = self.eegnet.spatial_filter[2](x) # Activation

        # Compute the aliased energy ratio
        aliased_energy_ratio = self.compute_aliased_energy(x_before_pooling, fs, self.pooling_factor_1)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Reset to the original mode
        if was_in_train_mode : self.train()

        return aliased_energy_ratio

    def compute_aliased_energy_block_2(self, x : torch.Tensor, fs : float) -> torch.Tensor :
        """
        Very similar to compute_aliased_energy_block_1, but for the second block (separable convolution).
        """

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Check if the model is in train or eval mode and set it to eval mode for the computation
        was_in_train_mode = self.training
        if was_in_train_mode : self.eval()

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Aliased energy computation
        
        # Compute the output up to the the layer preceding the pooling layer of the separable convolution
        x = self.eegnet.temporal_filter(x)
        x = self.eegnet.spatial_filter(x)
        x = self.eegnet.separable_convolution[0](x) # Depthwise convolution
        x = self.eegnet.separable_convolution[1](x) # Pointwise convolution
        x = self.eegnet.separable_convolution[2](x) # BatchNorm
        x_before_pooling = self.eegnet.separable_convolution[3](x) # Activation

        # Compute the fs after the first pooling layer
        fs_after_pooling_1 = fs / self.pooling_factor_1

        # Compute the aliased energy ratio
        aliased_energy_ratio = self.compute_aliased_energy(x_before_pooling, fs_after_pooling_1, self.pooling_factor_2)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Reset to the original mode
        if was_in_train_mode : self.train()

        return aliased_energy_ratio

    def compute_aliased_energy(self, x : torch.Tensor, fs : float, downsampling_factor : int) -> torch.Tensor :
        """
        Compute the aliased energy of a signal given its spectrum and the downsampling factor.

        The aliased energy is computed as the magnitude of the squared energy in the aliased frequency bands vs the squared energy of the whole signal.

        The aliased bands are all the frequencies above fs/(2 * downsampling_factor), where downsampling_factor is the factor by which the signal is downsampled.

        The final output is a tensor of shape (N, D) where N is the number of trials and D is the number of depth channels in the spatial convolution.
        N.b. Note that D is not the number of EEG Channels, but the number of depth channels in the spatial convolution. For example, if the spatial convolution has 8 depth channels, D will be 8.
        Unfortunately, there is a little bit of terminology overlap between EEG and deep learning.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor of shape (N, 1, C, T), where N is the number of trials, C is the number of channels, and T is the number of time samples.
        fs : float
            The original sampling frequency of the signal.

        Outputs
        -------
        aliased_energy_ratio : torch.Tensor
            A tensor of shape (N, D) where each element is the ratio of the aliased energy to the total energy for each trial and depth channel D.
        """

        # Compute the spectrum of the signal before and after pooling
        spectrum_before_pooling = torch.fft.rfft(x, dim = -1)
        freqs = torch.fft.rfftfreq(x.shape[-1], d = 1 / fs)

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Compute the aliased energy

        energy_whole_signal = torch.sum(torch.abs(spectrum_before_pooling) ** 2, dim = -1)

        aliased_freqs_idx = torch.where(freqs > fs / (2 * downsampling_factor))[0]
        aliased_energy = torch.sum(torch.abs(spectrum_before_pooling[..., aliased_freqs_idx]) ** 2, dim = -1)

        aliased_energy_ratio = aliased_energy / energy_whole_signal

        return aliased_energy_ratio



