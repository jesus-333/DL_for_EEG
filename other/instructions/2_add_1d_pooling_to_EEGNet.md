I want to add the possibility to use the new 1d pooling layer inside EEGNet.

For this task consider that the tensors used by EEGNet have shape `[B x D x C x T]` with
- `B` : Batch size.
- `D` : Depth dimension.
- `C` : Number of EEG Channels.
- `T` : Number of time samples.

# EEGNet pooling config

Add a new section inside EEGNet config called `pooling`.

This sections should have the following (MANDATORY) parameters 
- `pooling_type` (string)
- `pooling_factor_1` (integer)
- `pooling_factor_2` (integer)
- pooling specific parameters

The parameter `pooling_type` should specify the type of pooling used. Can have as value the name (in lowercase) of the various 1d pooling layer or `deafault`.
`default` will use the current average pooling implementation with `AvgPool2d`
 
The parameters called `pooling_factor_1` and `pooling_factor_2` should indicate how much pooling is applied by each one of the two pooling layer inside EEGNet.
E.g. `pooling_factor_1 = 2` it means halving the number of samples in the first pooling layer. More generally a pooling factor of `M` it means that the new number of samples will be `T/M`

The pooling specific parameters should specify, as the name suggest, all other parameters for the various pooling 1d alternative.
They should be doubled (one for the first pooling layer and one for the second pooling layer).

# Applying 1D pooling to multi channel EEG signal

The 1d pooling layer at the moment work on 1d signal (tensor of shape `[B x D x T]`).
EEGNet required multichannels EEG signals (tensor of shape `[B x D x C x T]`). 1d pooling should be applied channel wise to this signals (i.e. to each channel independently)
Write a small extension for each one of the 1d pooling layer to apply it to this type of data. Call them `{pooling layer name}_multichannel`.
