I want to add the possibility to use the new 1d pooling layer inside EEGNet.

For this task consider that the tensors used by EEGNet have shape `[B x D x C x T]` with
- `B` : Batch size.
- `D` : Depth dimension.
- `C` : Number of EEG Channels.
- `T` : Number of time samples.

# Applying 1D pooling to multi channel EEG signal

The 1d pooling layer at the moment work on 1d signal (tensor of shape `[B x D x T]`).
EEGNet required multichannels EEG signals (tensor of shape `[B x D x C x T]`). 1d pooling should be applied channel wise to this signals (i.e. to each channel independently)
Write a small extension for each one of the 1d pooling layer to apply it to this type of data. Call them `{pooling layer name}_multichannel`.
You can put them inside a new class in the corresponding 1d file, unless you have in mind a more elegant solution.

As for the direct implementation, for me, you could simply do a cycle that iterate over the various channels.
Otherwise, if you have a more optimized approach is ok for me.

# EEGNet pooling config and implementation

Add a new section inside EEGNet config called `pooling`.

This sections should have the following (MANDATORY) parameters 
- `pooling_type` (string)
- `pooling_factor_1` (integer)
- `pooling_factor_2` (integer)
- pooling specific parameters

The parameter `pooling_type` should specify the type of pooling used. It can have, as value, the name (in lowercase) of the various 1d pooling layer or `deafault`.
`default` will use the current average pooling implementation with `AvgPool2d`. Raise an error if it is an invalid value.
 
The parameters called `pooling_factor_1` and `pooling_factor_2` should indicate how much pooling is applied by each one of the two pooling layer inside EEGNet.
E.g. `pooling_factor_1 = 2` it means halving the number of samples in the first pooling layer. More generally a pooling factor of `M` it means that the new number of samples will be `T/M`
These value should be positive integer values. Negative values are allowed but means that the specific pooling layer is not applied. In this case print a warning when the EEGNet config dataclass is istantiated.
These new parameters should substitute the old `p_kernel_1` and `p_kernel_2`.

The pooling specific parameters should specify, as the name suggest, all other parameters for the various pooling 1d alternative.
They should be doubled (one for the first pooling layer and one for the second pooling layer).
If not specified go with default values (unless there are some parameters that NEED to be specified).

Update the EGGNet config class accordingly.

Inside the EEGNet code pooling should work in the following way. Create a function called `get_multichannel_pooling` inside `support_functions.py`
It must be an analogue of methods like `get_activation` and `get_dropout`.

The code inside EEGNet should be something similar
```python
pooling_layer_1 = support_functions.get_multichannel_pooling(config.pooling_layer_1_config)
pooling_layer_2 = support_functions.get_multichannel_pooling(config.pooling_layer_2_config)
```
And then you put these pooling layers inside the corresponding `nn.Sequential` blocks.

# Last extra task

Check all the files and subsitute the f-string in the format `f"The value of x is {}".format(x)` with the f-string in the format `f"The value of x is {x}`

# Final remarks

The complete description of the codebase are inside the folder `llm_wiki_jade`. You can start from the file `0_wiki_quickstart.md`
VERY IMPORTANT. Update all the content inside `llm_wiki_jade` accordingly.
If you need more informations/instructions/clarifications let me know.
