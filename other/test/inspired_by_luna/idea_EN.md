The core idea is to have a hierarchical encoder/decoder model that draws inspiration from hvEEGNet and LUNA. Both original papers are attached to the message.

Consider that all input signals, `x`, will be EEG samples, with dimension `B x 1 x C x T`, where:
1) `B` : Batch size
2) `1` : Depth dimension (1 for all input signals)
3) `C` : Number of channels (can change depending on the sample)
4) `T` : Number of temporal signals (can vary depending on the sample)

Regardless of the variability of `C` and `T`, the signals must all be sampled at 256Hz and have the same preprocessing (still to be decided but most likely some kind of filtering and a channel-by-channel z-score normalization, as in the LUNA paper).

From LUNA I want to take the idea of the channel encoder. That is, to have a transformer-like architecture that can map an arbitrary number of channels into a fixed number `N` of channels (what they call the Channel Unification Module in the paper).
This must exist in both an encoder version (from original montage to `N` channels) and a decoder version (from `N` channels to original montage). I always want to keep something like the N learnable queries that the LUNA paper had in the Channel Unification Module.
So I'll need something like this:

```python
x_output = channel_encoder(x_input)

# x_input.shape  = [B x 1 x C x T] 
# x_output.shape = [B x 1 x N x T] 
```

Once this is done, the EEG signal must be processed by the rest of the hierarchical model.
This model must be composed of modules with the same structure that transform the signal. Each block must also perform a pooling, with a factor of 2. The idea is to start from signals sampled at 256Hz and "assume" each pooling layer to be a downsampling that halves the frequency.
In this way, the deeper you go into the network, the more the features should relate to low frequencies and thus take into account increasingly wider portions of the signal.
I already know that pooling is not a true downsampling. I'm speaking at a more conceptual level about signal dimension and extracted features.
The output of each of these modules will also be mapped into a latent space, via a convolutional layer.
These latent spaces will have variable dimension since the model must be able to receive inputs of different lengths.
So I'll need something like this:

```python
x_output = hierarchical_block(x_input)
x_latent_space = cnn_map_latent_space(x_output)

# x_input.shape         = [B x 1 x N x T] 
# x_output.shape        = [B x 1 x N x T/2] 
# x_latent_space.shape  = [B x 2 x N x T/2]  
```
Note that the 2 in the latent space is because I need to have a pair of values, mean and std.

Starting from 256Hz signals, I'm considering having the hierarchical model architecture composed of 4 modules:
1) 256Hz --> 128Hz
2) 128Hz --> 64Hz
3) 64Hz --> 32Hz
4) 32Hz --> 16Hz

The output of the last block must in turn be mapped into a latent space, but instead of using a CNN, a transformer (or another architecture independent of the temporal dimension) will be used.
This is to ensure that the deepest latent space always has the same dimension.
The basic idea is quac
So I'll need something like this:

```python
x_output = last_hierarchical_block(x_input) 
x_latent_space = cnn_map_latent_space(x_output)

# x_input.shape         = [B x 1 x N x T/8] 
# x_output.shape        = [B x 1 x N x T/16] 
# x_latent_space.shape  = [B x 2 x L]  
```
Where `L` is the dimension of the latent space, chosen a priori. This is to ensure that the deepest latent space truly encodes information from the entire signal.

My basic idea for the `hierarchical_block` is to have something inspired by EEGNet. Note that this is more pseudocode than a real implementation, just to give a general idea.
```python
class hierarchical_block(torch.nn.Module) :
    def __init__(self, config : dict) :
        activation = config["activate"]

        temporal_filter = torch.nn.Conv2d(in_channels = 1, out_channels = config["temporal_filter_out_channels"], kernel_size = ... )
        spatial_filter  = torch.nn.Conv2d(config["temporal_filter_out_channels"] = 1, out_channels = config["temporal_filter_out_channels"], kernel_size = N, groups = config["temporal_filter_out_channels"]  ... )
        separable_conv = torch.nn.Sequential(
            ...
        )

        pooling = torch.nn.AvgPool2d(kernel_size = 2)

    def forward(self, x) :
        x = temporal_filter(x)  # x.shape = [B x temporal_filter_out_channels x C x T]
        x = activation(x)       
        x = spatial_filter(x)   # x.shape = [B x temporal_filter_out_channels x 1 x T]
        x = activation(x)
        x = separable_conv(x)   # x.shape = [B x temporal_filter_out_channels x 1 x T]
        x = pooling(x)          # x.shape = [B x temporal_filter_out_channels x 1 x T/2]
        
        # Reshape to move depth dimension to the channel dimension
        x = x.permute(0, 2, 1, 3) # x.shape = [B x 1 x temporal_filter_out_channels x T/2]

        return x
```

Essentially it's an EEGNet-like architecture where, however, I only have one pooling layer at the end and then I shift the axes to regain an output dimension consistent with the rest of the architecture.

For the decoder I was thinking of having a similar architecture but "flipped". Reverse the order of the layers and use upsampling layers instead of pooling. Possibly also transpose convolution instead of standard convolution.

For training, I'd like to do something pretraining-style, using various datasets of many different types of EEG. I was thinking of all the MOABB datasets plus part of the TUEG.
As a loss function, I was thinking of using the L1Smooth loss, more efficient than SoftDTW (the latter possibly to be tested if the model's performance is low).

I'd like you to do an in-depth analysis of this idea, its strengths and weaknesses.
Some points I'm already reflecting on myself:
1) Using the depth dimension as channels for subsequent layers. This is the easiest solution but I don't know how effective it is. Analysis and other proposals are welcome.
2) The decoder as a "flip" of the encoder, a simple solution (already functional with hvEEGNet). However, I'm wondering if there are more elegant or efficient solutions.
3) The absence of any kind of temporal attention within each `hierarchical_block`.
4) Possible modification of the second-to-last latent space as well, to have a fixed size.
5) I'm still very uncertain about how to implement the channel encoder.
6) Modifying the training using signals with different sampling rates (always, however, multiples or submultiples of 256Hz). This would require a modification of the architecture as presented above. The basic idea would be to inject the signal at the right point in the hierarchy plus possibly some kind of residual connection like ResNet. I already have it in mind but we can save the discussion for another time.

Obviously, other notes/weaknesses/improvements etc. are welcome.
