The core idea is to have a hierarchical encoder/decoder model that draws inspiration from hvEEGNet and LUNA. Both original papers are attached to the message.
All the Python code you see below should be interpreted more as pseudocode than proper code. It's just to give you a general idea of what I'm trying to do.

Consider that all input signals, `x`, will be EEG samples, with dimension `B x 1 x C x T`, where:
1) `B` : Batch size
2) `1` : Depth dimension (1 for all input signals)
3) `C` : Number of channels (can change depending on the sample)
4) `T` : Number of temporal signals (can vary depending on the sample)

Regardless of the variability of `C` and `T`, the signals must all be sampled at `256Hz` and have the same preprocessing (still to be decided but most likely some kind of filtering and a channel-by-channel z-score normalization, as in the LUNA paper).

From LUNA I want to take the idea of the channel encoder. Note get the idea/inspiration, not a direct copy. That is, to have a transformer-like architecture that can map an arbitrary number of channels into a fixed number `N` of channels (what they call the Channel Unification Module in the paper).
This must exist in both an encoder version (from original montage to `N` channels) and a decoder version (from `N` channels to original montage). I always want to keep something like the N learnable queries that the LUNA paper had in the Channel Unification Module.
So I'll need something like this:

```python
x_output = channel_encoder(x_input)

# x_input.shape  = [B x 1 x C x T] 
# x_output.shape = [B x 1 x N x T] 
```
For the channel encoding I will probably use a method similar to LUNA paper. Get the coordinate from some standard like the `mne standard 1020` and then map them in en embedding vector.
Note that I have not specified the internal architecture of `channel_encoder`. I will probably use some kind of patch division to reduce time dimension and then computed the attention between channel dimension and my set of arbitrary query.

Once this is done, the EEG signal must be processed by the rest of the hierarchical model.
This model must be composed of modules with the same structure that transform the signal. Each block must also perform a pooling, with a factor of 2. The idea is to start from signals sampled at `256Hz` and "assume" each pooling layer to be a downsampling that halves the frequency.
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

Starting from `256Hz` signals, I'm considering having the hierarchical model architecture composed of 4 modules:
1) 256Hz --> 128Hz
2) 128Hz --> 64Hz
3) 64Hz --> 32Hz
4) 32Hz --> 16Hz

The output of the last block must in turn be mapped into a latent space, but instead of using a CNN, a transformer (or another architecture independent of the temporal dimension) will be used.
This is to ensure that the deepest latent space always has the same dimension.
So I'll need something like this:

```python
x_output = last_hierarchical_block(x_input) 
x_latent_space = cnn_map_latent_space(x_output)

# x_input.shape         = [B x 1 x N x T/8] 
# x_output.shape        = [B x 1 x N x T/16] 
# x_latent_space.shape  = [B x 2 x L]  
```
Where `L` is the dimension of the latent space, chosen a priori. This is to ensure that the deepest latent space truly encodes information from the entire signal.

My basic idea for the `hierarchical_block` is to have something inspired by EEGNet.
```python
class hierarchical_block(torch.nn.Module) :
    def __init__(self, config : dict) :
        # Probably GELU or SELU
        self.activation = config["activate"]

        # Batch pooling and normalization still to decide how to implement them
        self.pooling = ...
        self.normalization = ... 
        
        # As in classic EEGNet
        # Kernel size will be probably smaller than the original hvEEGNet. I'm thinking of using a kernel with a size that is a multiple of 2 (16 or 32 probably) to keep the ratio nice
        self.temporal_filter = torch.nn.Conv2d(in_channels = 1, out_channels = config["temporal_filter_out_channels"], kernel_size = ... )

        # Basic idea : Have 1 spatial filter for each channel, get N output and stack them to get again N channels.
        self.spatial_filters_list = []
        for i in range(config['n_channels']) : # iterate over channels 
            self.spatial_filters_list.append(torch.nn.Conv2d(config["temporal_filter_out_channels"] = 1, out_channels = config["temporal_filter_out_channels"], kernel_size = N, groups = config["temporal_filter_out_channels"]  ... ))
        
        # It will be used to mix across different "Frequency" band
        # Eventually can have a more complex operation than the separable convolution of the classic EEGNet
        self.separable_conv = torch.nn.Sequential(
            ...
        )
        

    def forward(self, x_orig) :
        x = self.temporal_filter(x_orig)  # x.shape = [B x temporal_filter_out_channels x C x T]
        x = self.activation(x)       
        
        tmp_list = []
        for i in range(x.shape[2]) : # iterate over channels 
            tmp_x = self.spatial_filter_list[i](x)   # tmp_x.shape = [B x temporal_filter_out_channels x 1 x T]
            tmp_x = activation(tmp_x)
            tmp_list.append(tmp_x)

        # Probably it is written in the wrong way
        # I know it can be done more efficiently with a single CNN. As I said... this is just to give a general idea.
        x = torch.stack(tmp_list)   # x.shape = [B x temporal_filter_out_channels x N x T]

        x = separable_conv(x)   # x.shape = [B x temporal_filter_out_channels x N x T]
        
        # Residual connection. x_orig broadcasted to each depth dimension.
        x = x + x_orig          # x.shape = [B x temporal_filter_out_channels x N x T]

        x = pooling(x)          # x.shape = [B x temporal_filter_out_channels x N x T/2]

        return x
```

Essentially it's an EEGNet-like architecture where, however, I only have one pooling layer at the end and then I shift the axes to regain an output dimension consistent with the rest of the architecture.

For the decoder I was thinking of having a similar architecture but "flipped". Reverse the order of the layers and use upsampling layers instead of pooling.

For training, I'd like to do something pretraining-style, using various datasets of many different types of EEG. I was thinking of all the MOABB datasets. Plus maybe part of the TUEG
As a loss function, I was thinking of using the L1Smooth loss, more efficient than SoftDTW. Probably also an additional loss that take into consideration the frequency domain will be useful.

I'd like you to do an in-depth analysis of this idea, its strengths, and weaknesses.
Some points I'm already reflecting on myself:
1) The decoder as a "flip" of the encoder, a simple solution (already functional with hvEEGNet). However, I'm wondering if there are more elegant or efficient solutions.
2) The absence of any kind of temporal attention within each `hierarchical_block`.
3) How to handle TUEG bipolar montage if I use also that dataset.
4) I'm still very uncertain about how to implement the channel encoder.
5) How to implement masking
6) Modifying separable convolution to have all the depth dimension fused in a single one (i.e. the output of separable convolution will have shape `B x 1 x N x T`). This will also made the residual connection easier to implement (no broadcast in the sum)
7) Is it worth having the residual connection? And in case, there are better solution to implement it?

Obviously, other notes/weaknesses/improvements etc. are welcome.
