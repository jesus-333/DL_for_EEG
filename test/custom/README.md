Custom scripts not intended for `pytest` but to be run in a python terminal (e.g. `ipython`).

The purpose is to run the script and inspect/analyzed the created objects/data.
`pytest` should be used to ensure everything works/compiles without bugs or serious errors. These scripts are intended for hands-on testing/experimenting with jade's various features.

All this scripts required `jade` to be installed. First of all you need `hatchling`. You can easily install it through the command
```
pip install hatchiling
```

After that, to install `jade`, from the root of the repository (where the `src` folder is) run the following command.
```
hatchling build && pip install -e .
```

# List of files
- [`./moabb_dataset_download_and_visualization.py`](moabb_dataset_download_and_visualization.py) : 
    - Download a dataset inside the moabb colection
    - Create an istance of the `EEG_Dataset` class (see the file `dataset_time.py` inside `jade` source code) 
    - Visualize a random sample from the dataset
- [`./vEEGNet.py`](vEEGNet.py) :
    - Build a `vEEGNet` (the variational autoencoder) through `get_model`, for both latent mappings
    - Check that the reconstruction has **exactly** the shape of the input over a sweep of awkward geometries (a number of time samples that no pooling factor divides, a disabled pooling stage, a spatial kernel shorter than the number of channels)
    - Check the reparametrization trick, `reconstruct` / `generate`, the gradient flow with the ELBO, every `pooling_1d` operator, the config round-trip and the guards
- [`./vEEGNet_creation_and_training.py`](vEEGNet_creation_and_training.py) :
    - Create a `vEEGNet` and synthetic data of shape `[N x 1 x C x T]`
    - Train it for a few epochs with the ELBO and a linear KL warm-up (`jade.training.vae`)
    - Check that every term of the loss and both reconstruction metrics are tracked, that the KL weight ramps as expected, that the reconstruction error decreases and that the checkpoints are written
