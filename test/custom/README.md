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
