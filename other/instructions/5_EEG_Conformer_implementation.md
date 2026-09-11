I want to implement the EEG Conformer model inside `jade`. The model is presented in the paper called `EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization` (https://ieeexplore.ieee.org/document/9991178)

# Material

I provide to you the original paper and implementation. You will find the files inside `other/material`.
- `other/material/conformer_ORIG.py` : Original implementation
- `other/material/eeg_conformer_paper.pdf` : Original paper

You can search other information online if you needed.

# Implementation

The model should be implemented inside `src/jade/model` in a module called `EEGConformer.py`. 
Our implementation should support the custom pooling layer implemented in `src/jade/model/pooling_1d`.
As for other stuff inside `jade` also this model should have its one config. Implement the config module inside `src/jade/config/model/`

You could check the implementation of `EEGNet` if you want the example of another model implementation.

You do not need to implement any training script. I will do myself in the future.

# Final remarks

The complete description of the codebase are inside the folder `llm_wiki_jade`. You can start from the file `0_wiki_quickstart.md` and then read the part you need for this task.
VERY IMPORTANT. Update all the content inside `llm_wiki_jade` accordingly.
If you need more informations/instructions/clarifications let me know.
